! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_river_inflow
!
!> \brief MPAS ocean prescribed river inflow module
!> \details
!>  This module applies a prescribed, time-varying volume flux through a
!>  small set of named mesh edges, for representing river discharge into
!>  the ocean without a global-resolution forcing dataset.
!>
!>  The edges are named by global edge ID in the river inflow forcing
!>  file. Those fields are dimensioned on nRiverInflowEdges, which is not
!>  a decomposed dimension, so every MPI task holds the full forcing
!>  array and simply picks out the edges present in its own block. No
!>  communication is needed, and halo edges resolve locally.
!>
!>  A river inflow edge is usually a culled land-boundary edge, which has
!>  no velocity degree of freedom (edgeMask == 0, maxLevelEdgeTop == 0)
!>  and is skipped by every continuity loop in the dycore. The flux is
!>  therefore injected as an explicit divergence source term for the
!>  adjacent ocean cell, which is numerically equivalent to a flux
!>  boundary condition. The same mechanism works for interior edges.
!>
!>  Structured after ocn_tidal_forcing: an array is built once per
!>  timestep by ocn_river_inflow_build_array, then added to the thickness
!>  tendency by ocn_river_inflow_layer_thickness.
!
!-----------------------------------------------------------------------

module ocn_river_inflow

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_sort
   use mpas_timer
   use mpas_timekeeping
   use mpas_log, only: mpas_log_write
   use ocn_constants
   use ocn_config
   use ocn_mesh
   use ocn_diagnostics_variables, only: daysSinceStartOfSim

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_river_inflow_init, &
             ocn_river_inflow_build_array, &
             ocn_river_inflow_layer_thickness, &
             ocn_river_inflow_vel_tend, &
             ocn_river_inflow_active

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: riverInflowOn      !< master on/off switch for this module

   logical :: distributeUniform  !< .true.  for 'uniform' vertical distribution
                                 !< .false. for 'surface'

   integer :: nRiverEdgesLocal   !< number of river inflow edges present in
                                 !< this block (owned + halo)

   integer, dimension(:), allocatable :: &
      riverEdgeLocalIndex, &     !< local edge index of each local river edge
      riverEdgeDataIndex,  &     !< index into the (global) forcing array
      riverEdgeCellPos,    &     !< cell receiving a positive flux, 0 if that
                                 !< side is land or outside this block
      riverEdgeCellNeg           !< cell receiving a negative flux, 0 if that
                                 !< side is land or outside this block

   logical :: momentumOn         !< inflow also carries momentum
   logical :: velocityFromFile   !< inflow speed is read, not derived from Q

   real (kind=RKIND) :: momentumScaling !< scale factor on the momentum source
   real (kind=RKIND) :: maxInflowSpeed  !< upper bound on the inflow speed
   real (kind=RKIND) :: dtSeconds       !< model timestep, which bounds the
                                        !< momentum relaxation rate

   integer :: nRiverCellsLocal   !< number of distinct cells receiving inflow

   integer, dimension(:), allocatable :: &
      riverCellList              !< the distinct receiving cells, rebuilt each
                                 !< timestep because which cell receives the
                                 !< inflow depends on the sign of the flux

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_river_inflow_init
!
!> \brief   Initialize the prescribed river inflow module
!> \details
!>  Matches the global edge IDs listed in the river inflow forcing file
!>  against the edges present in this block, and resolves which ocean
!>  cell receives the inflow for each sign of the prescribed flux.
!>
!>  Must be called after the 'river_inflow' stream has been read.
!
!-----------------------------------------------------------------------

   subroutine ocn_river_inflow_init(domain, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain !< Input: domain

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), pointer :: timeVaryingForcingPool

      integer, dimension(:), pointer :: riverInflowEdgeGlobalID

      integer, pointer :: nRiverInflowEdges

      ! sorted copy of the global ID list, carrying the original position
      ! in the forcing array as a second row so the data index survives
      ! the sort
      integer, dimension(:,:), allocatable :: sortedID

      integer :: iRiver, iEdge, iLocal, iData
      integer :: cell1, cell2
      integer :: nMatchedOwned, nMatchedGlobal

      type (MPAS_TimeInterval_type) :: timeStep

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0

      riverInflowOn    = .false.
      nRiverEdgesLocal = 0
      nRiverCellsLocal = 0
      momentumOn       = .false.
      velocityFromFile = .false.
      momentumScaling  = 0.0_RKIND
      maxInflowSpeed   = 0.0_RKIND
      dtSeconds        = 0.0_RKIND

      if (.not. config_use_river_inflow_forcing) return

      select case (trim(config_river_inflow_vertical_distribution))
         case ('uniform')
            distributeUniform = .true.
         case ('surface')
            distributeUniform = .false.
         case default
            call mpas_log_write( &
               'ocn_river_inflow_init: unknown ' // &
               'config_river_inflow_vertical_distribution: ' // &
               trim(config_river_inflow_vertical_distribution), MPAS_LOG_CRIT)
            err = 1
            return
      end select

      momentumOn      = config_use_river_inflow_momentum
      momentumScaling = config_river_inflow_momentum_scaling
      maxInflowSpeed  = config_river_inflow_max_speed

      if (config_river_inflow_ramp < 0.0_RKIND .or. &
          config_river_inflow_ramp_delay < 0.0_RKIND) then
         call mpas_log_write('ocn_river_inflow_init: ' // &
            'config_river_inflow_ramp and config_river_inflow_ramp_delay ' // &
            'must be non-negative', MPAS_LOG_CRIT)
         err = 1
         return
      end if

      if (momentumOn) then
         if (maxInflowSpeed <= 0.0_RKIND) then
            call mpas_log_write('ocn_river_inflow_init: ' // &
               'config_river_inflow_max_speed must be positive', MPAS_LOG_CRIT)
            err = 1
            return
         end if

         call mpas_set_timeInterval(timeStep, timeString=config_dt)
         call mpas_get_timeInterval(timeStep, dt=dtSeconds)

         select case (trim(config_river_inflow_velocity_source))
            case ('derived')
               velocityFromFile = .false.
            case ('file')
               velocityFromFile = .true.
            case default
               call mpas_log_write( &
                  'ocn_river_inflow_init: unknown ' // &
                  'config_river_inflow_velocity_source: ' // &
                  trim(config_river_inflow_velocity_source), MPAS_LOG_CRIT)
               err = 1
               return
         end select
      end if

      call mpas_pool_get_subpool(domain % blocklist % structs, &
                                 'timeVaryingForcing', timeVaryingForcingPool)
      call mpas_pool_get_dimension(domain % blocklist % dimensions, &
                                   'nRiverInflowEdges', nRiverInflowEdges)
      call mpas_pool_get_array(timeVaryingForcingPool, &
                               'riverInflowEdgeGlobalID', riverInflowEdgeGlobalID)

      if (.not. associated(nRiverInflowEdges)) then
         call mpas_log_write('ocn_river_inflow_init: dimension ' // &
            'nRiverInflowEdges was not read. It must be present in the ' // &
            'river inflow forcing file.', MPAS_LOG_CRIT)
         err = 1
         return
      end if

      if (.not. associated(riverInflowEdgeGlobalID)) then
         call mpas_log_write('ocn_river_inflow_init: riverInflowEdgeGlobalID ' // &
            'is not available. It must be present in the river inflow ' // &
            'forcing file, read by the ''river_inflow'' stream.', MPAS_LOG_CRIT)
         err = 1
         return
      end if

      if (nRiverInflowEdges < 1) then
         call mpas_log_write('ocn_river_inflow_init: nRiverInflowEdges ' // &
            'is $i, but config_use_river_inflow_forcing is .true.', &
            MPAS_LOG_CRIT, intArgs=(/nRiverInflowEdges/))
         err = 1
         return
      end if

      !*** Sort the global IDs so each local edge can be looked up with a
      !*** binary search. Row 2 carries the position in the forcing array.

      allocate(sortedID(2, nRiverInflowEdges))
      do iRiver = 1, nRiverInflowEdges
         sortedID(1, iRiver) = riverInflowEdgeGlobalID(iRiver)
         sortedID(2, iRiver) = iRiver
      end do
      call mpas_quicksort(nRiverInflowEdges, sortedID)

      !*** First pass: count the river edges present in this block, over
      !*** owned and halo edges alike. Halo edges are included so that the
      !*** receiving cell is correct even when it is a halo cell.

      do iEdge = 1, nEdgesAll
         if (riverDataIndex(sortedID, nRiverInflowEdges, &
                            indexToEdgeID(iEdge)) > 0) then
            if (localOceanCell(cellsOnEdge(1, iEdge)) /= 0 .or. &
                localOceanCell(cellsOnEdge(2, iEdge)) /= 0) then
               nRiverEdgesLocal = nRiverEdgesLocal + 1
            end if
         end if
      end do

      allocate(riverEdgeLocalIndex(nRiverEdgesLocal), &
               riverEdgeDataIndex (nRiverEdgesLocal), &
               riverEdgeCellPos   (nRiverEdgesLocal), &
               riverEdgeCellNeg   (nRiverEdgesLocal), &
               riverCellList      (nRiverEdgesLocal))

      !*** Second pass: record the edges and resolve the receiving cells.

      iLocal        = 0
      nMatchedOwned = 0

      do iEdge = 1, nEdgesAll

         iData = riverDataIndex(sortedID, nRiverInflowEdges, &
                                indexToEdgeID(iEdge))
         if (iData <= 0) cycle

         if (iEdge <= nEdgesOwned) nMatchedOwned = nMatchedOwned + 1

         !*** A usable cell is a real cell of this block with at least one
         !*** active level. On a culled mesh the land side of a boundary
         !*** edge points at the nCells+1 dummy cell, whose maxLevelCell is
         !*** 0; a cell beyond the outermost halo is likewise not usable.
         !*** The prescribed flux is positive in the direction of positive
         !*** normalVelocity, i.e. from cellsOnEdge(1) toward cellsOnEdge(2),
         !*** so a positive flux is received by cellsOnEdge(2).

         cell2 = localOceanCell(cellsOnEdge(2, iEdge))
         cell1 = localOceanCell(cellsOnEdge(1, iEdge))

         if (cell1 == 0 .and. cell2 == 0) then
            !*** An owned river edge with ocean on neither side is a genuine
            !*** configuration error. The same situation on a halo edge just
            !*** means the receiving cell belongs to another block, so the
            !*** edge is skipped here and handled by its owner.
            if (iEdge <= nEdgesOwned) then
               call mpas_log_write('ocn_river_inflow_init: river inflow edge ' // &
                  'with global ID $i has no ocean cell on either side.', &
                  MPAS_LOG_CRIT, intArgs=(/indexToEdgeID(iEdge)/))
               err = 1
            end if
            cycle
         end if

         iLocal = iLocal + 1
         riverEdgeLocalIndex(iLocal) = iEdge
         riverEdgeDataIndex (iLocal) = iData
         riverEdgeCellPos   (iLocal) = cell2
         riverEdgeCellNeg   (iLocal) = cell1

      end do

      !*** Every listed edge must be owned by exactly one task. Summing the
      !*** owned matches across tasks catches both IDs that are absent from
      !*** the mesh and IDs listed more than once in the file.

      call mpas_dmpar_sum_int(domain % dminfo, nMatchedOwned, nMatchedGlobal)

      if (nMatchedGlobal /= nRiverInflowEdges) then
         call mpas_log_write('ocn_river_inflow_init: the forcing file lists ' // &
            '$i river inflow edges but $i were matched to owned mesh edges. ' // &
            'Check that riverInflowEdgeGlobalID holds indexToEdgeID values ' // &
            'for this mesh and contains no duplicates.', MPAS_LOG_CRIT, &
            intArgs=(/nRiverInflowEdges, nMatchedGlobal/))
         err = 1
      end if

      deallocate(sortedID)

      if (err == 0) then
         riverInflowOn   = .true.
         call mpas_log_write('ocn_river_inflow_init: $i river inflow edges ' // &
            'in the forcing file, $i present in this block ($i owned)', &
            intArgs=(/nRiverInflowEdges, nRiverEdgesLocal, nMatchedOwned/))
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_river_inflow_init!}}}

!***********************************************************************
!
!  routine ocn_river_inflow_build_array
!
!> \brief   Build the river inflow thickness tendency
!> \details
!>  Converts the prescribed, time-interpolated volume flux at each river
!>  inflow edge into a layer thickness tendency for the cell that
!>  receives it, and accumulates the per-cell total volume flux used by
!>  the barotropic subcycle of the split time integrators.
!>
!>  Called once per timestep, before the tendency routines.
!
!-----------------------------------------------------------------------

   subroutine ocn_river_inflow_build_array(domain, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain !< Input: domain

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), pointer :: &
         forcingPool, &
         statePool, &
         timeVaryingForcingPool

      real (kind=RKIND), dimension(:), pointer :: &
         riverInflowVolumeFlux, &
         riverInflowVelocity, &
         riverInflowVolumeFluxOnCell

      real (kind=RKIND), dimension(:,:), pointer :: &
         riverInflowLayerThicknessTendency, &
         riverInflowAccelZonal, &
         riverInflowAccelMeridional, &
         riverInflowMomentumRate, &
         layerThickness

      real (kind=RKIND) :: volumeFlux, absFlux, thicknessSum
      real (kind=RKIND) :: fluxLevel, hLevel, inflowSpeed, accel
      real (kind=RKIND) :: rate, capScale
      real (kind=RKIND) :: edgeSign, cosAngle, sinAngle
      real (kind=RKIND) :: ramp, t

      real (kind=RKIND), pointer :: forcingTimeIncrement

      integer :: iLocal, iCell, iEdge, k, kStart, kEnd, iList
      logical :: cellIsNew

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0

      if (.not. riverInflowOn) return

      call mpas_timer_start('river inflow build array')

      !*** The edge list built by ocn_river_inflow_init indexes the mesh
      !*** held by ocn_mesh, which is the first block, so this routine
      !*** works on that block only, as the other forcing modules do.

      call mpas_pool_get_subpool(domain % blocklist % structs, 'forcing', &
                                 forcingPool)
      call mpas_pool_get_subpool(domain % blocklist % structs, 'state', &
                                 statePool)
      call mpas_pool_get_subpool(domain % blocklist % structs, &
                                 'timeVaryingForcing', timeVaryingForcingPool)

      call mpas_pool_get_array(timeVaryingForcingPool, &
                               'riverInflowVolumeFlux', riverInflowVolumeFlux)
      call mpas_pool_get_array(forcingPool, &
                               'riverInflowLayerThicknessTendency', &
                               riverInflowLayerThicknessTendency)
      call mpas_pool_get_array(forcingPool, &
                               'riverInflowVolumeFluxOnCell', &
                               riverInflowVolumeFluxOnCell)
      call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)

      riverInflowLayerThicknessTendency(:,:) = 0.0_RKIND
      riverInflowVolumeFluxOnCell(:)         = 0.0_RKIND

      !*** Ramp the discharge up from zero, as the atmospheric forcing is
      !*** ramped, so the rivers do not switch on at full strength on a
      !*** resting ocean. forcingTimeIncrement places an RK4 substage within
      !*** the step. The momentum source is proportional to the discharge,
      !*** so it ramps with it. A ramp of zero disables the ramp.

      ramp = 1.0_RKIND
      if (config_river_inflow_ramp > 0.0_RKIND) then
         call mpas_pool_get_array(forcingPool, 'forcingTimeIncrement', &
                                  forcingTimeIncrement)
         t = daysSinceStartOfSim + forcingTimeIncrement/86400.0_RKIND
         if (t > config_river_inflow_ramp_delay) then
            ramp = tanh(2.0_RKIND*(t - config_river_inflow_ramp_delay) &
                        /config_river_inflow_ramp)
         else
            ramp = 0.0_RKIND
         end if
      end if

      if (momentumOn) then
         call mpas_pool_get_array(forcingPool, 'riverInflowAccelZonal', &
                                  riverInflowAccelZonal)
         call mpas_pool_get_array(forcingPool, 'riverInflowAccelMeridional', &
                                  riverInflowAccelMeridional)
         call mpas_pool_get_array(forcingPool, 'riverInflowMomentumRate', &
                                  riverInflowMomentumRate)
         riverInflowAccelZonal(:,:)      = 0.0_RKIND
         riverInflowAccelMeridional(:,:) = 0.0_RKIND
         riverInflowMomentumRate(:,:)    = 0.0_RKIND
         if (velocityFromFile) then
            call mpas_pool_get_array(timeVaryingForcingPool, &
                                     'riverInflowVelocity', riverInflowVelocity)
         end if
      end if

      nRiverCellsLocal = 0

      do iLocal = 1, nRiverEdgesLocal

         volumeFlux = ramp*riverInflowVolumeFlux(riverEdgeDataIndex(iLocal))
         iEdge      = riverEdgeLocalIndex(iLocal)

         !*** A zero flux has no direction, so it must not be tested against
         !*** the land side below.

         if (volumeFlux == 0.0_RKIND) cycle

         if (volumeFlux > 0.0_RKIND) then
            iCell = riverEdgeCellPos(iLocal)
         else
            iCell = riverEdgeCellNeg(iLocal)
         end if

         !*** On a boundary edge only one side is a valid ocean cell. A flux
         !*** directed at the land side has nowhere to go and means the
         !*** forcing file has the wrong sign, which would silently drop the
         !*** river, so it is fatal. Only the owning task reports it: only
         !*** the owner's log is guaranteed to be written. On a halo edge the
         !*** receiving cell may simply lie outside this block, and the owner
         !*** handles the edge.

         if (iCell == 0) then
            if (iEdge <= nEdgesOwned) then
               call mpas_log_write('ocn_river_inflow_build_array: flux at ' // &
                  'river inflow edge with global ID $i is directed at land. ' // &
                  'Check the sign convention of riverInflowVolumeFlux: it is ' // &
                  'positive from cellsOnEdge(1) toward cellsOnEdge(2).', &
                  MPAS_LOG_CRIT, intArgs=(/indexToEdgeID(iEdge)/))
               err = 1
            end if
            cycle
         end if

         !*** The inflow is an external source entering through the edge, so
         !*** the cell on the far side of an interior edge loses nothing.

         absFlux = abs(volumeFlux)

         riverInflowVolumeFluxOnCell(iCell) = &
            riverInflowVolumeFluxOnCell(iCell) + absFlux

         !*** Record the receiving cell once, for the momentum scatter.

         if (momentumOn) then
            cellIsNew = .true.
            do iList = 1, nRiverCellsLocal
               if (riverCellList(iList) == iCell) then
                  cellIsNew = .false.
                  exit
               end if
            end do
            if (cellIsNew) then
               nRiverCellsLocal = nRiverCellsLocal + 1
               riverCellList(nRiverCellsLocal) = iCell
            end if

            !*** The prescribed flux is positive from cellsOnEdge(1) toward
            !*** cellsOnEdge(2), which is the direction of the edge normal, so
            !*** the inward direction for the receiving cell is sign(flux).

            edgeSign = sign(1.0_RKIND, volumeFlux)
            cosAngle = cos(angleEdge(iEdge))
            sinAngle = sin(angleEdge(iEdge))
         end if

         !*** Vertical distribution. thicknessSum > 0 selects the 'uniform'
         !*** distribution; otherwise the whole flux goes into the topmost
         !*** active level, which is both the 'surface' option and the
         !*** fallback for a dry cell with no thickness to weight by, so that
         !*** it can rewet. The flux must land somewhere in either case: it
         !*** was already added to riverInflowVolumeFluxOnCell, which drives
         !*** the barotropic continuity equation, and the two must agree.

         thicknessSum = 0.0_RKIND
         if (distributeUniform) then
            do k = minLevelCell(iCell), maxLevelCell(iCell)
               thicknessSum = thicknessSum + layerThickness(k, iCell)
            end do
         end if

         kStart = minLevelCell(iCell)
         if (thicknessSum > 0.0_RKIND) then
            kEnd = maxLevelCell(iCell)
         else
            kEnd = kStart
         end if

         do k = kStart, kEnd

            if (thicknessSum > 0.0_RKIND) then
               fluxLevel = absFlux*layerThickness(k, iCell)/thicknessSum
            else
               fluxLevel = absFlux
            end if

            riverInflowLayerThicknessTendency(k, iCell) = &
               riverInflowLayerThicknessTendency(k, iCell) + &
               fluxLevel/areaCell(iCell)

            if (.not. momentumOn) cycle

            hLevel = layerThickness(k, iCell)
            if (hLevel <= 0.0_RKIND) cycle

            if (velocityFromFile) then
               inflowSpeed = abs(riverInflowVelocity(riverEdgeDataIndex(iLocal)))
            else
               !*** Consistent with the volume flux: the speed the water would
               !*** have crossing the full width of the mesh edge. On a coastal
               !*** mesh this underestimates the real discharge velocity, which
               !*** is why the speed can instead be read from the file.
               inflowSpeed = fluxLevel/(dvEdge(iEdge)*hLevel)
            end if

            !*** The derived speed grows as 1/hLevel as the cell dries.
            inflowSpeed = min(inflowSpeed, maxInflowSpeed)

            !*** Water entering at velocity v_in and mixing into the cell
            !*** changes the cell velocity u at rate*(v_in - u), where rate is
            !*** the inflow volume flux per unit cell volume at this level.
            !*** The rate*v_in part is accumulated here and the -rate*u part
            !*** is applied in ocn_river_inflow_vel_tend, where u is known.

            rate  = momentumScaling*fluxLevel/(areaCell(iCell)*hLevel)
            accel = rate*edgeSign*inflowSpeed

            riverInflowMomentumRate(k, iCell) = &
               riverInflowMomentumRate(k, iCell) + rate

            riverInflowAccelZonal(k, iCell) = &
               riverInflowAccelZonal(k, iCell) + accel*cosAngle
            riverInflowAccelMeridional(k, iCell) = &
               riverInflowAccelMeridional(k, iCell) + accel*sinAngle

         end do

      end do

      !*** Cap the relaxation rate at 1/dt. Beyond that the explicit source
      !*** would overshoot v_in within a step, which is what happens as
      !*** hLevel goes to zero in a drying cell. Both parts are scaled
      !*** together, so the velocity being relaxed toward is unchanged.

      if (momentumOn) then
         do iList = 1, nRiverCellsLocal
            iCell = riverCellList(iList)
            do k = minLevelCell(iCell), maxLevelCell(iCell)
               rate = riverInflowMomentumRate(k, iCell)
               if (rate*dtSeconds > 1.0_RKIND) then
                  capScale = 1.0_RKIND/(rate*dtSeconds)
                  riverInflowMomentumRate(k, iCell) = 1.0_RKIND/dtSeconds
                  riverInflowAccelZonal(k, iCell) = &
                     capScale*riverInflowAccelZonal(k, iCell)
                  riverInflowAccelMeridional(k, iCell) = &
                     capScale*riverInflowAccelMeridional(k, iCell)
               end if
            end do
         end do
      end if

      call mpas_timer_stop('river inflow build array')

   !--------------------------------------------------------------------

   end subroutine ocn_river_inflow_build_array!}}}

!***********************************************************************
!
!  routine ocn_river_inflow_layer_thickness
!
!> \brief   Add the river inflow tendency to the layer thickness tendency
!
!-----------------------------------------------------------------------

   subroutine ocn_river_inflow_layer_thickness(forcingPool, &
                                               layerThicknessTend, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: forcingPool !< Input: forcing information

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: layerThicknessTend

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: iCell, k, nCells

      real (kind=RKIND), dimension(:,:), pointer :: &
         riverInflowLayerThicknessTendency

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0

      if (.not. riverInflowOn) return

      call mpas_timer_start('river inflow thickness tendency')

      call mpas_pool_get_array(forcingPool, &
                               'riverInflowLayerThicknessTendency', &
                               riverInflowLayerThicknessTendency)

      ! as for the other forcing tendencies, only the 0 and 1 halos are needed
      nCells = nCellsHalo( 1 )

#ifdef MPAS_OPENACC
      !$acc enter data copyin(riverInflowLayerThicknessTendency)

      !$acc parallel loop present(layerThicknessTend, maxLevelCell, minLevelCell, &
      !$acc     riverInflowLayerThicknessTendency) &
      !$acc     private(k)
#else
      !$omp parallel
      !$omp do schedule(runtime) private(k)
#endif
      do iCell = 1, nCells
        do k = minLevelCell(iCell), maxLevelCell(iCell)
          layerThicknessTend(k,iCell) = layerThicknessTend(k,iCell) + &
                                        riverInflowLayerThicknessTendency(k,iCell)
        end do
      end do
#ifndef MPAS_OPENACC
      !$omp end do
      !$omp end parallel
#endif

#ifdef MPAS_OPENACC
      !$acc exit data delete(riverInflowLayerThicknessTendency)
#endif

      call mpas_timer_stop('river inflow thickness tendency')

   !--------------------------------------------------------------------

   end subroutine ocn_river_inflow_layer_thickness!}}}

!***********************************************************************
!
!  routine ocn_river_inflow_vel_tend
!
!> \brief   Add the river inflow momentum source to the velocity tendency
!> \details
!>  MPAS-Ocean solves the momentum equation in vector-invariant form at
!>  edges, so there is no flux-form momentum divergence with a boundary
!>  face to add the river momentum flux to. A river edge is also usually
!>  a culled land-boundary edge with no velocity degree of freedom at
!>  all. The momentum is therefore delivered as a body force on the
!>  receiving cell, projected onto that cell's surrounding active edges.
!>  This is a parameterized source rather than a true boundary
!>  condition: the momentum put into the cell is right, but the sub-cell
!>  structure of the jet is not resolved and the discrete operator is
!>  not energy-conserving.
!>
!>  Because the inflow adds mass as well as momentum, the source is
!>  rate*(v_in - u) rather than rate*v_in: the surrounding velocities
!>  relax toward the inflow velocity and cannot overshoot it. Each edge
!>  relaxes its own normal velocity toward the normal component of v_in.
!>
!>  The projection is the cell-to-edge average used for wind stress in
!>  ocn_surface_bulk_forcing_vel, namely the 0.5 weighting of
!>  mpas_vector_cell_to_edge_anisotropic followed by rotation onto the
!>  edge normal with angleEdge. It is applied here only to the handful
!>  of cells that receive inflow, rather than by calling the operator
!>  over every edge and level, because the source is nonzero in very few
!>  cells.
!
!-----------------------------------------------------------------------

   subroutine ocn_river_inflow_vel_tend(forcingPool, normalVelocity, &
                                        tendVel, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: forcingPool !< Input: forcing information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity !< Input: normal velocity at the current stage

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tendVel        !< Input/Output: accumulated velocity tendency

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), pointer :: &
         riverInflowAccelZonal, &
         riverInflowAccelMeridional, &
         riverInflowMomentumRate

      real (kind=RKIND) :: cosAngle, sinAngle

      integer :: iList, iCell, i, iEdge, k

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0

      if (.not. riverInflowOn) return
      if (.not. momentumOn)    return

      call mpas_timer_start('river inflow velocity tendency')

      call mpas_pool_get_array(forcingPool, 'riverInflowAccelZonal', &
                               riverInflowAccelZonal)
      call mpas_pool_get_array(forcingPool, 'riverInflowAccelMeridional', &
                               riverInflowAccelMeridional)
      call mpas_pool_get_array(forcingPool, 'riverInflowMomentumRate', &
                               riverInflowMomentumRate)

#ifdef MPAS_OPENACC
      !*** tendVel is resident on the device for the whole of ocn_tend_vel.
      !*** The source touches only the few edges around the cells that
      !*** receive inflow, so rather than launching a kernel over an
      !*** irregular scatter (where two adjacent receiving cells can write
      !*** the same edge), the accumulation is done on the host and the
      !*** array synchronised around it. This costs two transfers of
      !*** tendVel per call and is only paid when the momentum source is
      !*** switched on.
      !$acc update host(tendVel, normalVelocity)
#endif

      do iList = 1, nRiverCellsLocal

         iCell = riverCellList(iList)

         do i = 1, nEdgesOnCell(iCell)

            iEdge = edgesOnCell(i, iCell)

            !*** Only this task's own edges: the velocity tendency is
            !*** computed on owned edges and the velocity, not the tendency,
            !*** is halo exchanged. Every owned edge next to a receiving cell
            !*** is reached because that cell is in this block.

            if (iEdge > nEdgesOwned) cycle

            cosAngle = cos(angleEdge(iEdge))
            sinAngle = sin(angleEdge(iEdge))

            do k = minLevelEdgeBot(iEdge), maxLevelEdgeTop(iEdge)
               tendVel(k, iEdge) = tendVel(k, iEdge) + &
                  edgeMask(k, iEdge)*0.5_RKIND* &
                  (riverInflowAccelZonal     (k, iCell)*cosAngle + &
                   riverInflowAccelMeridional(k, iCell)*sinAngle - &
                   riverInflowMomentumRate   (k, iCell)*normalVelocity(k, iEdge))
            end do

         end do

      end do

#ifdef MPAS_OPENACC
      !$acc update device(tendVel)
#endif

      call mpas_timer_stop('river inflow velocity tendency')

   !--------------------------------------------------------------------

   end subroutine ocn_river_inflow_vel_tend!}}}

!***********************************************************************
!
!  function localOceanCell
!
!> \brief   Resolve a cellsOnEdge entry to a usable local ocean cell
!> \details
!>  Returns the cell index if it names a real cell of this block with at
!>  least one active level, and 0 otherwise. A culled mesh points the
!>  land side of a boundary edge at the nCells+1 dummy cell, and an edge
!>  in the outermost halo may reference a cell outside the block; both
!>  come back as 0.
!
!-----------------------------------------------------------------------

   integer function localOceanCell(iCell)!{{{

      integer, intent(in) :: iCell

      localOceanCell = 0

      if (iCell >= 1 .and. iCell <= nCellsAll) then
         if (maxLevelCell(iCell) > 0) localOceanCell = iCell
      end if

   end function localOceanCell!}}}

!***********************************************************************
!
!  function ocn_river_inflow_active
!
!> \brief   Report whether prescribed river inflow is in use
!> \details
!>  Lets the split time integrators skip the barotropic river inflow
!>  term entirely when the feature is off, so results are unchanged.
!
!-----------------------------------------------------------------------

   logical function ocn_river_inflow_active()!{{{

      ocn_river_inflow_active = riverInflowOn

   end function ocn_river_inflow_active!}}}

!***********************************************************************
!
!  function riverDataIndex
!
!> \brief   Look up a global edge ID in the sorted river edge list
!> \details
!>  Returns the index of the edge in the forcing array, or 0 if the given
!>  global edge ID is not a river inflow edge. sortedID is sorted on row
!>  1 (the global ID) and carries the forcing array position in row 2.
!
!-----------------------------------------------------------------------

   integer function riverDataIndex(sortedID, nID, globalID)!{{{

      integer, dimension(:,:), intent(in) :: sortedID
      integer,                 intent(in) :: nID
      integer,                 intent(in) :: globalID

      integer :: lo, hi, mid

      riverDataIndex = 0

      lo = 1
      hi = nID

      do while (lo <= hi)
         mid = (lo + hi)/2
         if (sortedID(1, mid) == globalID) then
            riverDataIndex = sortedID(2, mid)
            return
         else if (sortedID(1, mid) < globalID) then
            lo = mid + 1
         else
            hi = mid - 1
         end if
      end do

   end function riverDataIndex!}}}

!***********************************************************************

end module ocn_river_inflow

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
