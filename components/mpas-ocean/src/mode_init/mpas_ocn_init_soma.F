! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_soma
!
!> \brief MPAS ocean initialize case -- Simulating Ocean Mesoscale Activity (SOMA)
!> \author Todd Ringler
!> \date   10/08/2013
!> \details
!>  This module contains the routines for initializing the
!>  the idealized SOMA test case
!
!-----------------------------------------------------------------------

module ocn_init_soma

   use mpas_kind_types
   use mpas_io_units
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants

   use ocn_constants
   use ocn_config
   use ocn_init_vertical_grids
   use ocn_init_cell_markers

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public parameters
   !
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_init_setup_soma, &
             ocn_init_validate_soma

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_soma
!
!> \brief   Setup for soma test case
!> \author  Todd Ringler
!> \date    02/26/2014
!> \details
!>  This routine sets up the initial conditions for the
!>  SOMA configuration.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_setup_soma(domain, iErr)!{{{

   !--------------------------------------------------------------------

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: iErr

      ! local work variables
      type (block_type), pointer :: block_ptr

      type (mpas_pool_type), pointer :: meshPool, verticalMeshPool, statePool, forcingPool, tracersPool
      type (mpas_pool_type), pointer :: tracersSurfaceRestoringFieldsPool, tracersInteriorRestoringFieldsPool

      integer :: iCell, iEdge, iVertex, k, idx
      real (kind=RKIND) :: distance, deltaLon, deltaLat, xDistance, yDistance, zMid, sphereRadius
      real (kind=RKIND) :: lonCurrent, latCurrent
      real (kind=RKIND) :: deltay, depth, factor, latCenter, lonCenter, windStress
      real (kind=RKIND) :: temperature, salinity
      real (kind=RKIND), dimension(:), pointer :: interfaceLocations

      ! Define dimension pointers
      integer, pointer :: nVertLevels, nCells, nVertLevelsP1, nCellsSolve, nEdgesSolve, nVerticesSolve
      integer, pointer :: index_temperature, index_salinity, index_tracer1

      ! Define variable pointers
      logical, pointer :: on_a_sphere
      integer, dimension(:), pointer :: maxLevelCell
      real (kind=RKIND), dimension(:), pointer :: refBottomDepth, bottomCell, refZMid, fCell, fEdge, fVertex
      real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness
      real (kind=RKIND), pointer :: sphere_radius
      real (kind=RKIND), dimension(:), pointer :: lonCell, latCell, latEdge, latVertex, bottomDepth
      real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers, debugTracers
      real (kind=RKIND), dimension(:, :), pointer ::    activeTracersPistonVelocity, activeTracersSurfaceRestoringValue
      real (kind=RKIND), dimension(:, :, :), pointer :: activeTracersInteriorRestoringValue, activeTracersInteriorRestoringRate
      real (kind=RKIND), dimension(:), pointer :: windStressZonal, windStressMeridional

      ! assume no error
      iErr = 0

      ! test if SOMA is the desired configuration
      if(config_init_configuration .ne. trim('soma')) return

      call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
      call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)
      call mpas_pool_get_config(meshPool, 'sphere_radius', sphere_radius)
      sphereRadius = sphere_radius

      ! error checking
      if(.not. on_a_sphere) then
        call mpas_log_write( 'SOMA test case can only be defined on a spherical mesh.', MPAS_LOG_CRIT)
        iErr = 1
        return
      else
        call mpas_log_write( 'SOMA test case using spherical radius of size: $f ', realArgs=(/ sphereRadius /) )
      end if

      ! assign config variables
      nVertLevels  = config_soma_vert_levels
      print*, nVertLevels
      nVertLevelsP1 = nVertLevels + 1

      ! Define interface locations
      allocate( interfaceLocations( nVertLevelsP1 ) )
      !call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations )

      ! set center of SOMA domain
      ! Convert center locations to radians from degrees
      !latCenter = config_soma_center_latitude * pii / 180.0_RKIND
      !lonCenter = config_soma_center_longitude * pii / 180.0_RKIND
      latCenter = 45.0
      lonCenter = 0.0 

      ! Setup the vertical grid and layerThickness initial condition
      call mpas_log_write( 'setting up vertical grid and layer thickness')
      block_ptr => domain % blocklist
      do while(associated(block_ptr))
        call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
        call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
        call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
        call mpas_pool_get_dimension(meshPool, 'nEdgesSolve', nEdgesSolve)
        call mpas_pool_get_dimension(meshPool, 'nVerticesSolve', nVerticesSolve)
        call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
        call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
        call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
        call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
        call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)
        call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)

        ! Set layerThickness and restingThickness
        ! Uniform layer thickness across lat/lon
        layerThickness(1, :) = 10.0
        layerThickness(2, :) = 11.33 
        layerThickness(3, :) = 12.83
        layerThickness(4, :) = 14.53
        layerThickness(5, :) = 16.45
        layerThickness(6, :) = 18.63
        layerThickness(7, :) = 21.1
        layerThickness(8, :) = 23.9
        layerThickness(9, :) = 27.07
        layerThickness(10, :) = 30.65
        layerThickness(11, :) = 34.72
        layerThickness(12, :) = 39.32
        layerThickness(13, :) = 44.53
        layerThickness(14, :) = 50.43
        layerThickness(15, :) = 57.12
        layerThickness(16, :) = 64.69
        layerThickness(17, :) = 73.26
        layerThickness(18, :) = 82.97
        layerThickness(19, :) = 93.97
        layerThickness(20, :) = 106.42
        layerThickness(21, :) = 120.53
        layerThickness(22, :) = 136.5
        layerThickness(23, :) = 154.59
        layerThickness(24, :) = 175.08
        layerThickness(25, :) = 198.29
        layerThickness(26, :) = 224.57
        layerThickness(27, :) = 254.34
        layerThickness(28, :) = 288.05
        layerThickness(29, :) = 326.23
        layerThickness(30, :) = 369.47
        layerThickness(31, :) = 418.43
        do k = 1, nVertLevels
           restingThickness(k, :) = layerThickness(k, :)
        end do


         ! Set refBottomDepth
         refBottomDepth(1) = layerThickness(1,1)
         do k = 2, nVertLevels
            refBottomDepth(k) = refBottomDepth(k-1) + layerThickness(k,1) 
            print*, k,refBottomDepth(k)
         end do
         do k = 1, nVertLevels
            refZMid(k) = -(refBottomDepth(k) + refBottomDepth(k+1)) / 2.0
         end do

        block_ptr => block_ptr % next

      end do

      ! Set bathymetry
      call mpas_log_write( 'setting up bathymetry')
      block_ptr => domain % blocklist
      do while(associated(block_ptr))
        call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
        call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
        call mpas_pool_get_array(meshPool, 'latCell', latCell)
        call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
        call mpas_pool_get_array(meshPool, 'fCell', fCell)
        call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
        call mpas_pool_get_array(meshPool, 'fVertex', fVertex)
        call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
        call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)

        ! set bottomDepth
        bottomDepth(:) = 0.0_RKIND
        do iCell = 1, nCells
           lonCurrent = lonCell(iCell) * 180.0/pii
           latCurrent = latCell(iCell) * 180.0/pii

           deltaLon = abs(lonCurrent - lonCenter)
           if (deltaLon .gt. 180.0) deltaLon = deltaLon - 360.0
           deltaLat = latCurrent - latCenter
           distance = sqrt( deltaLon**2 + deltaLat**2 )
           factor = 1.0 - (distance / 28.0_RKIND)**2

           if(factor > -0.14) then
              bottomDepth(iCell) = 100.0 + 3400.0/2.0_RKIND &
                                 * (1.0 + tanh(factor/0.02))
              !bottomDepth(iCell) = 3400.0 + 100.0/2.0_RKIND &
              !                   * (1.0 + tanh(factor/0.02))
           else
              bottomDepth(iCell) = -1.0_RKIND
           endif

           ! Set maxLevelCell to -1 for cells to be culled
           if (bottomDepth(iCell) > 0.0_RKIND) then
             maxLevelCell(iCell) = 1
           else
             maxLevelCell(iCell) = -1
           endif

           ! Determine maxLevelCell based on bottomDepth and refBottomDepth
           ! Also set botomDepth based on refBottomDepth, since
           ! above bottomDepth was set with continuous analytical functions,
           ! and needs to be discrete
           if (maxLevelCell(iCell) > 0) then
             maxLevelCell(iCell) = nVertLevels
             if (nVertLevels .gt. 1) then
                do k = 1, nVertLevels
                  if (bottomDepth(iCell) < refBottomDepth(k) ) then
                      maxLevelCell(iCell) = k-1
                      bottomDepth(iCell) = refBottomDepth(k-1)
                      exit
                  end if
                end do
             end if
           end if

        enddo ! Looping through with iCell

        block_ptr => block_ptr % next

      enddo !  done setting bathymetry

      ! mark cells for culling
       block_ptr => domain % blocklist
       do while (associated(block_ptr))
          call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
          call ocn_mark_maxlevelcell(meshPool, iErr)
          block_ptr => block_ptr % next
       end do

      ! Set forcing boundary conditions and initial conditions for temperature and salinity
      call mpas_log_write( 'setting up forcing and initial T/S conditions')
      block_ptr => domain % blocklist
      do while(associated(block_ptr))
        call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
        call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
        call mpas_pool_get_array(meshPool, 'latCell', latCell)
        call mpas_pool_get_array(meshPool, 'latEdge', latEdge)
        call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
        call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
        call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
        call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
        call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
        call mpas_pool_get_dimension(tracersPool, 'index_tracer1', index_tracer1)
        call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
        call mpas_pool_get_array(tracersPool, 'debugTracers', debugTracers, 1)
        call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
        call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
        call mpas_pool_get_subpool(block_ptr % structs, 'forcing', forcingPool)
        call mpas_pool_get_array(forcingPool, 'windStressZonal', windStressZonal)
        call mpas_pool_get_array(forcingPool, 'windStressMeridional', windStressMeridional, 1)

        call mpas_pool_get_subpool(forcingPool, 'tracersSurfaceRestoringFields', tracersSurfaceRestoringFieldsPool)
        call mpas_pool_get_array(tracersSurfaceRestoringFieldsPool, 'activeTracersPistonVelocity', activeTracersPistonVelocity, 1)
        call mpas_pool_get_array(tracersSurfaceRestoringFieldsPool, &
          'activeTracersSurfaceRestoringValue', activeTracersSurfaceRestoringValue, 1)

        do iCell = 1, nCells
           lonCurrent = lonCell(iCell) * 180.0/pii
           latCurrent = latCell(iCell) * 180.0/pii

           ! Set initial temperature and salinity
           do k = 1, nVertLevels
              if ( associated(activeTracers) ) then
                 activeTracers(index_salinity, k, iCell) = 31.0
              end if
           enddo
           if ( associated(activeTracers) ) then
             activeTracers(index_temperature, 1, iCell) = 23.0
             activeTracers(index_temperature, 2, iCell) = 23.0
             activeTracers(index_temperature, 3, iCell) = 23.0
             activeTracers(index_temperature, 4, iCell) = 22.0
             activeTracers(index_temperature, 5, iCell) = 21.0
             activeTracers(index_temperature, 6, iCell) = 20.0
             activeTracers(index_temperature, 7, iCell) = 19.0
             activeTracers(index_temperature, 8, iCell) = 17.0
             activeTracers(index_temperature, 9, iCell) = 16.0
             activeTracers(index_temperature, 10, iCell) = 15.0
             activeTracers(index_temperature, 11, iCell) = 14.0
             activeTracers(index_temperature, 12, iCell) = 12.0
             activeTracers(index_temperature, 13, iCell) = 11.0
             activeTracers(index_temperature, 14, iCell) = 10.0
             activeTracers(index_temperature, 15, iCell) =  9.0
             activeTracers(index_temperature, 16, iCell) =  8.0
             activeTracers(index_temperature, 17, iCell) =  7.0
             activeTracers(index_temperature, 18, iCell) =  6.0
             activeTracers(index_temperature, 19, iCell) =  6.0
             activeTracers(index_temperature, 20, iCell) =  5.0
             activeTracers(index_temperature, 21, iCell) =  5.0
             activeTracers(index_temperature, 22, iCell) =  4.0
             activeTracers(index_temperature, 23, iCell) =  4.0
             activeTracers(index_temperature, 24, iCell) =  3.0
             activeTracers(index_temperature, 25, iCell) =  3.0
             activeTracers(index_temperature, 26, iCell) =  2.0
             activeTracers(index_temperature, 27, iCell) =  2.0
             activeTracers(index_temperature, 28, iCell) =  1.0
             activeTracers(index_temperature, 29, iCell) =  1.0
             activeTracers(index_temperature, 30, iCell) =  1.0
             activeTracers(index_temperature, 31, iCell) =  1.0
          end if

           if (config_soma_use_surface_temp_restoring) then
             ! surface restoring
             idx = index_temperature
             activeTracersSurfaceRestoringValue(idx,iCell) = (30.0 - 0.0)/60 * (75.0-latCurrent) + 0.0
             activeTracersPistonVelocity(idx,iCell) = config_soma_restoring_temp_piston_vel

             idx = index_salinity
             activeTracersSurfaceRestoringValue(idx,iCell) = 34.0_RKIND
             activeTracersPistonVelocity(idx,iCell) = 0.0_RKIND
           end if


           ! Set up debugging tracers
           if ( associated(debugTracers) ) then
              debugTracers(index_tracer1, :, iCell) = 1.0_RKIND
           end if

        end do ! iCell = 1, nCells

        ! Set wind stress
        do iCell = 1, nCells
           lonCurrent = lonCell(iCell) * 180.0/pii
           latCurrent = latCell(iCell) * 180.0/pii

           deltay =   (latCurrent - 47.5)/35.0
           factor = 1.0_RKIND -  deltay 
           windstress = factor * 0.08_RKIND * exp( -deltay**2 )  &
                 * cos(pii * deltay )

           windStressZonal(iCell) = windStress
           windStressMeridional(iCell) = 0.0_RKIND

        end do

       ! Set Coriolis parameters
       do iCell = 1, nCellsSolve
          fCell(iCell) = 2.0_RKIND * omega * sin(latCell(iCell))
       end do
       do iEdge = 1, nEdgesSolve
          fEdge(iEdge) = 2.0_RKIND * omega * sin(latEdge(iEdge))
       end do
       do iVertex = 1, nVerticesSolve
          fVertex(iVertex) = 2.0_RKIND * omega * sin(latVertex(iVertex))
       end do

        block_ptr => block_ptr % next
      end do

      call mpas_log_write( 'exiting ocn_init_setup_soma')

   !--------------------------------------------------------------------

   end subroutine ocn_init_setup_soma!}}}

!***********************************************************************
!
!  routine ocn_init_validate_soma
!
!> \brief   Validation for SOMA test case
!> \author  Todd Ringler
!> \date    02/26/2014
!> \details
!>  This routine validates the configuration options for the SOMA test case.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_soma(configPool, packagePool, iocontext, iErr)!{{{

   !--------------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext
      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_soma_vert_levels

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)
      if(config_init_configuration .ne. trim('soma')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool, 'config_soma_vert_levels', config_soma_vert_levels)

      if(config_vert_levels <= 0 .and. config_soma_vert_levels > 0) then
         config_vert_levels = config_soma_vert_levels
      else if (config_vert_levels <= 0) then
         call mpas_log_write( 'Validation failed for SOMA. Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
         iErr = 1
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_soma!}}}

end module ocn_init_soma

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
