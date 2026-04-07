! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vel_forcing_radiation_stress
!
!> \brief MPAS ocean wave radiation stress forcing
!> \author Steven Brus
!> \date   April 2026
!> \details
!>  This module computes the momentum tendency from wave radiation
!>  stress gradients. The radiation stress tensor (Sxx, Syy, Sxy)
!>  is received from WW3 at cell centers in geographic coordinates.
!>  The tensor divergence is computed using the Green-Gauss method,
!>  then projected onto edge normals and added to the velocity
!>  tendency.
!
!-----------------------------------------------------------------------

module ocn_vel_forcing_radiation_stress

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer
   use mpas_log

   use ocn_constants
   use ocn_config
   use ocn_mesh

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_vel_forcing_radiation_stress_tend, &
             ocn_vel_forcing_radiation_stress_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: &
      radiationStressOff  !< on/off switch for radiation stress forcing

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vel_forcing_radiation_stress_tend
!
!> \brief   Computes tendency from wave radiation stress gradients
!> \author  Steven Brus
!> \date    April 2026
!> \details
!>  Computes the radiation stress gradient forcing for the momentum
!>  equation using the Green-Gauss method for tensor divergence on
!>  the MPAS unstructured mesh.
!>
!>  The force per unit mass is:
!>    F = -(1/rho_sw) * div(S) / h
!>  where S is the radiation stress tensor and h is the layer thickness.
!>
!>  The divergence is computed in two steps:
!>  1. Compute div(S) at cell centers using Gauss' theorem
!>  2. Interpolate to edges and project onto edge normals
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_forcing_radiation_stress_tend( &
                               forcingPool, layerThickEdgeMean, &
                               tend, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         forcingPool          !< [in] Forcing pool with radiation stress

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickEdgeMean   !< [in] mean thickness at edge

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tend                 !< [inout] accumulated velocity tendency

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iCell, iEdge, i, k, &
         cell1, cell2, &
         kMin, kMax

      real (kind=RKIND) :: &
         sxxEdge, syyEdge, sxyEdge, &
         fluxX, fluxY, &
         divSxEdge, divSyEdge, &
         forceNormal, &
         cosAngle, sinAngle, &
         invAreaCell1, &
         signEdge, dvEdgeLocal, &
         hEdge

      real (kind=RKIND), dimension(:), pointer :: &
         radStressSxx, radStressSyy, radStressSxy

      real (kind=RKIND), dimension(:), pointer :: &
         radStressDivX, radStressDivY

      real (kind=RKIND), dimension(:), allocatable :: &
         divSx, divSy

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0
      if (radiationStressOff) return
      call mpas_timer_start('vel radiation stress')

      ! Get radiation stress arrays from forcing pool
      call mpas_pool_get_array(forcingPool, 'radiationStressSxx', &
                               radStressSxx)
      call mpas_pool_get_array(forcingPool, 'radiationStressSyy', &
                               radStressSyy)
      call mpas_pool_get_array(forcingPool, 'radiationStressSxy', &
                               radStressSxy)
      call mpas_pool_get_array(forcingPool, 'radiationStressDivX', &
                               radStressDivX)
      call mpas_pool_get_array(forcingPool, 'radiationStressDivY', &
                               radStressDivY)

      ! If arrays are not allocated (package not active), return
      if (.not. associated(radStressSxx)) then
         call mpas_timer_stop('vel radiation stress')
         return
      end if

      ! Allocate work arrays for divergence at cells
      allocate(divSx(nCellsAll+1))
      allocate(divSy(nCellsAll+1))

      divSx(:) = 0.0_RKIND
      divSy(:) = 0.0_RKIND

      !-----------------------------------------------------------------
      ! Step 1: Compute tensor divergence at cell centers
      ! using the Green-Gauss (divergence theorem) approach
      !
      ! For each cell, sum over edges:
      !   divSx = (1/A) * sum_e [ sign_e * (Sxx_e*cos(a) + Sxy_e*sin(a)) * dv_e ]
      !   divSy = (1/A) * sum_e [ sign_e * (Sxy_e*cos(a) + Syy_e*sin(a)) * dv_e ]
      !
      ! where Sxx_e = average of the two cells adjacent to edge e
      !-----------------------------------------------------------------

      !$omp parallel
      !$omp do schedule(runtime) &
      !$omp    private(i, iEdge, cell1, cell2, &
      !$omp            cosAngle, sinAngle, signEdge, dvEdgeLocal, &
      !$omp            sxxEdge, syyEdge, sxyEdge, &
      !$omp            fluxX, fluxY, invAreaCell1)
      do iCell = 1, nCellsOwned
         invAreaCell1 = 1.0_RKIND / areaCell(iCell)
         do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            cell1 = cellsOnEdge(1, iEdge)
            cell2 = cellsOnEdge(2, iEdge)

            cosAngle = cos(angleEdge(iEdge))
            sinAngle = sin(angleEdge(iEdge))
            signEdge = real(edgeSignOnCell(i, iCell), RKIND)
            dvEdgeLocal = dvEdge(iEdge)

            ! Interpolate tensor components to edge (simple average)
            sxxEdge = 0.5_RKIND * (radStressSxx(cell1) &
                                 + radStressSxx(cell2))
            syyEdge = 0.5_RKIND * (radStressSyy(cell1) &
                                 + radStressSyy(cell2))
            sxyEdge = 0.5_RKIND * (radStressSxy(cell1) &
                                 + radStressSxy(cell2))

            ! Normal component of each "row" of the tensor
            fluxX = sxxEdge * cosAngle + sxyEdge * sinAngle
            fluxY = sxyEdge * cosAngle + syyEdge * sinAngle

            ! Accumulate divergence
            divSx(iCell) = divSx(iCell) &
                         + signEdge * fluxX * dvEdgeLocal * invAreaCell1
            divSy(iCell) = divSy(iCell) &
                         + signEdge * fluxY * dvEdgeLocal * invAreaCell1
         end do
      end do
      !$omp end do
      !$omp end parallel

      ! Store divergence for diagnostics
      if (associated(radStressDivX)) then
         do iCell = 1, nCellsOwned
            radStressDivX(iCell) = divSx(iCell)
            radStressDivY(iCell) = divSy(iCell)
         end do
      end if

      !-----------------------------------------------------------------
      ! Step 2: Compute edge-normal forcing from cell-centered divergence
      !
      ! For each edge, interpolate divergence from adjacent cells and
      ! project onto edge normal:
      !   F_n = -(divSx_e * cos(a) + divSy_e * sin(a)) / (rho_sw * h_e)
      !-----------------------------------------------------------------

      !$omp parallel
      !$omp do schedule(runtime) &
      !$omp    private(cell1, cell2, cosAngle, sinAngle, k, kMin, kMax, &
      !$omp            divSxEdge, divSyEdge, forceNormal, hEdge)
      do iEdge = 1, nEdgesOwned
         cell1 = cellsOnEdge(1, iEdge)
         cell2 = cellsOnEdge(2, iEdge)

         cosAngle = cos(angleEdge(iEdge))
         sinAngle = sin(angleEdge(iEdge))

         ! Interpolate divergence to edge (average of two cells)
         divSxEdge = 0.5_RKIND * (divSx(cell1) + divSx(cell2))
         divSyEdge = 0.5_RKIND * (divSy(cell1) + divSy(cell2))

         ! Project onto edge normal and negate (radiation stress
         ! gradient opposes wave propagation direction)
         kMin = minLevelEdgeBot(iEdge)
         kMax = maxLevelEdgeTop(iEdge)

         do k = kMin, kMax
            hEdge = layerThickEdgeMean(k, iEdge)
            if (hEdge > 0.0_RKIND) then
               forceNormal = -(divSxEdge * cosAngle &
                             + divSyEdge * sinAngle) &
                             / (rho_sw * hEdge)
               tend(k, iEdge) = tend(k, iEdge) &
                               + edgeMask(k, iEdge) * forceNormal
            end if
         end do
      end do
      !$omp end do
      !$omp end parallel

      deallocate(divSx)
      deallocate(divSy)

      call mpas_timer_stop('vel radiation stress')

   !--------------------------------------------------------------------

   end subroutine ocn_vel_forcing_radiation_stress_tend!}}}

!***********************************************************************
!
!  routine ocn_vel_forcing_radiation_stress_init
!
!> \brief   Initializes ocean radiation stress forcing
!> \author  Steven Brus
!> \date    April 2026
!> \details
!>  This routine initializes quantities related to radiation stress
!>  forcing in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_forcing_radiation_stress_init(err)!{{{

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0
      radiationStressOff = .true.

      if (config_use_radiation_stress_forcing) then
         radiationStressOff = .false.
         call mpas_log_write( &
            'Radiation stress forcing from waves is enabled')
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_vel_forcing_radiation_stress_init!}}}

!***********************************************************************

end module ocn_vel_forcing_radiation_stress

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
