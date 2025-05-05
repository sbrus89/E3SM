! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_thick_hadv
!
!> \brief MPAS ocean horizontal advection for thickness
!> \author Doug Jacobsen
!> \date   16 September 2011
!> \details
!>  This module contains the routine for computing
!>  tendencies for thickness from horizontal advection
!
!-----------------------------------------------------------------------

module ocn_thick_hadv

   use mpas_timer
   use mpas_derived_types
   use mpas_pool_routines
   use ocn_constants
   use ocn_config
   use ocn_mesh

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

   public :: ocn_thick_hadv_tend, &
             ocn_thick_hadv_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: thickHadvOn

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_thick_hadv_tend
!
!> \brief   Computes tendency term from horizontal advection of thickness
!> \author  Doug Jacobsen
!> \date    15 September 2011
!> \details
!>  This routine computes the horizontal advection tendency for
!>  thicknes based on current state and user choices of forcings.
!
!-----------------------------------------------------------------------

   subroutine ocn_thick_hadv_tend(normalVelocity, layerThickEdgeFlux, tend, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity    !< Input: velocity

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickEdgeFlux     !< Input: flux-related thickness at edge

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tend          !< Input/Output: velocity tendency

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iEdge, k, i, iCell

      real (kind=RKIND) :: flux, invAreaCell

      real (kind=RKIND):: hCrit, hRampMin
      real (kind=RKIND), parameter:: hRampCoef = 4.0_RKIND 

      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing tendencies
      ! note that the user can choose multiple options and the
      !   tendencies will be added together
      !
      !-----------------------------------------------------------------

      err = 0

      if(.not.thickHadvOn) return

      call mpas_timer_start("thick hadv")

      !!
      hCrit = config_drying_min_cell_height + &
                              config_drying_safety_height

      hRampMin = config_zero_drying_velocity_ramp_hmin ;

#ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc     present(tend, normalVelocity, dvEdge, &
      !$acc             layerThickEdgeFlux, edgeSignOnCell, &
      !$acc             minLevelEdgeBot, maxLevelEdgeBot) &
      !$acc     private(i, k, invAreaCell, flux)
#else
      !$omp parallel
      !$omp do schedule(runtime) private(invAreaCell, i, iEdge, k, flux)
#endif
      do iCell = 1, nCellsOwned
        invAreaCell = 1.0_RKIND / areaCell(iCell)
        do i = 1, nEdgesOnCell(iCell)
          iEdge = edgesOnCell(i, iCell)
          do k = minLevelEdgeBot(iEdge), maxLevelEdgeTop(iEdge)
            flux = normalVelocity(k, iEdge) * dvEdge(iEdge) * layerThickEdgeFlux(k, iEdge)
            
            !!! DW: 
            if ( config_use_wetting_drying .AND. & 
                      config_use_subgrid_wetting_drying ) then 
                if ( layerThickEdgeFlux(k,iEdge) <= hCrit ) flux = 0.0_RKIND

              !     
              ! if ( layerThickEdgeFlux(k,iEdge) <= hCrit ) then
              !   flux = 0.0_RKIND ;
              ! elseif ( (layerThickEdgeFlux(k,iEdge) > hCrit) .and. &
              !   (layerThickEdgeFlux(k,iEdge) <= config_zero_drying_velocity_ramp_hmax) ) then
              !
              !   flux = flux* &
              !      tanh(hRampCoef*(layerThickEdgeFlux(k,iEdge) - hRampMin)/hRampMin)
              ! endif
              !        
            endif
            !!!!
 
            tend(k, iCell) = tend(k, iCell) + edgeSignOnCell(i, iCell) * flux * invAreaCell
          end do
        end do
      end do
#ifndef MPAS_OPENACC
      !$omp end do
      !$omp end parallel
#endif

      call mpas_timer_stop("thick hadv")

   !--------------------------------------------------------------------

   end subroutine ocn_thick_hadv_tend!}}}

!***********************************************************************
!
!  routine ocn_thick_hadv_init
!
!> \brief   Initializes ocean horizontal thickness advection
!> \author  Doug Jacobsen
!> \date    16 September 2011
!> \details
!>  This routine initializes quantities related to horizontal thickness
!>  advection in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_thick_hadv_init(err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! call individual init routines for each parameterization
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      thickHadvOn = .true.

      if(config_disable_thick_hadv) thickHadvOn = .false.

      err = 0

   !--------------------------------------------------------------------

   end subroutine ocn_thick_hadv_init!}}}

!***********************************************************************

end module ocn_thick_hadv

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

! vim: foldmethod=marker
