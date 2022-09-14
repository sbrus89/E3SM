! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_subgrid
!
!> \brief   MPAS ocean subgrid wetting and drying 
!> \authors Damrongsak Wirasaet, Steven Brus 
!> \date    September 2022 
!> \details
!>  This module contains routines for subgrid wetting and drying 
!>  Design document located in :
!>    MPAS-Model/docs/ocean/design_docs
!
!-----------------------------------------------------------------------

module ocn_subgrid

   use mpas_kind_types
   use mpas_constants
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timekeeping
   use mpas_timer
   use ocn_constants
   use ocn_config
   use ocn_mesh
   use ocn_diagnostics_variables

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

   public :: ocn_compute_subgrid_forcing, &
             ocn_subgrid_init, &

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_compute_subgrid_correction
!
!> \brief   
!> \author  Damrongsak Wirasaet, Steven Brus
!> \date    September 2022
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_compute_subgrid_correction(err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: Error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------


   end subroutine ocn_compute_subgrid_correction!}}}


!***********************************************************************
!
!  routine ocn_subgrid_init
!
!> \brief   Initializes subgrid wetting and drying module.
!> \author  Damrongsak Wirasaet, Steven Brus
!> \date    September 2022
!> \details
!>  This routine initializes the subgrid wetting and drying module
!
!-----------------------------------------------------------------------

   subroutine ocn_subgrid_init(domain,err)!{{{

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      err = 0

      !-----------------------------------------------------------------

   end subroutine ocn_subgrid_init!}}}

end module ocn_subgrid

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
