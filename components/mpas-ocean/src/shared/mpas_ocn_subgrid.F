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
!> \authors Steven Brus, Damrongsak Wirasaet
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

   real(kind=RKIND), public, dimension(:,:), pointer :: &
      subgridWetVolumeCellTable, &
      subgridWetVolumeEdgeTable, &
      subgridWetVolumeVertexTable, &
      subgridSshCellTableRange, &
      subgridSshEdgeTableRange, &
      subgridSshVertexTableRange

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
!  routine ocn_subgrid_layer_thickness_lookup
!
!> \brief   
!> \author  Steven Brus, Damrongsak Wirasaet
!> \date    September 2022
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_subgrid_layer_thickness_lookup(ssh, &
                                                 subgridTable, &
                                                 subgridTableRange, &
                                                 bathymetry, &
                                                 layerThickness)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: ssh
      real (kind=RKIND), dimension(:), intent(in) :: subgridTable
      real (kind=RKIND), dimension(:), intent(in) :: subgridTableRange
      real (kind=RKIND), intent(in) :: bathymetry

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(out) :: layerThickness

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      real (kind=RKIND) :: tableMin
      real (kind=RKIND) :: tableMax
      real (kind=RKIND) :: deltaZ
      real (kind=RKIND) :: ssh0, ssh1

      tableMin = subgridTableRange(1)
      tableMax = subgridTableRange(2)
      deltaZ = (tableMax - tableMin)/(subgridTableLevels-1)

      if (ssh > tableMax) then
        layerThickness = ssh + bathymetry
        return 

      else if (ssh < tableMin) then
        layerThickness = 0.0_RKIND
        return

      else 
        do lev = 1, subgridTableLevels-1
          ssh0 = (real(lev,RKIND)-1.0_RKIND)*deltaZ + tableMin
          ssh1 = ssh0 + deltaZ

          if ((ssh <= ssh1) .and. (ssh >= ssh0)) then
            layerThickness = ((ssh-ssh0)*subgridTable(lev+1) - (ssh-ssh1)*subgridTable(lev))/deltaZ
            return
          end if

        end do
      end if      

   end subroutine ocn_subgrid_layer_thickness_lookup!}}}

!***********************************************************************
!
!  routine ocn_subgrid_ssh_lookup
!> \brief   
!> \author  Steven Brus, Damrongsak Wirasaet
!> \date    September 2022
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_subgrid_ssh_lookup(layerThickness, &
                                     subgridTable, &
                                     subgridTableRange, &
                                     bathymetry, &
                                     ssh)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: layerThickness
      real (kind=RKIND), dimension(:), intent(in) :: subgridTable
      real (kind=RKIND), dimension(:), intent(in) :: subgridTableRange
      real (kind=RKIND), intent(in) :: bathymetry

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(out) :: ssh

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      real (kind=RKIND) :: tableMin
      real (kind=RKIND) :: tableMax
      real (kind=RKIND) :: layerThicknessMax
      real (kind=RKIND) :: deltaZ
      real (kind=RKIND) :: ssh0, ssh1

      tableMin = subgridTableRange(1)
      tableMax = subgridTableRange(2)
      deltaZ = (tableMax - tableMin)/(subgridTableLevels-1)

      layerThicknessMax = subgridTable(subgridTableLevels)
      if (layerThickness > layerThicknessMax) then
        ssh = layerThckness - bathymetry
        return 

      else        

        do lev = 1, subgridTableLevels-1
          ssh0 = (real(lev,RKIND)-1.0_RKIND)*deltaZ + tableMin
          ssh1 = ssh0 + deltaZ
  
          layerThickness0 = subgridTable(lev)
          layerThickness1 = subgridTable(lev+1)
  
          if ((layerThickness <= layerThickness1) .and. (layerThickness >= layerThickness0)) then
            phi0 = (layerThickness-layerThickness1)/(layerThickness0-layerThickness1)
            phi1 = (layerThickness-layerThickness0)/(layerThickness1-layerThickness0)
            ssh = phi0*ssh0 + phi1*ssh1
            return
          end if
  
        end do
      end if

   end subroutine ocn_subgrid_layer_thickness_lookup!}}}

!***********************************************************************
!
!  routine ocn_subgrid_init
!
!> \brief   Initializes subgrid wetting and drying module.
!> \author  Steven Brus, Damrongsak Wirasaet
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

      integer, intent(out) :: err

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: statePool

      err = 0

      if (config_use_subgrid_wetting_drying .and. (nVertLevels .ne. 1)) then
        call mpas_log_write('config_config_use_subgrid_wetting_drying = .true. requires single layer' , MPAS_LOG_CRIT)
      end if 

      block => domain%blocklist
      call mpas_pool_get_subpool(block%structs, 'state', statePool)

      call mpas_pool_get_array(statePool, 'subgridWetVolumeCellTable', &
                               subgridWetVolumeCellTable)
      call mpas_pool_get_array(statePool, 'subgridWetVolumeEdgeTable', &
                               subgridWetVolumeEdgeTable)
      call mpas_pool_get_array(statePool, 'subgridWetVolumeVertexTable', &
                               subgridWetVolumeVertexTable)
      call mpas_pool_get_array(statePool, 'subgridSshCellTableRange', &
                               subgridSshCellTableRange)
      call mpas_pool_get_array(statePool, 'subgridSshEdgeTableRange', &
                               subgridSshEdgeTableRange)
      call mpas_pool_get_array(statePool, 'subgridSshVertexTableRange', &
                               subgridSshVertexTableRange)

   end subroutine ocn_subgrid_init!}}}

end module ocn_subgrid

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
