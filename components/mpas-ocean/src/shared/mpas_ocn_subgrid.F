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


   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_subgrid_layer_thickness_lookup, &
             ocn_subgrid_ssh_lookup, &
             ocn_subgrid_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   !real(kind=RKIND), public, dimension(:,:), pointer :: &
   !   subgridWetVolumeCellTable, &
   !   subgridWetVolumeEdgeTable, &
   !   subgridWetVolumeVertexTable, &
   !   subgridSshCellTableRange, &
   !   subgridSshEdgeTableRange, &
   !   subgridSshVertexTableRange
   !real(kind=RKIND), public, dimension(:), pointer :: &
   !   subgridEdgeBathymetry, &
   !   subgridVertexBathymetry

   !integer, pointer :: nSubgridTableLevels
   real(kind=RKIND) :: eps = 1e-10


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

   subroutine ocn_subgrid_layer_thickness_lookup(zeta, &
                                                 subgridTable, &
                                                 subgridTableRange, &
                                                 bathymetry, &
                                                 layerThick)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: zeta
      real (kind=RKIND), dimension(:), intent(in) :: subgridTable
      real (kind=RKIND), dimension(:), intent(in) :: subgridTableRange
      real (kind=RKIND), intent(in) :: bathymetry

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(out) :: layerThick

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: lev
      integer :: nSubgridTableLevels

      real (kind=RKIND) :: tableMin
      real (kind=RKIND) :: tableMax
      real (kind=RKIND) :: deltaZ
      real (kind=RKIND) :: zeta0, zeta1

      tableMin = subgridTableRange(1)
      tableMax = subgridTableRange(2)
      nSubgridTableLevels = config_subgrid_table_levels_init
      deltaZ = (tableMax - tableMin)/(nSubgridTableLevels-1)

      if (zeta >= tableMax) then
        layerThick = zeta + bathymetry
        !call mpas_log_write('thickness lookup, above table: $r $r $r',realArgs=(/layerThick,zeta,bathymetry/))
      else if (zeta <= tableMin) then
        !layerThick = 0.0_RKIND
        layerThick = config_drying_min_cell_height + eps
        !call mpas_log_write('thickness lookup, below table: $r $r $r',realArgs=(/layerThick,zeta,bathymetry/))
      else 
        do lev = 1, nSubgridTableLevels-1
          zeta0 = (real(lev,RKIND)-1.0_RKIND)*deltaZ + tableMin
          zeta1 = zeta0 + deltaZ

          if ((zeta <= zeta1) .and. (zeta >= zeta0)) then
            layerThick = ((zeta-zeta0)*subgridTable(lev+1) - (zeta-zeta1)*subgridTable(lev))/deltaZ
            !call mpas_log_write('thickness lookup, in table: $r $r $r $r',realArgs=(/layerThick,zeta,bathymetry,zeta+bathymetry/))
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

   subroutine ocn_subgrid_ssh_lookup(layerThick, &
                                     subgridTable, &
                                     subgridTableRange, &
                                     bathymetryMean, &
                                     bathymetryMin, &
                                     zeta)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: layerThick
      real (kind=RKIND), dimension(:), intent(in) :: subgridTable
      real (kind=RKIND), dimension(:), intent(in) :: subgridTableRange
      real (kind=RKIND), intent(in) :: bathymetryMean
      real (kind=RKIND), intent(in) :: bathymetryMin

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(inout) :: zeta

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: lev
      integer :: nSubgridTableLevels

      real (kind=RKIND) :: deltaZ
      real (kind=RKIND) :: tableMin, tableMax
      real (kind=RKIND) :: layerThickMin, layerThickMax
      real (kind=RKIND) :: layerThick0, layerThick1
      real (kind=RKIND) :: zeta0, zeta1
      real (kind=RKIND) :: phi0, phi1

      tableMin = subgridTableRange(1)
      tableMax = subgridTableRange(2)
      nSubgridTableLevels = config_subgrid_table_levels_init
      deltaZ = (tableMax - tableMin)/(nSubgridTableLevels-1)

      layerThickMin = subgridTable(1)
      layerThickMax = subgridTable(nSubgridTableLevels)
      !call mpas_log_write('ssh in: $r',realArgs=(/zeta/))

      do lev = 1,nSubgridTableLevels
        zeta0 = (real(lev,RKIND)-1.0_RKIND)*deltaZ + tableMin
        layerThick0 = subgridTable(lev)
        !call mpas_log_write('ssh, layerThick: $r $r',realArgs=(/zeta0,layerThick0/))
      enddo

      if (layerThick >= layerThickMax) then
        zeta = layerThick - bathymetryMean
        !call mpas_log_write('ssh lookup, above table: $r $r $r',realArgs=(/layerThick,zeta,bathymetryMean/))

      else if (layerThick <= layerThickMin) then
        zeta = config_drying_min_cell_height + eps - bathymetryMin
        !call mpas_log_write('ssh lookup, below table: $r $r $r',realArgs=(/layerThick,zeta,bathymetryMin/))

      else        

        do lev = 1, nSubgridTableLevels-1
          zeta0 = (real(lev,RKIND)-1.0_RKIND)*deltaZ + tableMin
          zeta1 = zeta0 + deltaZ
  
          layerThick0 = subgridTable(lev)
          layerThick1 = subgridTable(lev+1)

          !call mpas_log_write('ssh lookup, in table level $i: $r $r $r $r',intArgs=(/lev/),realArgs=(/zeta0,zeta1,layerThick0,layerThick1/))
  
          if ((layerThick <= layerThick1) .and. (layerThick >= layerThick0)) then
            phi0 = (layerThick-layerThick1)/(layerThick0-layerThick1)
            phi1 = (layerThick-layerThick0)/(layerThick1-layerThick0)
            zeta = phi0*zeta0 + phi1*zeta1

            !call mpas_log_write('ssh lookup, in table: $r $r $r $r',realArgs=(/layerThick,zeta,bathymetryMean,zeta+bathymetryMean/))
            return
          end if
  
        end do
      end if

      !call mpas_log_write('$r',realArgs=(/zeta/))

   end subroutine ocn_subgrid_ssh_lookup!}}}

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
      type(mpas_pool_type), pointer :: meshPool

      err = 0
      if (.not. config_use_subgrid_wetting_drying) then
        return
      end if

      !if (nVertLevels .ne. 1) then
      !  call mpas_log_write('config_config_use_subgrid_wetting_drying = .true. requires single layer' , MPAS_LOG_CRIT)
      !end if 

      block => domain%blocklist
      call mpas_pool_get_subpool(block%structs, 'mesh', meshPool)

      !call mpas_pool_get_array(meshPool, 'subgridWetVolumeCellTable', &
      !                         subgridWetVolumeCellTable)
      !call mpas_pool_get_array(meshPool, 'subgridWetVolumeEdgeTable', &
      !                         subgridWetVolumeEdgeTable)
      !call mpas_pool_get_array(meshPool, 'subgridWetVolumeVertexTable', &
      !                         subgridWetVolumeVertexTable)
      !call mpas_pool_get_array(meshPool, 'subgridSshCellTableRange', &
      !                         subgridSshCellTableRange)
      !call mpas_pool_get_array(meshPool, 'subgridSshEdgeTableRange', &
      !                         subgridSshEdgeTableRange)
      !call mpas_pool_get_array(meshPool, 'subgridSshVertexTableRange', &
      !                         subgridSshVertexTableRange)
      !call mpas_pool_get_array(meshPool, 'subgridEdgeBathymetry', &
      !                         subgridEdgeBathymetry)
      !call mpas_pool_get_array(meshPool, 'subgridVertexBathymetry', &
      !                         subgridVertexBathymetry)

      !call mpas_pool_get_dimension(meshPool, 'nSubgridTableLevels', &
      !                             nSubgridTableLevels)


   end subroutine ocn_subgrid_init!}}}

end module ocn_subgrid

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
