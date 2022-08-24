! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the
! LICENSE file
! distributed with this code, or at
! http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_subgrid
!
!> \brief MPAS ocean initialize case -- Subgrid bathymetry look up tables 
!> \author Steven Brus
!> \date   06/29/2022
!> \details
!>  This module contains the routines for initializing the
!>  lookup tables for the subgrid wetting and drying implementation
!
!-----------------------------------------------------------------------

module ocn_init_subgrid

   use mpas_kind_types
   use mpas_io_units
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_stream_manager
   use mpas_io_streams
   use mpas_io
   use mpas_dmpar

   use ocn_constants
   use ocn_config
   use ocn_init_interpolation

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

   public :: ocn_init_setup_subgrid, &
             ocn_init_validate_subgrid

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   integer :: nLatTopo, nLonTopo
   type (field1DReal) :: topoLat, topoLon
   type (field2DReal) :: topoIC

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_subgrid
!
!> \brief   Set-up for test case
!> \author  Steven Brus
!> \date    08/19/2021
!> \details
!
!-----------------------------------------------------------------------

   subroutine ocn_init_setup_subgrid(domain, iErr)!{{{

   !--------------------------------------------------------------------

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: iErr

      type (block_type), pointer :: block_ptr

      type (mpas_pool_type), pointer :: meshPool, statePool
      type (mpas_pool_type), pointer :: forcingPool, diagnosticsPool

      integer, pointer :: nCellsSolve
      real (kind=RKIND), dimension(:), pointer :: lonCell, latCell
      real (kind=RKIND), dimension(:), pointer :: xCell, yCell, zCell
      real (kind=RKIND), dimension(:), pointer :: ssh, ssh_sal, ssh_sal_grad
      real (kind=RKIND), dimension(:), pointer :: areaCell, dcEdge

      character (len=StrKIND), pointer :: config_init_configuration


      ! assume no error
      iErr = 0

      ! get and test if this is the configuration specified
      call mpas_pool_get_config(domain % configs,'config_init_configuration', config_init_configuration)
      if(config_init_configuration .ne. trim('subgrid')) return

      !load data that required to initialize the ocean simulation
      block_ptr => domain % blocklist
      do while(associated(block_ptr))
        call mpas_pool_get_subpool(block_ptr % structs, 'mesh',meshPool)
        call mpas_pool_get_subpool(block_ptr % structs, 'state',statePool)
        call mpas_pool_get_subpool(block_ptr % structs, 'forcing',forcingPool)
        call mpas_pool_get_subpool(block_ptr % structs, 'diagnostics',diagnosticsPool)

        call mpas_pool_get_dimension(meshPool, 'nCellsSolve',nCellsSolve)
        call mpas_pool_get_dimension(meshPool, 'nEdgesArray',nEdgesArray)
        call mpas_pool_get_config(meshPool, 'sphere_radius', sphere_radius)

        call mpas_pool_get_array(meshPool, 'latCell', latCell)
        call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
        call mpas_pool_get_array(meshPool, 'xCell', xCell)
        call mpas_pool_get_array(meshPool, 'yCell', yCell)
        call mpas_pool_get_array(meshPool, 'zCell', zCell)
        call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
        call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
        call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)


        block_ptr => block_ptr % next
      end do

  !--------------------------------------------------------------------

   end subroutine ocn_init_setup_subgrid!}}}

!***********************************************************************
!
!  routine ocn_init_define_subgrid_resolution
!
!> \brief   
!> \author  Steven Brus 
!> \date    06/29/2022
!> \details
!
!-----------------------------------------------------------------------

    subroutine ocn_init_define_subgrid_resolution(domain, iErr)!{{{



    end subroutine ocn_init_define_subgrid_resolution!}}}

!***********************************************************************
!
!  routine ocn_init_setup_subgrid_read_topo
!
!> \brief   Read the topography IC file
!> \author  Doug Jacobsen
!> \date    03/04/2014
!> \details
!>  This routine reads the topography IC file, including latitude and longitude
!>   information for topography data.
!
!-----------------------------------------------------------------------

    subroutine ocn_init_setup_subgrid_read_topo(domain, iErr)!{{{
       type (domain_type), intent(inout) :: domain
       integer, intent(out) :: iErr

       type (MPAS_Stream_type) :: topographyStream

       iErr = 0

       ! Define stream for depth levels
       call MPAS_createStream(topographyStream, domain % iocontext, config_subgrid_topography_file, MPAS_IO_NETCDF, &
                              MPAS_IO_READ, ierr=iErr)

       ! Setup topoLat, topoLon, and topoIC fields for stream to be read in
       topoLat % fieldName = trim(config_subgrid_topography_lat_varname)
       topoLat % dimSizes(1) = nLatTopo
       topoLat % dimNames(1) = trim(config_subgrid_topography_nlat_dimname)
       topoLat % isVarArray = .false.
       topoLat % isPersistent = .true.
       topoLat % isActive = .true.
       topoLat % hasTimeDimension = .false.
       topoLat % block => domain % blocklist
       allocate(topoLat % attLists(1))
       allocate(topoLat % array(nLatTopo))

       topoLon % fieldName = trim(config_subgrid_topography_lon_varname)
       topoLon % dimSizes(1) = nLonTopo
       topoLon % dimNames(1) = trim(config_subgrid_topography_nlon_dimname)
       topoLon % isVarArray = .false.
       topoLon % isPersistent = .true.
       topoLon % isActive = .true.
       topoLon % hasTimeDimension = .false.
       topoLon % block => domain % blocklist
       allocate(topoLon % attLists(1))
       allocate(topoLon % array(nLonTopo))

       topoIC % fieldName = trim(config_subgrid_topography_varname)
       topoIC % dimSizes(1) = nLonTopo
       topoIC % dimSizes(2) = nLatTopo
       topoIC % dimNames(1) = trim(config_subgrid_topography_nlon_dimname)
       topoIC % dimNames(2) = trim(config_subgrid_topography_nlat_dimname)
       topoIC % isVarArray = .false.
       topoIC % isPersistent = .true.
       topoIC % isActive = .true.
       topoIC % hasTimeDimension = .false.
       topoIC % block => domain % blocklist
       allocate(topoIC % attLists(1))
       allocate(topoIC % array(nLonTopo, nLatTopo))

       ! Add topoLat, topoLon, and topoIC fields to stream
       call MPAS_streamAddField(topographyStream, topoLat, iErr)
       call MPAS_streamAddField(topographyStream, topoLon, iErr)
       call MPAS_streamAddField(topographyStream, topoIC, iErr)

       ! Read stream
       call MPAS_readStream(topographyStream, 1, iErr)

       ! Close stream
       call MPAS_closeStream(topographyStream)

       if (config_subgrid_topography_latlon_degrees) then
          topoLat % array(:) = topoLat % array(:) * pii / 180.0_RKIND
          topoLon % array(:) = topoLon % array(:) * pii / 180.0_RKIND
       end if

    end subroutine ocn_init_setup_subgrid_read_topo!}}}

!***********************************************************************
!
!  routine ocn_init_validate_subgrid
!
!> \brief   Validation for test case
!> \author  Steven Brus
!> \date    08/19/2021
!> \details
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_subgrid(configPool, packagePool,iocontext, iErr)!{{{

   !--------------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: configPool
      type (mpas_pool_type), intent(inout) :: packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration',config_init_configuration)

      if(config_init_configuration .ne. trim('subgrid')) return

      inputFile = MPAS_io_open(config_subgrid_topography_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
      if (iErr .ne. 0) then
         call mpas_log_write( 'could not open file '// trim(config_subgrid_topography_file), MPAS_LOG_CRIT)
         return
      end if

      call MPAS_io_inq_dim(inputFile, config_subgrid_topography_nlat_dimname, nLatTopo, iErr)
      call MPAS_io_inq_dim(inputFile, config_subgrid_topography_nlon_dimname, nLonTopo, iErr)

      call MPAS_io_close(inputFile, iErr)

   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_subgrid!}}}

!***********************************************************************

end module ocn_init_subgrid

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
