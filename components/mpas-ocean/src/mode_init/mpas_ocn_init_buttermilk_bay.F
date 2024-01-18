! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_parabolic_bowl
!
!> \brief MPAS ocean initialize case -- TEMPLATE
!> \author D. Wirasaet, S. Brus
!> \date   May-June 2022
!> \details
!>  This module contains the routines for initializing the
!>  parabolic_bowl initial condition (Thacker's problem)
!>
!>
!.......................................................................
!-----------------------------------------------------------------------
!
!>  In order to add a new initial condition, do the following:
!>  1. In src/core_ocean/mode_init, copy these to your new initial condition name:
!>     cp mpas_ocn_init_TEMPLATE.F mpas_ocn_init_your_new_name.F
!>     cp Registry_TEMPLATE.xml Registry_ocn_your_new_name.xml
!>
!>  2. In those two new files, replace the following text:
!>     TEMPLATE, FILL_IN_AUTHOR, FILL_IN_DATE
!>     TEMPLATE uses underscores (subroutine names), like your_new_name.
!>
!>  3. Add a #include line for your registry to
!>     src/core_ocean/mode_init/Registry.xml
!>
!>  4. Copy and change TEMPLATE lines in src/core_ocean/mode_init/mpas_ocn_init_mode.F
!>
!>  5. Add these dependency lines by following TEMPLATE examples in:
!>     in src/core_ocean/mode_init/Makefile
!
!-----------------------------------------------------------------------

module ocn_init_Buttermilk_bay

   use mpas_kind_types
   use mpas_io_units
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar

   use ocn_constants
   use ocn_config
   use ocn_init_vertical_grids
   use ocn_init_cell_markers
   use ocn_subgrid

   use mpas_constants
   use mpas_io
   use mpas_io_streams
   use mpas_stream_manager


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

   public :: ocn_init_setup_Buttermilk_bay, &
             ocn_init_validate_Buttermilk_bay

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!   real (kind=RKIND):: LL, CC
!   real (kind=RKIND):: oneMC2, sqrtOneMC2, oneMC
!   real (kind=RKIND),  parameter :: eps=1.0e-10
   real(kind=RKIND), dimension(:,:), pointer :: &
      subgridWetVolumeCellTable, &
      subgridWetVolumeEdgeTable, &
      subgridWetVolumeVertexTable, &
      subgridSshCellTableRange, &
      subgridSshEdgeTableRange, &
      subgridSshVertexTableRange, &
      subgridWetFractionCellTable, &
      subgridWetFractionEdgeTable, &
      subgridWetFractionVertexTable
   real(kind=RKIND), dimension(:), pointer :: &
      subgridEdgeBathymetryMean, &
      subgridVertexBathymetryMean, &
      subgridCellBathymetryMin, &
      subgridEdgeBathymetryMin, &
      subgridVertexBathymetryMin, &
      subgridLayerThicknessDebug
   integer, pointer :: nSubgridTableLevels

   ! For netcdf topobathy input variables
   integer :: nLatTopo, nLonTopo

   type (field1DReal) :: topoLat, topoLon
   type (field2DReal) :: topoIC

   real(kind=RKIND), parameter:: eps = 1.0e-10_RKIND ;   
!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_parabolic_bowl
!
!> \brief   Setup for this initial condition
!> \author  D. Wirasaet and S. Brus
!> \date    May-June 2022
!> \details
!>  This routine sets up the initial conditions for this case.
!>   To be run in sigma vertical coordinates and single-layer
!
!-----------------------------------------------------------------------

  subroutine ocn_init_setup_Buttermilk_bay(domain, iErr)!{{{
    use mpas_vector_operations  ! To calcutate edgeNormalVector
    
    implicit none
    !--------------------------------------------------------------------

    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: iErr

    type (block_type), pointer :: block_ptr
    type (mpas_pool_type), pointer :: meshPool
    type (mpas_pool_type), pointer :: statePool
    type (mpas_pool_type), pointer :: tracersPool
    type (mpas_pool_type), pointer :: verticalMeshPool
    type (mpas_pool_type), pointer :: forcingPool

    ! local variables
    integer :: iCell, iEdge, iVertex, k, idx
    real (kind=RKIND) :: yMin, yMax, xMin, xMax, dcEdgeMin, dcEdgeMinGlobal
    real (kind=RKIND) :: yMinGlobal, yMaxGlobal, yMidGlobal, xMinGlobal, xMaxGlobal
    real (kind=RKIND) :: localVar1, localVar2
    real (kind=RKIND), dimension(:), pointer :: interfaceLocations

    ! Define dimension pointers
    integer, pointer :: nCellsSolve, nEdgesSolve, nVerticesSolve, nVertLevels, nVertLevelsP1
    integer, pointer :: index_temperature, index_salinity
    integer, pointer :: maxEdges

    ! Define variable pointers
    logical, pointer :: on_a_sphere
    integer, dimension(:), pointer :: minLevelCell, maxLevelCell
    integer, dimension(:), pointer :: nEdgesOnCell
    integer, dimension(:,:), pointer :: verticesOnCell, verticesOnEdge
    integer, dimension(:,:), pointer :: cellsOnEdge, cellsOnVertex
    real (kind=RKIND), dimension(:), pointer :: xCell, yCell, refBottomDepth, refZMid, &
         vertCoordMovementWeights, bottomDepth, fCell, fEdge, fVertex, dcEdge
    real (kind=RKIND), dimension(:,:), pointer:: zMid

    real (kind=RKIND), dimension(:), pointer:: xEdge, yEdge, xVertex, yVertex
    real (kind=RKIND) :: minBottomDepth, maxBottomDepth, globalMaxBottomDepth, globalMinBottomDepth
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers
    
    real (kind=RKIND), dimension(:), pointer :: ssh
    real (kind=RKIND), dimension(:), pointer :: areaCell
    real (kind=RKIND), dimension(:,:), pointer :: edgeNormalVectors
    real (kind=RKIND), dimension(:,:), pointer :: normalVelocity
    ! Elevation Bcs
    real (kind=RKIND), dimension(:), pointer :: tidalInputMask

    
    real (kind=RKIND):: HH, uu, vv
    real (kind=RKIND):: RR, num, den
    real (kind=RKIND):: xshift = 0.0, yshift = 0.0 
    real (kind=RKIND) :: layerThicknessEdgeAverage
    real (kind=RKIND), dimension(:,:), allocatable :: rSubgridPoints, sSubgridPoints
    real (kind=RKIND), dimension(:), allocatable :: subgridBathymetryValues, subgridAreas
    real (kind=RKIND), dimension(:), allocatable :: subgridSshValues
    real (kind=RKIND), dimension(:), allocatable :: subgridUValues, subgridVValues
    real (kind=RKIND), dimension(:), allocatable :: uVelocityAverage, vVelocityAverage
    integer :: nSubgridCell, nSubgridEdge, nSubgridVertex
    integer :: nSubgridTriPerSlice
    integer :: v1, v2
    integer :: c1, c2
    real (kind=RKIND) :: x(3), y(3)
    integer :: slice, nSlice
    real (kind=RKIND) :: deltaZ



! DW
    integer:: i, j, jj
    integer:: nsubgridCellEdge, iEdgeSegment
    real (kind=RKIND):: pi
    real (kind=RKIND), dimension(:,:), allocatable :: cellEdgeBathymetryValues
    real (kind=RKIND), dimension(:), allocatable:: dsEdge
    real (kind=RKIND), dimension(:), allocatable:: xSubgridCell, ySubgridCell
    real (kind=RKIND):: bathymetryMin, bathymetryMax 
    iErr = 0

    call ocn_subgrid_init(domain,iErr)

    if(config_init_configuration .ne. trim('buttermilk_bay')) return

    ! Determine vertical grid for configuration
    call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
    call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
    call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)

    nVertLevels  = config_Buttermilk_bay_vert_levels ;
    nVertLevelsP1 = nVertLevels + 1

    allocate(interfaceLocations(nVertLevelsP1))
    call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations, ocnConfigs ) ;
    !! Mental note: interfaceLocatons = (k-1)/N ;

    ! Initalize min/max values to large positive and negative values
    yMin = 1.0E10_RKIND
    yMax = -1.0E10_RKIND
    xMin = 1.0E10_RKIND
    xMax = -1.0E10_RKIND
    dcEdgeMin = 1.0E10_RKIND

    ! Determine local min and max values.
    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)

       call mpas_pool_get_dimension( meshPool, 'nCellsSolve', nCellsSolve )
       call mpas_pool_get_dimension( meshPool, 'nEdgesSolve', nEdgesSolve )

       call mpas_pool_get_array(meshPool, 'xCell', xCell)
       call mpas_pool_get_array(meshPool, 'yCell', yCell)
       call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

       yMin = min( yMin, minval(yCell(1:nCellsSolve)))
       yMax = max( yMax, maxval(yCell(1:nCellsSolve)))
       xMin = min( xMin, minval(xCell(1:nCellsSolve)))
       xMax = max( xMax, maxval(xCell(1:nCellsSolve)))
       dcEdgeMin = min( dcEdgeMin, minval(dcEdge(1:nEdgesSolve)))

       block_ptr => block_ptr % next
    end do

    ! Determine global min and max values.
    call mpas_dmpar_min_real(domain % dminfo, yMin, yMinGlobal)
    call mpas_dmpar_max_real(domain % dminfo, yMax, yMaxGlobal)
    call mpas_dmpar_min_real(domain % dminfo, xMin, xMinGlobal)
    call mpas_dmpar_max_real(domain % dminfo, xMax, xMaxGlobal)
    call mpas_dmpar_min_real(domain % dminfo, dcEdgeMin, dcEdgeMinGlobal)

    pi = acos(-1.0_RKIND)  
 
    xshift = xMin  
    yshift = (3.0_RKIND*yMin + dcEdgeMin*sin(pi/3.0_RKIND))/3.0_RKIND 
    ! print*, "xMin, yMin = ", xMin, yMin, dcEdgeMin 
    ! print*, "xshift, yshift = ", xshift, yshift  

    !***********************************************************************
    !
    !  Topography
    !
    !***********************************************************************

    call mpas_log_write( 'Reading bathymetry from a NetCDF file')
 
    if (config_Buttermilk_bay_topography_source == 'latlon_file' .or. &
             config_Buttermilk_bay_topography_source == 'xy_file' ) then
       call mpas_log_write( 'Reading topography data from file.')
       call ocn_init_setup_Buttermilk_bay_read_topo(domain, iErr)
    endif

    !--------------------------------------------------------------------
    ! Use this section to set initial values
    !--------------------------------------------------------------------

    block_ptr => domain % blocklist
blockptr_doloop:do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

       call mpas_pool_get_subpool(block_ptr % structs, 'forcing', forcingPool)
       call mpas_pool_get_array(forcingPool, 'tidalInputMask', tidalInputMask)

       call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels) ;
       call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve) ;
       call mpas_pool_get_dimension(meshPool, 'nEdgesSolve', nEdgesSolve) ; 
       call mpas_pool_get_dimension(meshPool, 'nVerticesSolve', nVerticesSolve) ; 
       call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges)
      
       call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)

       call mpas_pool_get_array(meshPool, 'xCell', xCell)
       call mpas_pool_get_array(meshPool, 'yCell', yCell)
       call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
       call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'minLevelCell', minLevelCell)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
       call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
       call mpas_pool_get_array(meshPool, 'areaCell', areaCell)

       call mpas_pool_get_array(meshPool, 'verticesOnCell', verticesOnCell)
       call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
       call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
       call mpas_pool_get_array(meshPool, 'cellsOnVertex', cellsOnVertex)

       call mpas_pool_get_array(meshPool, 'fCell', fCell)
       call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
       call mpas_pool_get_array(meshPool, 'fVertex', fVertex)
       
       call mpas_pool_get_array(meshPool, 'xEdge', xEdge ) 
       call mpas_pool_get_array(meshPool, 'yEdge', yEdge ) 
       call mpas_pool_get_array(meshPool, 'xVertex', xVertex ) 
       call mpas_pool_get_array(meshPool, 'yVertex', yVertex ) 

       call mpas_pool_get_array(statePool, 'zMid', zMid, 1) ;

       call mpas_pool_get_array(statePool, 'ssh', ssh, 1)
       call mpas_pool_get_array(meshPool, 'edgeNormalVectors', edgeNormalVectors ) ;
       call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity ) ;

       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)

       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)

       call mpas_pool_get_array(meshPool, 'subgridWetVolumeCellTable', &
                                subgridWetVolumeCellTable)
       call mpas_pool_get_array(meshPool, 'subgridWetVolumeEdgeTable', &
                                subgridWetVolumeEdgeTable)
       call mpas_pool_get_array(meshPool, 'subgridWetVolumeVertexTable', &
                                subgridWetVolumeVertexTable)
       call mpas_pool_get_array(meshPool, 'subgridWetFractionCellTable', &
                                subgridWetFractionCellTable)
       call mpas_pool_get_array(meshPool, 'subgridWetFractionEdgeTable', &
                                subgridWetFractionEdgeTable)
       call mpas_pool_get_array(meshPool,'subgridWetFractionVertexTable',&
                                subgridWetFractionVertexTable)
       call mpas_pool_get_array(meshPool, 'subgridSshCellTableRange', &
                                subgridSshCellTableRange)
       call mpas_pool_get_array(meshPool, 'subgridSshEdgeTableRange', &
                                subgridSshEdgeTableRange)
       call mpas_pool_get_array(meshPool, 'subgridSshVertexTableRange', &
                                subgridSshVertexTableRange)
       call mpas_pool_get_array(meshPool, 'subgridEdgeBathymetryMean', &
                                subgridEdgeBathymetryMean)
       call mpas_pool_get_array(meshPool, 'subgridVertexBathymetryMean', &
                                subgridVertexBathymetryMean)
       call mpas_pool_get_array(meshPool, 'subgridCellBathymetryMin', &
                                subgridCellBathymetryMin)
       call mpas_pool_get_array(meshPool, 'subgridEdgeBathymetryMin', &
                                subgridEdgeBathymetryMin)
       call mpas_pool_get_array(meshPool, 'subgridVertexBathymetryMin', &
                                subgridVertexBathymetryMin)
       call mpas_pool_get_dimension(meshPool, 'nSubgridTableLevels', &
                                    nSubgridTableLevels)
       call mpas_pool_get_array(meshPool, 'subgridLayerThicknessDebug', &
                                subgridLayerThicknessDebug)


       ! if config_parabolic_bowl_adjust_domain_center == .true.,
       ! Adjust center of the mesh so that its center is located at (0,0)
       if ( config_Buttermilk_bay_adjust_domain ) then
          xCell = xCell - xshift ; 
          yCell = yCell - yshift ;

          xEdge = xEdge - xshift ;
          yEdge = yEdge - yshift ;

          xVertex = xVertex - xshift ;
          yVertex = yVertex - yshift ; 

          ! get min, max coordinates of model domain   !
          ! after adjusting the coordinates             !
          yMin = min( yMin, minval(yCell(1:nCellsSolve)))
          yMax = max( yMax, maxval(yCell(1:nCellsSolve)))
          xMin = min( xMin, minval(xCell(1:nCellsSolve)))
          xMax = max( xMax, maxval(xCell(1:nCellsSolve)))

          ! Determine global min and max values.
          call mpas_dmpar_min_real(domain % dminfo, yMin, yMinGlobal)
          call mpas_dmpar_max_real(domain % dminfo, yMax, yMaxGlobal)
          call mpas_dmpar_min_real(domain % dminfo, xMin, xMinGlobal)
          call mpas_dmpar_max_real(domain % dminfo, xMax, xMaxGlobal)
       end if

       ! Initlialze vector
       call mpas_initialize_vectors(meshPool) ; 

       minLevelCell(:) = 1
       do iCell = 1, nCellsSolve
         ! Set up vertical grid
         maxLevelCell(iCell) = nVertLevels ; ! sigma coordinates 
       end do


       do iCell = 1, nCellsSolve       
          ! Set temperature
          activeTracers(index_temperature, :, iCell) = 10.0_RKIND

          ! Set salinity
          activeTracers(index_salinity, :, iCell) = 30.0_RKIND

          ! Set Coriolis parameters, if other than zero
          fCell(iCell) = config_Buttermilk_bay_coriolis_parameter ; 
      end do

      do iEdge = 1, nEdgesSolve
         fEdge(iEdge) = config_Buttermilk_bay_coriolis_parameter ;  
      end do

      do iVertex = 1, nVerticesSolve
         fVertex(iVertex) = config_Buttermilk_bay_coriolis_parameter ;
      end do


      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Cells
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      print*, "Begin cells", nCellsSolve

      nSubgridTriPerSlice = config_Buttermilk_bay_subgrid_level**2
      allocate(rSubgridPoints(3,maxEdges*nSubgridTriPerSlice), sSubgridPoints(3,maxEdges*nSubgridTriPerSlice))
      call ocn_init_define_subgrid_points(config_Buttermilk_bay_subgrid_level, rSubgridPoints, sSubgridPoints)
      allocate(subgridBathymetryValues(maxEdges*nSubgridTriPerSlice), subgridAreas(maxEdges*nSubgridTriPerSlice))
      allocate(subgridSshValues(maxEdges*nSubgridTriPerSlice))
      allocate(subgridUValues(maxEdges*nSubgridTriPerSlice))
      allocate(subgridVValues(maxEdges*nSubgridTriPerSlice))

      allocate(xSubgridCell(maxEdges*nSubgridTriPerSlice)) 
      allocate(ySubgridCell(maxEdges*nSubgridTriPErSlice)) 
      xSubgridCell = 0.0_RKIND  
      ySubgridCell = 0.0_RKIND 

      open( unit = 101, file = 'BM_bathy.dat', action = 'write' ) ; 
      open( unit = 102, file = 'BM_ssh.dat', action = 'write' ) ; 


      do iCell = 1,nCellsSolve
        !
        ! Evaluate subgrid bathymetry at centers of sub-triangles for cell slices 
        ! (all subdivided triangles for each cell slice are gathered into 
        !  subgridBathymetryValues and subgridAreas)
        !---------------------------------------------------------------

        nSubgridCell = 0  ! Counter for all subgrid triangles over cell slices
        do slice = 1,nEdgesOnCell(iCell) ! Loop over cell slices

          v1 = verticesOnCell(slice,iCell)
          if (slice+1 <= nEdgesOnCell(iCell)) then
            v2 = verticesOnCell(slice+1,iCell)
          else
            v2 = verticesOnCell(1,iCell)
          endif

          ! Cell slice coordinates
          x(1) = xCell(iCell)
          y(1) = yCell(iCell)

          x(2) = xVertex(v1)
          y(2) = yVertex(v1)

          x(3) = xVertex(v2)
          y(3) = yVertex(v2)

          call ocn_init_evaluate_subgrid_data( x, y, nSubgridTriPerSlice, nSubgridCell, rSubgridPoints, sSubgridPoints, &
                                         subgridBathymetryValues, subgridAreas, subgridSshValues, &
                                         subgridUValues, subgridVValues, xSubgridCell, ysubgridCell ) !{{{
        enddo
      
        do jj = 1, nSubgridCell
           WRITE( 101, '(I6,3F16.8)' ) iCell, xSubgridCell(jj), ysubgridCell(jj), subgridBathymetryValues(jj) ; 
           WRITE( 102, '(I6,3F16.8)' ) iCell, xSubgridCell(jj), ysubgridCell(jj), subgridSshValues(jj) ; 
        end do 

        ! Evaluate bounds of look-up table range
        !---------------------------------------------------------------        
        bathymetryMin = maxval( subgridBathymetryValues(1:nSubgridCell) ) 
        bathymetryMax = minval( subgridBathymetryValues(1:nSubgridCell) )

        if ( abs(bathymetryMin - bathymetryMax) > 100.0*eps ) then
!            subgridSshCellTableRange(1,iCell) = -maxval(subgridBathymetryValues(1:nSubgridCell)) + config_drying_min_cell_height + eps
!            subgridSshCellTableRange(2,iCell) = -minval(subgridBathymetryValues(1:nSubgridCell))
            subgridSshCellTableRange(1,iCell) = -maxval(subgridBathymetryValues(1:nSubgridCell))
            subgridSshCellTableRange(2,iCell) = -minval(subgridBathymetryValues(1:nSubgridCell))
        else
            ! flat bathy !
!            subgridSshCellTableRange(1,iCell) =  -bathymetryMin + config_drying_min_cell_height + eps
!            subgridSshCellTableRange(2,iCell) =  -bathymetryMin + 2.0*config_drying_min_cell_height + eps  
            subgridSshCellTableRange(1,iCell) =  -bathymetryMin 
            subgridSshCellTableRange(2,iCell) =  -bathymetryMin + 2.0*config_drying_min_cell_height  
        endif 

        ! Evaluate subgrid bathymetry
        !---------------------------------------------------------------
        ! DW
        bottomDepth(iCell) = sum(subgridBathymetryValues(1:nSubgridCell)*subgridAreas(1:nSubgridCell))/sum(subgridAreas(1:nSubgridCell))
        subgridCellBathymetryMin(iCell) = maxval(subgridBathymetryValues(1:nSubgridCell))

        ! Vertical integration of wet fraction
        !---------------------------------------------------------------

        call ocn_init_vertical_integration(iCell,subgridSshCellTableRange, nSubgridCell, subgridBathymetryValues, subgridAreas, &
                                           subgridWetVolumeCellTable, subgridWetFractionCellTable)

        ! Evaluate wet layerThickness average
        !---------------------------------------------------------------
!!$        call ocn_init_wet_average(nSubgridCell, subgridBathymetryValues, subgridSshValues, subgridAreas, layerThickness(1,iCell))
        call ocn_init_wet_average_ssh(nSubgridCell, subgridBathymetryValues, subgridSshValues, subgridAreas, ssh(iCell))!{{{
      enddo

      do iCell = 1, nCellsSolve
          call ocn_subgrid_ssh_lookup( config_drying_min_cell_height, & 
                                       subgridWetVolumeCellTable(:,iCell), &
                                       subgridSshCellTableRange(:,iCell),&
                                       bottomDepth(iCell),&
                                       subgridCellBathymetryMin(iCell),&
                                       subgridSShCellTableRange(3,iCell) )

         if ( ssh(iCell) < subgridSshCellTableRange(3,iCell) ) then
                ssh(iCell) = subgridSshCellTableRange(3,iCell) ;  
         end if 
      end do

      close( unit = 101 ) ;
      close( unit = 102 ) ; 

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Edges
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      print*, "Begin edges", nEdgesSolve

      allocate(uVelocityAverage(nEdgesSolve))
      allocate(vVelocityAverage(nEdgesSolve))

      nSubgridCellEdge=config_Buttermilk_bay_subgrid_level 
      allocate( cellEdgeBathymetryValues(3,nSubgridCellEdge) )   
      allocate( dsEdge(nSubgridCellEdge) ) 
      cellEdgeBathymetryValues = -99999 ; 


      do iEdge = 1,nEdgesSolve
        !--------------------------------------------------------------
        ! Evaluate subgrid bathymetry at centers of sub-triangles for edge slices
        ! (all subdivided triangles for each edge slice are gathered into 
        !  subgridBathymetryValues and subgridAreas)
        !---------------------------------------------------------------

        nSlice = 0
        do slice = 1,2
          if (cellsOnEdge(slice,iEdge) <= nCellsSolve) then
            nSlice = nSlice + 1
          endif
        enddo

        nSubgridEdge = 0  ! Counter for all subgrid triangles over edge slices
        do slice = 1,nSlice ! Loop over edge slices

          ! Edge slice coordinates
          x(1) = xVertex(verticesOnEdge(1,iEdge))
          y(1) = yVertex(verticesOnEdge(1,iEdge))

          x(2) = xVertex(verticesOnEdge(2,iEdge))
          y(2) = yVertex(verticesOnEdge(2,iEdge))

          x(3) = xCell(cellsOnEdge(slice,iEdge))
          y(3) = yCell(cellsOnEdge(slice,iEdge))


          call ocn_init_evaluate_subgrid_data(x, y, nSubgridTriPerSlice, nSubgridEdge, rSubgridPoints, sSubgridPoints, &
                                              subgridBathymetryValues, subgridAreas, subgridSshValues, subgridUValues, subgridVValues, &
                                              xSubgridCell, ySubgridCell )
  
        enddo

        ! Evaluate velocity average
        !---------------------------------------------------------------
        call ocn_init_wet_average(nSubgridCell, subgridBathymetryValues, subgridSshValues, subgridAreas, layerThicknessEdgeAverage, &
                                  subgridUValues, subgridVValues, uVelocityAverage(iEdge), vVelocityAverage(iEdge))

        if ( .NOT. config_Buttermilk_bay_subgrid_edge_bathymetry_max_pixel ) then
           ! Evaluate bounds of look-up table range
           !---------------------------------------------------------------

!           subgridSshEdgeTableRange(1,iEdge) = -maxval(subgridBathymetryValues(1:nSubgridEdge)) + config_drying_min_cell_height + eps
!           subgridSshEdgeTableRange(2,iEdge) = -minval(subgridBathymetryValues(1:nSubgridEdge))
           subgridSshEdgeTableRange(1,iEdge) = -maxval(subgridBathymetryValues(1:nSubgridEdge))
           subgridSshEdgeTableRange(2,iEdge) = -minval(subgridBathymetryValues(1:nSubgridEdge))

           ! Evaluate subgrid bathymetry
           !---------------------------------------------------------------

           subgridEdgeBathymetryMean(iEdge) = sum(subgridBathymetryValues(1:nSubgridEdge))/real(nSubgridEdge,RKIND) 
           subgridEdgeBathymetryMin(iEdge)  = maxval(subgridBathymetryValues(1:nSubgridEdge))

           ! Vertical integration of wet fraction
           !---------------------------------------------------------------

           call ocn_init_vertical_integration(iEdge,subgridSshEdgeTableRange, nSubgridEdge, subgridBathymetryValues, subgridAreas, &
                                           subgridWetVolumeEdgeTable, subgridWetFractionEdgeTable)
        else
           ! DW: Use the higher values of the pair of subcells along the
           !     cell edge
           cellEdgeBathymetryValues(1,:) = subgridBathymetryValues(1:2*nSubgridCellEdge - 1:2) 
           if ( nslice > 1 ) then
             cellEdgeBathymetryValues(2,:) = &
                subgridBathymetryValues(nSubgridTriPerSlice+2*nsubgridCellEdge - 1:nSubgridTriPerSlice+1:-2)
           else
             cellEdgeBathymetryValues(2,:) = cellEdgeBathymetryValues(1,:)     
           endif

           do iEdgeSegment = 1, nSubgridCellEdge
              cellEdgeBathymetryValues(3,iEdgeSegment) = minval(cellEdgeBathymetryValues(1:2,iEdgeSegment) ) 
           end do 
           dsEdge(:) = sqrt( (x(2) - x(1))*(x(2) - x(1)) + (y(2) - y(1))*(y(2)- y(1)) )/nSubgridCellEdge 


           ! Evaluate bounds of look-up table range
           !---------------------------------------------------------------

!           subgridSshEdgeTableRange(1,iEdge) = -maxval(CellEdgeBathymetryValues(3,1:nSubgridCellEdge)) + config_drying_min_cell_height + eps
!           subgridSshEdgeTableRange(2,iEdge) = -minval(CellEdgeBathymetryValues(3,1:nSubgridCellEdge))
           subgridSshEdgeTableRange(1,iEdge) = -maxval(CellEdgeBathymetryValues(3,1:nSubgridCellEdge))
           subgridSshEdgeTableRange(2,iEdge) = -minval(CellEdgeBathymetryValues(3,1:nSubgridCellEdge))

           ! Evaluate bounds of look-up table range
           !---------------------------------------------------------------

           subgridEdgeBathymetryMean(iEdge) = sum(cellEdgeBathymetryValues(3,1:nSubgridCellEdge))/real(nSubgridCellEdge,RKIND) 
           subgridEdgeBathymetryMin(iEdge)  = maxval(cellEdgeBathymetryValues(3,1:nSubgridCellEdge))

           ! Vertical integration of wet fraction
           !---------------------------------------------------------------
           call ocn_init_vertical_integration( iEdge, subgridSshEdgeTableRange, &
                      nSubgridCellEdge, cellEdgeBathymetryValues(3,:), dsEdge, subgridWetVolumeEdgeTable, subgridWetFractionEdgeTable )
        endif

      end do

      !
      ! find an ssh value corresponding to drying_min_cell-height 
      ! of each edge 
      do iEdge = 1,nEdgesSolve
          call ocn_subgrid_ssh_lookup( config_drying_min_cell_height, & 
                                       subgridWetVolumeEdgeTable(:,iEdge),&
                                       subgridSshEdgeTableRange(:,iEdge),&
                                       subgridEdgeBathymetryMean(iEdge),&
                                       subgridEdgeBathymetryMin(iEdge),&
                                       subgridSshEdgeTableRange(3,iEdge) )
      end do


      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Vertex
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      print*, "Begin vertex", nVerticesSolve

vertex:do iVertex = 1,nVerticesSolve

        ! Evaluate subgrid bathymetry at centers of sub-triangles for vertex triangle
        !---------------------------------------------------------------

        nSlice = 0
        do slice = 1,3
          if (cellsOnVertex(slice,iVertex) <= nCellsSolve) then
            nSlice = nSlice + 1
          endif
        enddo

        if (nSlice < 3) then
           cycle vertex
        endif

        nSubgridVertex = 0  ! Counter for all subgrid triangles over edge slices
        do slice = 1,nSlice

          c1 = cellsOnVertex(slice,iVertex)
          if (slice < 3) then
            c2 = cellsOnVertex(slice+1,iVertex)  
          else
            c2 = cellsOnVertex(1,iVertex)
          endif

          ! Vertex slice coordinates
          x(1) = xCell(c1)
          y(1) = yCell(c1)

          x(2) = xCell(c2)
          y(2) = yCell(c2)

          x(3) = xVertex(iVertex)
          y(3) = yVertex(iVertex)

          call ocn_init_evaluate_subgrid_data(x, y, nSubgridTriPerSlice, nSubgridVertex, rSubgridPoints, sSubgridPoints, &
                                              subgridBathymetryValues, subgridAreas)
        enddo

        ! Evaluate bounds of look-up table range
        !---------------------------------------------------------------
!        subgridSshVertexTableRange(1,iVertex) = -maxval(subgridBathymetryValues(1:nSubgridVertex)) + config_drying_min_cell_height + eps
!        subgridSshVertexTableRange(2,iVertex) = -minval(subgridBathymetryValues(1:nSubgridVertex))
        subgridSshVertexTableRange(1,iVertex) = -maxval(subgridBathymetryValues(1:nSubgridVertex))
        subgridSshVertexTableRange(2,iVertex) = -minval(subgridBathymetryValues(1:nSubgridVertex))


        ! Evaluate subgrid bathymetry
        !---------------------------------------------------------------
        subgridVertexBathymetryMean(iVertex) = sum(subgridBathymetryValues(1:nSubgridVertex)*subgridAreas(1:nSubgridVertex))/sum(subgridAreas(1:nSubgridVertex))
        subgridVertexBathymetryMin(iVertex)  = maxval(subgridBathymetryValues(1:nSubgridVertex))


        ! Vertical integration of wet fraction
        !---------------------------------------------------------------

        call ocn_init_vertical_integration(iVertex,subgridSshVertexTableRange, nSubgridVertex, subgridBathymetryValues, subgridAreas, &
                                           subgridWetVolumeVertexTable, subgridWetFractionVertexTable)

      enddo vertex

      !
      ! find an ssh value corresponding to drying_min_cell-height 
      ! of each edge 
      do iVertex = 1,nVerticesSolve

          call ocn_subgrid_ssh_lookup( config_drying_min_cell_height, & 
                                       subgridWetVolumeVertexTable(:,iVertex),&
                                       subgridSshVertexTableRange(:,iVertex),&
                                       subgridVertexBathymetryMean(iVertex),&
                                       subgridVertexBathymetryMin(iVertex),&
                                       subgridSshVertexTableRange(3,iVertex) )
      end do

       ! Find max bottom depth
       maxBottomDepth = maxval( bottomDepth ) ; 
       minBottomDepth = minval( bottomDepth ) ; 
       call mpas_dmpar_max_real( domain % dminfo, maxBottomDepth, globalMaxBottomDepth ) ;
       call mpas_dmpar_min_real( domain % dminfo, minBottomDepth, globalMinBottomDepth ) ;

       ! Set refBottomDepth and refZMid
       do k = 1, nVertLevels
         refBottomDepth(k) = globalMaxBottomDepth*interfaceLocations(k+1) ; 
         refZMid(k) = -0.5_RKIND*( interfaceLocations(k+1) + interfaceLocations(k))*globalMaxBottomDepth ; 
       end do

       ! Set vertCoordMovementWeights
       vertCoordMovementWeights(:) = 1.0_RKIND

       ! Set velocity
       do iEdge = 1, nEdgesSolve

         if (config_use_subgrid_wetting_drying) then
            do k = 1, nVertLevels 
               normalVelocity(k,iEdge) = uVelocityAverage(iEdge)*edgeNormalVectors(1,iEdge) &
                                       + vVelocityAverage(iEdge)*edgeNormalVectors(2,iEdge) ;
            end do
         else
            call ocn_init_Buttermilk_bay_velocity(xEdge(iEdge), yEdge(iEdge), uu, vv)
            do k = 1, nVertLevels 
               normalVelocity(k,iEdge) = uu*edgeNormalVectors(1,iEdge) + vv*edgeNormalVectors(2,iEdge) ;
            end do
         end if
       end do

       ! Set layer thickness and ssh
       if (config_use_wetting_drying) then

         do iCell = 1, nCellsSolve
           ! Set up vertical grid
           maxLevelCell(iCell) = nVertLevels ; ! sigma coordinates 
          end do

          open( unit = 103, file = 'ssh_cell.dat', action = 'write') 
          do iCell = 1, nCellsSolve
            !
            ! make sure depth is thick enough via ssh = TOTAL_DEPTH - bottomDepth
            ! add a thin layer of nlayer*config_drying_min_cellhight 
            ! 
 
            if (config_use_subgrid_wetting_drying) then

!              call ocn_subgrid_ssh_lookup(layerThickness(1,iCell),&
!                                          subgridWetVolumeCellTable(:,iCell),&
!                                          subgridSshCellTableRange(:,iCell),&
!                                          bottomDepth(iCell),&
!                                          subgridCellBathymetryMin(iCell),&
!                                          ssh(iCell))
              call ocn_subgrid_layer_thickness_lookup(ssh(iCell), & 
                                             subgridWetVolumeCellTable(:,iCell), &
                                             subgridSshCellTableRange(:,iCell),&
                                             bottomDepth(iCell),&
                                             LayerThickness(1,iCell))

            else
!              call ocn_init_Buttermilk_bay_bathymetry(xCell(iCell),yCell(iCell),bottomDepth(iCell))
!              call ocn_init_Buttermilk_bay_ssh(xCell(iCell),yCell(iCell),bottomDepth(iCell),ssh(iCell))
              ssh(iCell) = - bottomDepth(iCell) + &
                max(ssh(iCell) + bottomDepth(iCell), &
                maxLevelCell(iCell)*(config_drying_min_cell_height + eps))

              do k = 1, maxLevelCell(iCell)
                layerThickness(k,iCell) = max(config_drying_min_cell_height + eps, &
                  (ssh(iCell) + bottomDepth(iCell))/real(maxLevelCell(iCell),RKIND))
               
                if (layerThickness(k,iCell) < config_drying_min_cell_height) then
                  call mpas_log_write('layerThickness($i,$i)=$r', MPAS_LOG_CRIT, &
                    intArgs=(/k,iCell/), &
                    realArgs=(/layerThickness(k,iCell)/))
                end if
              end do
           endif
          
          
            do k = 1, maxLevelCell(iCell)
              restingThickness(k,iCell) = bottomDepth(iCell)/maxLevelCell(iCell)
            end do

            write(103,*) iCell, ssh(iCell), LayerThickness(1,iCell) ; 

          end do
          close( unit = 103 ) ;

       end if 

       block_ptr => block_ptr % next
    end do blockptr_doloop

     do iCell = 1,nCellsSolve
         call ocn_subgrid_layer_thickness_lookup(ssh(iCell), & 
                                              subgridWetVolumeCellTable(:,iCell), &
                                              subgridSshCellTableRange(:,iCell),&
                                              bottomDepth(iCell),&
                                              subgridLayerThicknessDebug(iCell))
     enddo
     !

     open( unit = 101, file = 'bcmask.dat', action = 'write' ) ; 
     ! Set tidal boundary mask
     do iCell = 1, nCellsSolve
       tidalInputMask(iCell) = 0.0_RKIND
       if ( yCell(iCell) < (yMin+(dcEdgeMin*sin(pi/3.0_RKIND)/2.0_RKIND))  &  
         .and.  yCell(iCell) > (yMin-(dcEDgeMin*sin(pi/3.0_RKIND)/2.0_RKIND)) ) then

          if ( (xCell(iCell) - dcEdgeMin/2.0_RKIND) > 2048.0_RKIND  &
           .and.  (xCell(iCell) + dcEdgeMin/2.0_RKIND) < 3072.0_RKIND ) then
              tidalInputMask(iCell) = 1.0_RKIND
          end if
        ! spread it over multiple cells
        ! if (yCell(iCell) > (25.0e3 - 3*dcEdgeMinGlobal)) then
        !  tidalInputMask(iCell) = exp(-((yCell(iCell)-25.0e3)/dcEdgeMinGlobal)**2.0)
       end if
       write(101,'(I10,F16.8)') iCell, tidalInputMask(iCell) ; 
     end do
     
     close( 101 ) ;   
    !

    deallocate(interfaceLocations)
    if (config_global_ocean_topography_source == 'latlon_file') then
        call mpas_log_write( 'Cleaning up topography IC fields')
        call ocn_init_Buttermilk_bay_destroy_topo_fields()
    endif
    !--------------------------------------------------------------------

    print*, "****** End Butter milk bay init *****" ;

    return ;  
  end subroutine ocn_init_setup_Buttermilk_bay!}}}

!***********************************************************************
!
!  routine ocn_init_define_subgrid_points
!
!> \brief   Define subgrid points on reference triangle
!> \author  Steven Brus 
!> \date    November 2022 
!> \details Gives the r and s coordinates of the subgrid triangle vertices
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_define_subgrid_points(nSubgridLevel, rSubgridPoints, sSubgridPoints)!{{{
           
      implicit none 

      integer, intent(in) :: nSubgridLevel
      real (kind=RKIND), dimension(:,:), intent(inout) :: rSubgridPoints, sSubgridPoints

      integer :: i,j,k
      integer :: nSubgridTri
      real (kind=RKIND) :: dx
      real (kind=RKIND), dimension(:), allocatable :: xPoints

   !--------------------------------------------------------------------

      ! Equi-spaced nodes on -1,+1
      allocate(xPoints(nSubgridLevel+1))
      dx = 2.0_RKIND/real(nSubgridLevel,RKIND)
      xPoints(1) = -1.0_RKIND
      do i = 2,nSubgridLevel+1
        xPoints(i) = xPoints(i-1)+dx   
      enddo      

      !   * 
      !   |\
      !   | \
      !   |  \
      !   |   \
      ! s *----*
      !   |\ u |\
      !   | \  | \
      !   |  \ |  \
      !   | l \| l \
      !   *----*----*
      !        r

      ! Trianglulate tensor product of equi-spaced nodes on reference triangle
      ! r - horizontal coordinate (i index), s - vertical coordinate (j index)
      nSubgridTri = 0
      do j = 1,nSubgridLevel
        do i = 1,nSubgridLevel+1 - j 

          nSubgridTri = nSubgridTri+1 

          ! lower triangle in pair (see l in triangle above)
          rSubgridPoints(1,nSubgridTri) = xPoints(i)
          sSubgridPoints(1,nSubgridTri) = xPoints(j)
          
          rSubgridPoints(2,nSubgridTri) = xPoints(i+1)
          sSubgridPoints(2,nSubgridTri) = xPoints(j)

          rSubgridPoints(3,nSubgridTri) = xPoints(i)
          sSubgridPoints(3,nSubgridTri) = xPoints(j+1)

          ! upper triangle in pair (see u in triangle above, doesn't occur next to hypotenuse)          
          if (i < nSubgridLevel+1 - j) then
            
            nSubgridTri = nSubgridTri + 1

            rSubgridPoints(1,nSubgridTri) = xPoints(i+1)
            sSubgridPoints(1,nSubgridTri) = xPoints(j)
            
            rSubgridPoints(2,nSubgridTri) = xPoints(i+1)
            sSubgridPoints(2,nSubgridTri) = xPoints(j+1)

            rSubgridPoints(3,nSubgridTri) = xPoints(i)
            sSubgridPoints(3,nSubgridTri) = xPoints(j+1)

          endif

        enddo
      enddo

      print*, "exit ocn_init_define_subgrid_points"

   
   !--------------------------------------------------------------------

   end subroutine ocn_init_define_subgrid_points!}}}

!***********************************************************************
!
!  routine ocn_init_evaluate_subgrid_data
!
!> \brief   Evaluate subgrid infromation 
!> \author  Steven Brus
!> \date    November 2022
!> \details Evaluate data on subgrid triangles for a given triangluar grid cell region 
!>  
!
!-----------------------------------------------------------------------


!   subroutine ocn_init_evaluate_subgrid_data(xTri, yTri, nSubgridTri, nSubgridCV, rSubgridPoints, sSubgridPoints, &
!                                             subgridBathymetryValues, subgridAreas, &
!                                             subgridSshValues, subgridUValues, subgridVValues)!{{{
   subroutine ocn_init_evaluate_subgrid_data( xTri, yTri, nSubgridTri, nSubgridCV, rSubgridPoints, sSubgridPoints, & 
                                   subgridBathymetryValues, subgridAreas, &
                                   subgridSshValues, subgridUValues, subgridVValues, &
                                   xSubgridCell, ySubgridCell ) !{{{

           
      implicit none 
      real (kind=RKIND), intent(inout) :: xTri(3), yTri(3)
      integer, intent(in) :: nSubgridTri
      integer, intent(inout) :: nSubgridCV
      real (kind=RKIND), dimension(:,:), intent(in) :: rSubgridPoints, sSubgridPoints
      real (kind=RKIND), dimension(:), intent(inout) :: subgridBathymetryValues
      real (kind=RKIND), dimension(:), intent(inout) :: subgridAreas
      real (kind=RKIND), dimension(:), intent(inout),optional :: subgridSshValues
      real (kind=RKIND), dimension(:), intent(inout),optional :: subgridUValues, subgridVValues

      real (kind=RKIND), dimension(:), intent(inout), optional :: xSubgridCell, ySubgridCell

      real (kind=RKIND) :: rCenter, sCenter
      real (kind=RKIND) :: xCenter, yCenter
      real (kind=RKIND) :: xSubgridPoints(3), ySubgridPoints(3)
      real (kind=RKIND) :: x, y
      real (kind=RKIND) :: area


      integer :: iPt,i

   !--------------------------------------------------------------------

     ! Coordinates of physical triangle
     ! (ensure counter-clockwise numering)
     call ocn_init_tri_area(xTri(:), yTri(:), area)
     if (area < 0.0_RKIND) then
      x = xTri(1)
      y = yTri(1) 
      xTri(1) = xTri(2) 
      yTri(1) = yTri(2)
      xTri(2) = x ;
      yTri(2) = y ; 
      area = abs(area) ; 
     endif

      do iPt = 1,nSubgridTri

        ! Counter over all subcells within cell/edge/vertex control volume
        nSubgridCV = nSubgridCV + 1

        ! Center sub-triangle (on reference triangle)
        rCenter = sum(rSubgridPoints(:,iPt))/3.0_RKIND
        sCenter = sum(sSubgridPoints(:,iPt))/3.0_RKIND

        ! Transformation of sub-triangle center to physical coordinates
        call ocn_init_tri_coordinate_transform(rCenter, sCenter, xTri, yTri, xCenter, yCenter)

        ! Evaluate bathymetry
        call ocn_init_Buttermilk_bay_bathymetry(xCenter, yCenter, subgridBathymetryValues(nSubgridCV))

        ! Transformation of sub-triangle vertices to physical coordinates
        do i = 1,3
           call ocn_init_tri_coordinate_transform(rSubgridPoints(i,iPt), sSubgridPoints(i,iPt), xTri, yTri, xSubgridPoints(i), ySubgridPoints(i))
        enddo
 
        ! Calculate area of sub-triangle
        call ocn_init_tri_area(xSubgridPoints(:), ySubgridPoints(:), subgridAreas(nSubgridCV))

        ! Optionally evalulate ssh !
        if (present(subgridsshValues)) then
           call ocn_init_Buttermilk_bay_ssh(xCenter, yCenter, subgridBathymetryValues(nSubgridCV), subgridSshValues(nSubgridCV))
        endif

        if (present(subgridUValues).and.present(subgridVValues)) then
           call ocn_init_Buttermilk_bay_velocity(xCenter, yCenter, subgridUValues(nSubgridCV), subgridVValues(nSubgridCV))
        endif   

        if ( present(xSubgridCell) ) then
            xSubgridCell(nSubgridCV) = xCenter ;
        endif

        if ( present(ySubgridCell) ) then
            ySubgridCell(nSubgridCV) = yCenter ;
        end if
      enddo

   !--------------------------------------------------------------------

   end subroutine ocn_init_evaluate_subgrid_data!}}}

!***********************************************************************
!
!  routine ocn_init_parabolic_bowl_bathymetry
!
!> \brief   Compute bathymetry 
!> \author  Steven Brus, D. Wirasaet
!> \date    November 2022
!> \details Return the value of the bathymetry at a given x,y point
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_Buttermilk_bay_bathymetry(x, y, depth)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x, y
      real (kind=RKIND), intent(out) :: depth

      integer:: ix(2)
      real (kind=RKIND) :: xc(2)
      real (kind=RKIND) :: dx(2), x0(2), xN(2), bath(4), f(4), den
      real (kind=RKIND) :: xv(2), yv(2) 

      !--------------------------------------------------------------------
      !
      ! DEM must be on a uniform grid
      x0 = (/ topoLon%array(1), topoLat%array(1) /) ;
      xN = (/ topoLon%array(nLonTopo), topoLat%array(nLatTopo) /) ; 
      dx = (/ topoLon%array(2) - topoLon%array(1), &
              topoLat%array(2) - topoLat%array(1) /) ;
     
      ! Bilienar interpolation
      xc = (/ x, y /) ; 
      ix = floor( (xc - x0)/dx ) + 1 ;
         
      if ( ( ix(1) >= 1 .and. ix(1) < nLonTopo ) .and. &
              ( ix(2) >= 1 .and. ix(2) < nLatTopo ) ) then 
        ! include the west and soutth bourdaries of 
        ! a given uniform raster DEM       
        bath(1) = TopoIC%array(ix(1),ix(2))  ;
        bath(2) = TopoIC%array(ix(1)+1,ix(2)) ; 
        bath(3) = TopoIC%array(ix(1)+1,ix(2)+1) ; 
        bath(4) = TopoIC%array(ix(1),ix(2)) ;

        xv(1) = topoLon%array(ix(1)) ;
        xv(2) = topoLon%array(ix(1)+1) ; 

        yv(1) = topoLat%array(ix(2)) ;
        yv(2) = topoLat%array(ix(2)+1) ;

        den = dx(1)*dx(2) ;
       
        f(1) = ( xc(1) - xv(2) )*( xc(2) - yv(2) )/den ;
        f(2) = ( xc(1) - xv(1) )*( xc(2) - yv(2) )/(-den) ;
        f(3) = ( xc(1) - xv(1) )*( xc(2) - yv(1) )/den  ;
        f(4) = ( xc(1) - xv(2) )*( xc(2) - yv(1) )/(-den) ; 

        depth = sum( bath*f ) ;              
      else
        ! nearest extrapolation !
        ix(1) = merge( 1, ix(1), ix(1) < 1 ) ; 
        ix(1) = merge( nLonTopo, ix(1), ix(1) >= nLonTopo ) ; 
        ix(2) = merge( 1, ix(2), ix(2) < 1 )  ;
        ix(2) = merge( nLatTopo, ix(2), ix(2) >= nLatTopo ) ; 
      
        depth = TopoIC%array(ix(1),ix(2)) ;
      endif       
            
      return ;       
   end subroutine ocn_init_Buttermilk_bay_bathymetry!}}}

!***********************************************************************
!
!  routine ocn_init_parabolic_bowl_ssh
!
!> \brief   Compute initial ssh field
!> \author  Steven Brus, D. Wirasaet
!> \date    November 2022
!> \details Use exact solution to compute ssh field for initial conditions
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_Buttermilk_bay_ssh(x, y, bottomDepth, ssh)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x, y
      real (kind=RKIND), intent(in) :: bottomDepth
      real (kind=RKIND), intent(out) :: ssh
      real (kind=RKIND) :: RR

   !--------------------------------------------------------------------

!      RR = sqrt(x**2 + y**2) ;
! 
!      ssh = (sqrtOneMC2/oneMC) - 1.0_RKIND - ((RR**2)/(LL**2))*( (oneMC2/(oneMC**2)) - 1.0_RKIND ) ;
!      ssh = config_parabolic_bowl_b0*ssh ; 
!      !ssh = - bottomDepth + max(ssh + bottomDepth, config_drying_min_cell_height + eps)

      ssh = 0.0_RKIND ;
      ! ssh = merge( -bottomDepth + eps, ssh, ssh + bottomDepth < eps ) ;
      ! ssh = - bottomDepth + max( ssh + bottomDepth, config_drying_min_cell_height + eps ) ; 

      ssh = - bottomDepth + max( ssh + bottomDepth, 0.0_RKIND ) ; 

      return 
   !--------------------------------------------------------------------

   end subroutine ocn_init_Buttermilk_bay_ssh!}}}

!***********************************************************************
!
!  routine ocn_init_parabolic_bowl_velocity
!
!> \brief   Compute initial velocity field
!> \author  Steven Brus, D. Wirasaet
!> \date    November 2022
!> \details Use exact solution to comupte velocity field for initial conditions
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_Buttermilk_bay_velocity(x, y, u, v)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x, y
      real (kind=RKIND), intent(out) :: u, v 
      real (kind=RKIND) :: RR, HH

   !--------------------------------------------------------------------

      u = 0.0_RKIND ; 
      v = 0.0_RKIND ;
   !--------------------------------------------------------------------

   end subroutine ocn_init_Buttermilk_bay_velocity!}}}

!***********************************************************************
!
!  routine ocn_init_tri_area
!
!> \brief   Compute triangle area
!> \author  Steven Brus
!> \date    November 2022
!> \details Compute area of triangle given vertex coordinates
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_tri_area(x, y, area)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x(3), y(3)
      real (kind=RKIND), intent(out) :: area

   !--------------------------------------------------------------------

      area = 0.5_RKIND*(x(2)*y(3) - y(2)*x(3) - x(1)*y(3) + y(1)*x(3) + x(1)*y(2) - y(1)*x(2))

   !--------------------------------------------------------------------

   end subroutine ocn_init_tri_area!}}}

!***********************************************************************
!
!  routine ocn_init_tri_coordinate_transform
!
!> \brief   Transform reference triangle coordinates to mesh coordinates
!> \author  Steven Brus
!> \date    November 2022
!> \details Evaluate r,s reference coordinates in x,y mesh
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_tri_coordinate_transform(r, s, xTri, yTri, x, y)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: r, s
      real (kind=RKIND), intent(in) :: xTri(3), yTri(3)
      real (kind=RKIND), intent(out) :: x, y

   !--------------------------------------------------------------------

      ! Transformation of sub-triangle center to physical coordinates
      x = 0.5_RKIND*(-(r+s)*xTri(1) + (1.0_RKIND+r)*xTri(2) + (1.0_RKIND+s)*xTri(3))
      y = 0.5_RKIND*(-(r+s)*yTri(1) + (1.0_RKIND+r)*yTri(2) + (1.0_RKIND+s)*yTri(3))

   !--------------------------------------------------------------------

   end subroutine ocn_init_tri_coordinate_transform!}}}

!***********************************************************************
!
!  routine ocn_init_vertical_integration
!
!> \brief   Compute the wet volume per unit area lookup table
!> \author  Steven Brus
!> \date    November 2022
!> \details Integrate the wet fraction over discrete ssh values 
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_vertical_integration(iCV, subgridSshTableRange, nSubgridCV, subgridBathymetryValues, subgridAreas, &
                                            subgridWetVolumeTable, subgridWetFractionTable)!{{{
           
      implicit none 

      integer, intent(in) :: iCV
      real (kind=RKIND), dimension(:,:), intent(in) :: subgridSshTableRange
      integer, intent(in) :: nSubgridCV
      real (kind=RKIND), dimension(:), intent(in) :: subgridBathymetryValues, subgridAreas
      real (kind=RKIND), dimension(:,:), intent(inout) :: subgridWetVolumeTable
      real (kind=RKIND), dimension(:,:), intent(inout) :: subgridWetFractionTable

      real (kind=RKIND) :: deltaZ, ssh, pVal
      integer :: lev, tri
      

   !--------------------------------------------------------------------
   
      deltaZ = (subgridSshTableRange(2,iCV)-subgridSshTableRange(1,iCV))/real(nSubgridTableLevels-1,RKIND);

      !  subgridWetVolumeTable(1,iCV) = config_drying_min_cell_height + eps 
      subgridWetVolumeTable(1,iCV) = 0.0_RKIND
      ssh = subgridSshTableRange(1,iCV) + deltaZ
      subgridWetVolumeTable(2,iCV) = subgridWetVolumeTable(1,iCV)*sum(subgridAreas(1:nSubgridCV))
      subgridWetFractionTable(1,iCV) = 0.0_RKIND
      do lev = 2,nSubgridTableLevels
        do tri = 1,nSubgridCV

          pVal = 0.0_RKIND
          if (subgridBathymetryValues(tri) + ssh >= 0.0_RKIND) then
            pVal = 1.0_RKIND
          endif

          subgridWetVolumeTable(lev,iCV) = subgridWetVolumeTable(lev,iCV) + pVal*deltaZ*subgridAreas(tri)
          subgridWetFractionTable(lev,iCV) = subgridWetFractionTable(lev,iCV) + pVal*subgridAreas(tri)
  
        enddo

        if (lev < nSubgridTableLevels) then
          subgridWetVolumeTable(lev+1,iCV) = subgridWetVolumeTable(lev,iCV)
        endif

        subgridWetVolumeTable(lev,iCV) = subgridWetVolumeTable(lev,iCV)/sum(subgridAreas(1:nSubgridCV))
        subgridWetFractionTable(lev,iCV) = subgridWetFractionTable(lev,iCV)/sum(subgridAreas(1:nSubgridCV))

        ssh = ssh + deltaZ
      enddo

   !--------------------------------------------------------------------

   end subroutine ocn_init_vertical_integration!}}}


!***********************************************************************
!
!  routine ocn_init_wet_average
!
!> \brief   Compute thickness and velocity averages over wet area 
!> \author  Steven Brus
!> \date    November 2022
!> \details Compute the thickness and velocity averages over an area
!>          based on the subgrid wet area
!
!  DW:  Shall we rename it to 'ocn_init_grid_average'?  
!-----------------------------------------------------------------------

   subroutine ocn_init_wet_average(nSubgridCV, subgridBathymetryValues, subgridSshValues, subgridAreas, subgridThicknessAverage, &
                                   subgridUValues, subgridVValues, subgridUAverage, subgridVAverage)!{{{
           
      implicit none 

      integer, intent(in) :: nSubgridCV
      real (kind=RKIND), dimension(:), intent(in) :: subgridBathymetryValues, subgridAreas
      real (kind=RKIND), dimension(:), intent(in) :: subgridSshValues
      real (kind=RKIND), intent(inout) :: subgridThicknessAverage
      real (kind=RKIND), dimension(:), intent(in), optional :: subgridUValues, subgridVValues
      real (kind=RKIND), intent(inout), optional :: subgridUAverage, subgridVAverage

      real (kind=RKIND) :: deltaZ, ssh, pVal
      real (kind=RKIND) :: averageDepth, wetArea, layerThicknessValue
      real (kind=RKIND) :: H_int 
      real (kind=RKIND) :: HU_int, HV_int
      logical :: computeVelAverage
      integer :: lev, tri
      

   !--------------------------------------------------------------------
   
      averageDepth = sum(subgridBathymetryValues(1:nSubgridCV))/real(nSubgridCV,RKIND) 

      computeVelAverage = .false.
      if (present(subgridUValues) .and. present(subgridVValues) .and. &
          present(subgridUAverage) .and. present(subgridVAverage)) then
         computeVelAverage = .true.
      endif

      H_int = 0.0_RKIND
      wetArea = 0.0_RKIND
      HU_int = 0.0_RKIND
      HV_int = 0.0_RKIND
      do tri = 1,nSubgridCV

        layerThicknessValue = subgridBathymetryValues(tri) + subgridSshValues(tri)
        if (layerThicknessValue < config_drying_min_cell_height + eps) then
          layerThicknessValue = config_drying_min_cell_height + eps
        endif
        H_int = H_int + layerThicknessValue*subgridAreas(tri)
        wetArea = wetArea + subgridAreas(tri)

        if (computeVelAverage) then
           HU_int = HU_int + layerThicknessValue*subgridUValues(tri)*subgridAreas(tri)
           HV_int = HV_int + layerThicknessValue*subgridVValues(tri)*subgridAreas(tri)
        endif
      enddo

      if (computeVelAverage) then
         subgridUAverage = HU_int/H_int
         subgridVAverage = HV_int/H_int
      endif
      subgridThicknessAverage = H_int/wetArea
      

   !--------------------------------------------------------------------

   end subroutine ocn_init_wet_average!}}}

   ! 
   subroutine ocn_init_wet_average_ssh( nSubgridCV, subgridBathymetryValues, subgridSshValues, subgridAreas, sshWetAverage)!{{{
      implicit none

      integer, intent(in) :: nSubgridCV
      real (kind=RKIND), dimension(:), intent(in) :: subgridBathymetryValues, subgridAreas
      real (kind=RKIND), dimension(:), intent(in) :: subgridSshValues
      real (kind=RKIND), intent(inout) :: sshWetAverage

      real (kind=RKIND) :: deltaZ, ssh, pVal
      real (kind=RKIND) :: averageDepth, wetArea, layerThicknessValue
      integer :: lev, tri


      sshWetAverage = 0.0_RKIND 
      wetArea = 0.0_RKIND 

      do tri = 1, nsubgridCV
        layerThicknessValue = subgridBathymetryValues(tri) + subgridSshValues(tri)

!        if (layerThicknessValue > config_drying_min_cell_height + eps ) then
        if ( layerThicknessValue > 0.0_RKIND ) then 
           sshWetAverage = sshWetAverage + subgridSshValues(tri)*subgridAreas(tri)
           wetArea = wetArea + subgridAreas(tri)
        endif
      end do

      if ( WetArea > 0.0_RKIND ) then
          sshWetAverage = sshWetAverage/WetArea ;
      else
!          sshWetAverage = -maxval(subgridBathymetryValues(1:nsubgridCV)) & 
!              + config_drying_min_cell_height ;
         sshWetAverage = -maxval(subgridBathymetryValues(1:nsubgridCV)) ;     
      endif

      return ;
   end subroutine ocn_init_wet_average_ssh

!***********************************************************************
!
!  routine ocn_init_setup_global_ocean_read_topo
!
!> \brief   Read the topography IC file
!> \author  D. Wirasaet, S. Brus
!> \date    Sep 2023 
!> \details
!>  This routine reads the topography IC file, including latitude and longitude
!>   information for topography data.
!>
!>  Adapted from ocn_init_setup_global_ocean_read_topo
!-----------------------------------------------------------------------
    subroutine ocn_init_setup_Buttermilk_bay_read_topo(domain, iErr)!{{{
       type (domain_type), intent(inout) :: domain
       integer, intent(out) :: iErr

       type (MPAS_Stream_type) :: topographyStream

       iErr = 0

       ! Define stream for depth levels
       call MPAS_createStream(topographyStream, domain % iocontext, &
                config_Buttermilk_bay_topography_file, MPAS_IO_NETCDF, &
                                                   MPAS_IO_READ, ierr=iErr)

       ! Setup topoLat, topoLon, and topoIC fields for stream to be read in
       topoLat % fieldName = trim(config_Buttermilk_bay_topography_lat_varname)
       topoLat % dimSizes(1) = nLatTopo
       topoLat % dimNames(1) = trim(config_Buttermilk_bay_topography_nlat_dimname)
       topoLat % isVarArray = .false.
       topoLat % isPersistent = .true.
       topoLat % isActive = .true.
       topoLat % hasTimeDimension = .false.
       topoLat % block => domain % blocklist
       allocate(topoLat % attLists(1))
       allocate(topoLat % array(nLatTopo))

       topoLon % fieldName = trim(config_Buttermilk_bay_topography_lon_varname)
       topoLon % dimSizes(1) = nLonTopo
       topoLon % dimNames(1) = trim(config_Buttermilk_bay_topography_nlon_dimname)
       topoLon % isVarArray = .false.
       topoLon % isPersistent = .true.
       topoLon % isActive = .true.
       topoLon % hasTimeDimension = .false.
       topoLon % block => domain % blocklist
       allocate(topoLon % attLists(1))
       allocate(topoLon % array(nLonTopo))

       topoIC % fieldName = trim(config_Buttermilk_bay_topography_varname)
       topoIC % dimSizes(1) = nLonTopo
       topoIC % dimSizes(2) = nLatTopo
       topoIC % dimNames(1) = trim(config_Buttermilk_bay_topography_nlon_dimname)
       topoIC % dimNames(2) = trim(config_Buttermilk_bay_topography_nlat_dimname)
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
       topoIC%array = -topoIC%array ; 


       ! Close stream
       call MPAS_closeStream(topographyStream)

       if ( config_Buttermilk_bay_topography_latlon_degrees .and. &
              config_Buttermilk_bay_topography_source == 'latlon_file' ) then
          topoLat % array(:) = topoLat % array(:) * pii / 180.0_RKIND
          topoLon % array(:) = topoLon % array(:) * pii / 180.0_RKIND
       end if

    end subroutine ocn_init_setup_Buttermilk_bay_read_topo!}}}


!***********************************************************************
!
!  routine ocn_init_Buttermilk_bay_destroy_topo_fields
!
!> \brief   Topography field cleanup routine
!> \author  D. Wirasaet and S. Brus
!> \date    Sep 2023
!> \details
!>  This routine destroys the fields that were created to hold topography
!>  initial condition information
!>
!>  NOTE: adapteed from ocn_init_global_ocaen_destroy_topo_fileds
!-----------------------------------------------------------------------

    subroutine ocn_init_Buttermilk_bay_destroy_topo_fields()!{{{
       implicit none 

       deallocate(topoIC % array)
       deallocate(topoLat % array)
       deallocate(topoLon % array)
    end subroutine ocn_init_Buttermilk_bay_destroy_topo_fields!}}}


!***********************************************************************
!
!  routine ocn_init_Buttermilk_bay
!
!> \brief   Validation for this initial condition
!> \author  D. Wirasaet and Steven Brus
!> \date   Sep 2022
!> \details
!>  This routine validates the configuration options for this case.
!> 
!-----------------------------------------------------------------------
   subroutine ocn_init_validate_Buttermilk_bay(configPool, packagePool, iocontext, iErr)!{{{
      implicit none 

   !--------------------------------------------------------------------
      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      ! character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_Buttermilk_bay_vert_levels


      type (mpas_io_context_type), pointer :: iocontext_ptr
      type (MPAS_IO_Handle_type) :: inputFile
      character (len=StrKIND), pointer :: config_init_configuration, &
                                          config_Buttermilk_bay_topography_source, &
                                          config_Buttermilk_bay_topography_file, &
                                          config_Buttermilk_bay_topography_nlat_dimname, &
                                          config_Buttermilk_bay_topography_nlon_dimname 

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)

      if(config_init_configuration .ne. trim('buttermilk_bay')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool,  &
     &      'config_Buttermilk_bay_vert_levels', config_Buttermilk_bay_vert_levels)

      if(config_vert_levels <= 0 .and. config_Buttermilk_bay_vert_levels > 0) then
         config_vert_levels = config_Buttermilk_bay_vert_levels
      else if (config_vert_levels <= 0) then
         call mpas_log_write( 'Validation failed for Buttermilk bay.'// &
     &     'Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
         iErr = 1
      end if


      !---------- adapted from ocn_init_validate_global_ocean
      call mpas_pool_get_config(configPool, 'config_Buttermilk_bay_topography_source', &
                                config_Buttermilk_bay_topography_source)
      call mpas_pool_get_config(configPool, 'config_Buttermilk_bay_topography_file', &
                                config_Buttermilk_bay_topography_file)
      call mpas_pool_get_config(configPool, 'config_Buttermilk_bay_topography_nlat_dimname', &
                                config_Buttermilk_bay_topography_nlat_dimname)
      call mpas_pool_get_config(configPool, 'config_Buttermilk_bay_topography_nlon_dimname', &
                                config_Buttermilk_bay_topography_nlon_dimname)


      call mpas_log_write( config_Buttermilk_bay_topography_source  )  
      call mpas_log_write( config_Buttermilk_bay_topography_file ) 
      call mpas_log_write( config_Buttermilk_bay_topography_nlat_dimname )  
      call mpas_log_write( config_Buttermilk_bay_topography_nlon_dimname ) 

      if (config_Buttermilk_bay_topography_source /= 'latlon_file' .and. &
          config_Buttermilk_bay_topography_source /= 'xy_file') then
         call mpas_log_write( 'Unexpected value for & 
    &               config_Buttermilk_bay_topography_source: ' &
             // trim(config_Buttermilk_bay_topography_source), MPAS_LOG_CRIT)
         iErr = 1
         return
      end if

      ! print*, "message ",  config_Buttermilk_bay_topography_file, config_Buttermilk_bay_topography_source     
      if (config_Buttermilk_bay_topography_file == 'none' .and. &
          (config_Buttermilk_bay_topography_source == 'latlon_file' .or. &
          config_Buttermilk_bay_topography_source == 'xy_file') ) then
         call mpas_log_write( 'Validation failed for Buttermilk bay test case. ' &
           // 'Invalid filename for config_Buttermilk_bay_topography_file ' &
           // config_Buttermilk_bay_topography_file // '  ' &
           // config_Buttermilk_bay_topography_source , MPAS_LOG_CRIT)
         iErr = 1
         return
      end if

      call mpas_log_write( ' in ocn_init_validate_buttermilk_bay '//config_Buttermilk_bay_topography_source ) 

      if (config_Buttermilk_bay_topography_source == 'latlon_file' .or. &
           config_Buttermilk_bay_topography_source == 'xy_file'  ) then

!!         call mpas_log_write( ' before io_open', MPAS_LOG_CRIT ) 
          
!!         inputFile = MPAS_io_open(trim(config_Buttermilk_bay_topography_file), & 
!!                        MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!!         if (iErr /= 0) then
!!            call mpas_log_write( 'could not open file '// &
!!     &              trim(config_Buttermilk_bay_topography_file), MPAS_LOG_CRIT)
!!            return
!!         end if

!!         call mpas_log_write( 'after io_open', MPAS_LOG_CRIT ) 

         nLatTopo = 3585 ; 
         nLonTopo = 3585 ; 
!!         call MPAS_io_inq_dim(inputFile, config_Buttermilk_bay_topography_nlat_dimname, nLatTopo, iErr)
!!         call MPAS_io_inq_dim(inputFile, config_Buttermilk_bay_topography_nlon_dimname, nLonTopo, iErr)

!!         call MPAS_io_close(inputFile, iErr)

!!         call mpas_log_write( ' in ocn_init_validate_buttermilk_bay &
!!                        after io_close', MPAS_LOG_CRIT ) 

!!       if ( config_Buttermilk_bay_topography_latlon_degrees .and. &
!!              config_Buttermilk_bay_topography_source == 'latlon_file' ) then
!!          topoLat % array(:) = topoLat % array(:) * pii / 180.0_RKIND
!!          topoLon % array(:) = topoLon % array(:) * pii / 180.0_RKIND
!!       end if

      end if
      !----------

      call mpas_log_write( ' Done ocn_init_validate_buttermilk_bay ' ) 
   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_ButterMilk_bay!}}}


!!***********************************************************************
!!
!!  routine ocn_init_validate_global_ocean
!!
!!> \brief   Validation for global ocean test case
!!> \author  Doug Jacobsen
!!> \date    03/04/2014
!!> \details
!!>  This routine validates the configuration options for the global ocean test case.
!!
!!-----------------------------------------------------------------------
!
!   subroutine ocn_init_validate_global_ocean(configPool, packagePool, iocontext, iErr)!{{{
!
!   !--------------------------------------------------------------------
!
!      type (mpas_pool_type), intent(inout) :: configPool, packagePool
!      type (mpas_io_context_type), intent(inout), target :: iocontext
!      integer, intent(out) :: iErr
!
!      type (mpas_io_context_type), pointer :: iocontext_ptr
!      type (MPAS_IO_Handle_type) :: inputFile
!      character (len=StrKIND), pointer :: config_init_configuration, &
!                                          config_global_ocean_topography_source, &
!                                          config_global_ocean_depth_file, &
!                                          config_global_ocean_depth_dimname, &
!                                          config_global_ocean_temperature_file, &
!                                          config_global_ocean_salinity_file, &
!                                          config_global_ocean_tracer_nlat_dimname, &
!                                          config_global_ocean_tracer_nlon_dimname, &
!                                          config_global_ocean_tracer_ndepth_dimname, &
!                                          config_global_ocean_topography_file, &
!                                          config_global_ocean_topography_nlat_dimname, &
!                                          config_global_ocean_topography_nlon_dimname, &
!                                          config_global_ocean_windstress_file, &
!                                          config_global_ocean_windstress_nlat_dimname, &
!                                          config_global_ocean_windstress_nlon_dimname, &
!                                          config_global_ocean_land_ice_topo_file, &
!                                          config_global_ocean_land_ice_topo_nlat_dimname, &
!                                          config_global_ocean_land_ice_topo_nlon_dimname, &
!                                          config_global_ocean_swData_file, &
!                                          config_global_ocean_swData_nlon_dimname, &
!                                          config_global_ocean_swData_nlat_dimname, &
!                                          config_global_ocean_ecosys_file, &
!                                          config_global_ocean_ecosys_nlat_dimname, &
!                                          config_global_ocean_ecosys_nlon_dimname, &
!                                          config_global_ocean_ecosys_ndepth_dimname
!
!      integer, pointer :: config_vert_levels, config_global_ocean_tracer_vert_levels, &
!                                          config_global_ocean_ecosys_vert_levels
!      logical, pointer :: config_use_ecosysTracers
!      logical, pointer :: landIceInitActive, config_global_ocean_depress_by_land_ice
!      logical, pointer :: criticalPassagesActive, config_global_ocean_deepen_critical_passages
!
!      iocontext_ptr => iocontext
!      iErr = 0
!
!      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)
!
!      if(config_init_configuration .ne. trim('global_ocean')) return
!
!      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
!
!      call mpas_pool_get_config(configPool, 'config_global_ocean_topography_source', &
!                                config_global_ocean_topography_source)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_depth_file', &
!                                config_global_ocean_depth_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_depth_dimname', &
!                                config_global_ocean_depth_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_temperature_file', &
!                                config_global_ocean_temperature_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_salinity_file', &
!                                config_global_ocean_salinity_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_tracer_nlat_dimname', &
!                                config_global_ocean_tracer_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_tracer_nlon_dimname', &
!                                config_global_ocean_tracer_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_tracer_ndepth_dimname', &
!                                config_global_ocean_tracer_ndepth_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_tracer_vert_levels', &
!                                config_global_ocean_tracer_vert_levels)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_topography_file', &
!                                config_global_ocean_topography_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_topography_nlat_dimname', &
!                                config_global_ocean_topography_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_topography_nlon_dimname', &
!                                config_global_ocean_topography_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_depress_by_land_ice', &
!                                config_global_ocean_depress_by_land_ice)
!      call mpas_pool_get_config(configPool, 'config_use_ecosysTracers', &
!                                config_use_ecosysTracers)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_land_ice_topo_file', &
!                                config_global_ocean_land_ice_topo_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_land_ice_topo_nlat_dimname', &
!                                config_global_ocean_land_ice_topo_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_land_ice_topo_nlon_dimname', &
!                                config_global_ocean_land_ice_topo_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_deepen_critical_passages', &
!                                config_global_ocean_deepen_critical_passages)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_windstress_file', &
!                                config_global_ocean_windstress_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_windstress_nlat_dimname', &
!                                config_global_ocean_windstress_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_windstress_nlon_dimname', &
!                                config_global_ocean_windstress_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_swData_file', &
!                                config_global_ocean_swData_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_swData_nlat_dimname', &
!                                config_global_ocean_swData_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_swData_nlon_dimname', &
!                                config_global_ocean_swData_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_ecosys_file', &
!                                config_global_ocean_ecosys_file)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_ecosys_nlat_dimname', &
!                                config_global_ocean_ecosys_nlat_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_ecosys_nlon_dimname', &
!                                config_global_ocean_ecosys_nlon_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_ecosys_ndepth_dimname', &
!                                config_global_ocean_ecosys_ndepth_dimname)
!      call mpas_pool_get_config(configPool, 'config_global_ocean_ecosys_vert_levels', &
!                                config_global_ocean_ecosys_vert_levels)
!
!      call mpas_pool_get_package(packagePool, 'landIceInitActive', landIceInitActive)
!      if ( config_global_ocean_depress_by_land_ice) then
!         landIceInitActive = .true.
!      end if
!
!      call mpas_pool_get_package(packagePool, 'criticalPassagesActive', criticalPassagesActive)
!      if ( config_global_ocean_deepen_critical_passages) then
!         criticalPassagesActive = .true.
!      end if
!
!      if (trim(config_global_ocean_depth_file) == 'none') then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!             // 'Invalid filename for config_global_ocean_depth_file', MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      inputFile = MPAS_io_open(config_global_ocean_depth_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!      if (iErr .ne. 0) then
!         call mpas_log_write( 'could not open file '// trim(config_global_ocean_depth_file), MPAS_LOG_CRIT)
!         return
!      end if
!
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_depth_dimname, nDepthOutput, iErr)
!
!      call MPAS_io_close(inputFile, iErr)
!
!      if (trim(config_global_ocean_temperature_file) == 'none') then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!             // 'Invalid filename for config_global_ocean_temperature_file', MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      if (trim(config_global_ocean_salinity_file) == 'none') then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!             // 'Invalid filename for config_global_ocean_salinity_file', MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      inputFile = MPAS_io_open(config_global_ocean_temperature_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!      if (iErr .ne. 0) then
!         call mpas_log_write( 'could not open file '// trim(config_global_ocean_temperature_file), MPAS_LOG_CRIT)
!         return
!      end if
!
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_tracer_nlat_dimname, nLatTracer, iErr)
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_tracer_nlon_dimname, nLonTracer, iErr)
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_tracer_ndepth_dimname, nDepthTracer, iErr)
!
!      call MPAS_io_close(inputFile, iErr)
!
!      if (config_global_ocean_tracer_vert_levels <= 0 .and. nDepthTracer > 0) then
!         config_global_ocean_tracer_vert_levels = nDepthTracer
!      else if(config_global_ocean_tracer_vert_levels <= 0) then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!              // 'Value of config_global_ocean_tracer_vert_levels=-1 ' &
!              // 'but nDepthTracer was not correctly read from input file.', MPAS_LOG_CRIT)
!         iErr = 1
!      end if
!
!      if (trim(config_global_ocean_windstress_file) == 'none') then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!             // 'Invalid filename for config_global_ocean_windstress_file', MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      inputFile = MPAS_io_open(config_global_ocean_swData_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_swData_nlat_dimname, nLatSW, iErr)
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_swData_nlon_dimname, nLonSW, iErr)
!
!      call MPAS_io_close(inputFile, iErr)
!
!      if (config_global_ocean_topography_source /= 'latlon_file' .and. &
!          config_global_ocean_topography_source /= 'mpas_variable') then
!         call mpas_log_write( 'Unexpected value for config_global_ocean_topography_source: ' &
!             // trim(config_global_ocean_topography_source), MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      if (config_global_ocean_topography_file == 'none' .and. &
!          config_global_ocean_topography_source == 'latlon_file') then
!         call mpas_log_write( 'Validation failed for global ocean. ' &
!             // 'Invalid filename for config_global_ocean_topography_file', MPAS_LOG_CRIT)
!         iErr = 1
!         return
!      end if
!
!      if (config_global_ocean_topography_source == 'latlon_file') then
!         inputFile = MPAS_io_open(config_global_ocean_topography_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!         if (iErr /= 0) then
!            call mpas_log_write( 'could not open file '// trim(config_global_ocean_topography_file), MPAS_LOG_CRIT)
!            return
!         end if
!
!         call MPAS_io_inq_dim(inputFile, config_global_ocean_topography_nlat_dimname, nLatTopo, iErr)
!         call MPAS_io_inq_dim(inputFile, config_global_ocean_topography_nlon_dimname, nLonTopo, iErr)
!
!         call MPAS_io_close(inputFile, iErr)
!      end if
!
!      inputFile = MPAS_io_open(config_global_ocean_windstress_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!      if (iErr .ne. 0) then
!         call mpas_log_write( 'could not open file '// trim(config_global_ocean_windstress_file), MPAS_LOG_CRIT)
!         return
!      end if
!
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_windstress_nlat_dimname, nLatWind, iErr)
!      call MPAS_io_inq_dim(inputFile, config_global_ocean_windstress_nlon_dimname, nLonWind, iErr)
!
!      call MPAS_io_close(inputFile, iErr)
!
!      if (config_vert_levels <= 0 .and. nDepthOutput > 0) then
!         config_vert_levels = nDepthOutput
!      else if(config_vert_levels <= 0) then
!         call mpas_log_write( 'Validation failed for global ocean. Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
!         iErr = 1
!      end if
!
!      if ( config_use_ecosysTracers ) then
!         if (trim(config_global_ocean_ecosys_file) == 'none') then
!            call mpas_log_write( &
!              'Validation failed for global ocean. Invalid filename for config_global_ocean_windstress_file', MPAS_LOG_CRIT)
!            iErr = 1
!            return
!         end if
!
!         inputFile = MPAS_io_open(config_global_ocean_ecosys_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!
!         call MPAS_io_inq_dim(inputFile, config_global_ocean_ecosys_nlat_dimname, nLatEcosys, iErr)
!         call MPAS_io_inq_dim(inputFile, config_global_ocean_ecosys_nlon_dimname, nLonEcosys, iErr)
!         call MPAS_io_inq_dim(inputFile, config_global_ocean_ecosys_ndepth_dimname, nDepthEcosys, iErr)
!
!         call MPAS_io_close(inputFile, iErr)
!
!         if (config_global_ocean_ecosys_vert_levels <= 0 .and. nDepthEcosys > 0) then
!            config_global_ocean_ecosys_vert_levels = nDepthEcosys
!         else if(config_global_ocean_ecosys_vert_levels <= 0) then
!            call mpas_log_write( 'Validation failed for global ocean. ' &
!                 // 'Value of config_global_ocean_ecosys_vert_levels=-1, ' &
!                 // 'but nDepthEcosys was not correctly read from input file.', MPAS_LOG_CRIT)
!            iErr = 1
!         end if
!
!      end if
!
!      if (config_global_ocean_depress_by_land_ice) then
!         if (config_global_ocean_land_ice_topo_file == 'none' .and. &
!             config_global_ocean_topography_source == 'latlon_file') then
!            call mpas_log_write( 'Validation failed for global ocean. '// &
!               'Invalid filename for config_global_ocean_land_ice_topo_file', MPAS_LOG_CRIT)
!            iErr = 1
!            return
!         end if
!
!         if(config_global_ocean_topography_source == 'latlon_file') then
!            inputFile = MPAS_io_open(config_global_ocean_land_ice_topo_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=iErr)
!            if (iErr /= 0) then
!               call mpas_log_write( 'could not open file '// trim(config_global_ocean_land_ice_topo_file), MPAS_LOG_CRIT)
!               return
!            end if
!
!            call MPAS_io_inq_dim(inputFile, config_global_ocean_land_ice_topo_nlat_dimname, nLatLandIceThk, iErr)
!            call MPAS_io_inq_dim(inputFile, config_global_ocean_land_ice_topo_nlon_dimname, nLonLandIceThk, iErr)
!
!            call MPAS_io_close(inputFile, iErr)
!         end if
!      end if
!
!   !--------------------------------------------------------------------
!
!   end subroutine ocn_init_validate_global_ocean!}}}
!



!!***********************************************************************
!!
!!  routine ocn_init_setup_global_ocean_read_topo
!!
!!> \brief   Read the topography IC file
!!> \author  Doug Jacobsen
!!> \date    03/04/2014
!!> \details
!!>  This routine reads the topography IC file, including latitude and longitude
!!>   information for topography data.
!!
!!-----------------------------------------------------------------------
!
!    subroutine ocn_init_setup_global_ocean_read_topo(domain, iErr)!{{{
!       type (domain_type), intent(inout) :: domain
!       integer, intent(out) :: iErr
!
!       type (MPAS_Stream_type) :: topographyStream
!
!       iErr = 0
!
!       ! Define stream for depth levels
!       call MPAS_createStream(topographyStream, domain % iocontext, config_global_ocean_topography_file, MPAS_IO_NETCDF, &
!                              MPAS_IO_READ, ierr=iErr)
!
!       ! Setup topoLat, topoLon, and topoIC fields for stream to be read in
!       topoLat % fieldName = trim(config_global_ocean_topography_lat_varname)
!       topoLat % dimSizes(1) = nLatTopo
!       topoLat % dimNames(1) = trim(config_global_ocean_topography_nlat_dimname)
!       topoLat % isVarArray = .false.
!       topoLat % isPersistent = .true.
!       topoLat % isActive = .true.
!       topoLat % hasTimeDimension = .false.
!       topoLat % block => domain % blocklist
!       allocate(topoLat % attLists(1))
!       allocate(topoLat % array(nLatTopo))
!
!       topoLon % fieldName = trim(config_global_ocean_topography_lon_varname)
!       topoLon % dimSizes(1) = nLonTopo
!       topoLon % dimNames(1) = trim(config_global_ocean_topography_nlon_dimname)
!       topoLon % isVarArray = .false.
!       topoLon % isPersistent = .true.
!       topoLon % isActive = .true.
!       topoLon % hasTimeDimension = .false.
!       topoLon % block => domain % blocklist
!       allocate(topoLon % attLists(1))
!       allocate(topoLon % array(nLonTopo))
!
!       topoIC % fieldName = trim(config_global_ocean_topography_varname)
!       topoIC % dimSizes(1) = nLonTopo
!       topoIC % dimSizes(2) = nLatTopo
!       topoIC % dimNames(1) = trim(config_global_ocean_topography_nlon_dimname)
!       topoIC % dimNames(2) = trim(config_global_ocean_topography_nlat_dimname)
!       topoIC % isVarArray = .false.
!       topoIC % isPersistent = .true.
!       topoIC % isActive = .true.
!       topoIC % hasTimeDimension = .false.
!       topoIC % block => domain % blocklist
!       allocate(topoIC % attLists(1))
!       allocate(topoIC % array(nLonTopo, nLatTopo))
!
!       ! Add topoLat, topoLon, and topoIC fields to stream
!       call MPAS_streamAddField(topographyStream, topoLat, iErr)
!       call MPAS_streamAddField(topographyStream, topoLon, iErr)
!       call MPAS_streamAddField(topographyStream, topoIC, iErr)
!
!       if(config_global_ocean_topography_has_ocean_frac) then
!          oceanFracIC % fieldName = trim(config_global_ocean_topography_ocean_frac_varname)
!          oceanFracIC % dimSizes(1) = nLonTopo
!          oceanFracIC % dimSizes(2) = nLatTopo
!          oceanFracIC % dimNames(1) = trim(config_global_ocean_topography_nlon_dimname)
!          oceanFracIC % dimNames(2) = trim(config_global_ocean_topography_nlat_dimname)
!          oceanFracIC % isVarArray = .false.
!          oceanFracIC % isPersistent = .true.
!          oceanFracIC % isActive = .true.
!          oceanFracIC % hasTimeDimension = .false.
!          oceanFracIC % block => domain % blocklist
!          allocate(oceanFracIC % attLists(1))
!          allocate(oceanFracIC % array(nLonTopo, nLatTopo))
!
!          call MPAS_streamAddField(topographyStream, oceanFracIC, iErr)
!       end if
!
!       ! Read stream
!       call MPAS_readStream(topographyStream, 1, iErr)
!
!       ! Close stream
!       call MPAS_closeStream(topographyStream)
!
!       if (config_global_ocean_topography_latlon_degrees) then
!          topoLat % array(:) = topoLat % array(:) * pii / 180.0_RKIND
!          topoLon % array(:) = topoLon % array(:) * pii / 180.0_RKIND
!       end if
!
!    end subroutine ocn_init_setup_global_ocean_read_topo!}}}



!***********************************************************************

end module ocn_init_Buttermilk_bay

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
