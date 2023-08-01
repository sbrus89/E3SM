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

module ocn_init_parabolic_bowl

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

   public :: ocn_init_setup_parabolic_bowl, &
             ocn_init_validate_parabolic_bowl

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   real (kind=RKIND):: LL
   real (kind=RKIND):: oneMC2, sqrtOneMC2, oneMC
   real (kind=RKIND),  parameter :: eps=1.0e-10
   real(kind=RKIND), dimension(:,:), pointer :: &
      subgridWetVolumeCellTable, &
      subgridWetVolumeEdgeTable, &
      subgridWetVolumeVertexTable, &
      subgridSshCellTableRange, &
      subgridSshEdgeTableRange, &
      subgridSshVertexTableRange
   real(kind=RKIND), dimension(:), pointer :: &
      subgridEdgeBathymetryMean, &
      subgridVertexBathymetryMean, &
      subgridCellBathymetryMin, &
      subgridEdgeBathymetryMin, &
      subgridVertexBathymetryMin, &
      subgridLayerThicknessDebug
   integer, pointer :: nSubgridTableLevels

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_parabolic_bowl
!
!> \brief   Setup for this initial condition
!> \author  D. Wirasaet
!> \date    May-June 2022
!> \details
!>  This routine sets up the initial conditions for this case.
!>   To be run in sigma vertical coordinates and single-layer
!
!-----------------------------------------------------------------------

  subroutine ocn_init_setup_parabolic_bowl(domain, iErr)!{{{
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

  
    real (kind=RKIND):: HH, uu, vv
    real (kind=RKIND):: CC, RR, num, den
    real (kind=RKIND):: xshift = 0.0, yshift = 0.0 
    real (kind=RKIND), dimension(:,:), allocatable :: rSubgridPoints, sSubgridPoints
    real (kind=RKIND), dimension(:), allocatable :: subgridBathymetryValues, subgridAreas
    real (kind=RKIND), dimension(:), allocatable :: subgridSshValues
    integer :: nSubgridCell, nSubgridEdge, nSubgridVertex
    integer :: nSubgridTriPerSlice
    integer :: v1, v2
    integer :: c1, c2
    real (kind=RKIND) :: x(3), y(3)
    integer :: slice, nSlice
    integer :: i,j
    real (kind=RKIND) :: deltaZ

    iErr = 0

    call ocn_subgrid_init(domain,iErr)

    if(config_init_configuration .ne. trim('parabolic_bowl')) return

    ! Determine vertical grid for configuration
    call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
    call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
    call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)

    ! print*, "nVertLevelsP1 = ", nVertLevelsP1 ; 
    ! print*, "nVertLevels = ", nVertLevels ; 
    ! print*, "config_parabolic_bowl_vert_levels", config_parabolic_bowl_vert_levels ; 
    nVertLevels  = config_parabolic_bowl_vert_levels ;
    nVertLevelsP1 = nVertLevels + 1

   
    !
    ! you may restrict your case geometry as follows:
    ! if ( on_a_sphere ) call mpas_log_write('The TEMPLATE configuration can only be applied ' &
    !     // 'to a planar mesh. Exiting...', MPAS_LOG_CRIT)

    allocate(interfaceLocations(nVertLevelsP1))
    ! call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations )
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

    xshift = 0.5_RKIND*(xMin + xMax) ;
    yshift = 0.5_RKIND*(YMin + yMax) ;  
    
    !--------------------------------------------------------------------
    ! Use this section to set initial values
    !--------------------------------------------------------------------

    !
    den = ((config_parabolic_bowl_omega**2.0) - (config_parabolic_bowl_Coriolis_parameter**2.0)) ; 
    num = (8.0_RKIND*config_parabolic_bowl_gravity*config_parabolic_bowl_b0) ;
    LL  = sqrt(num/den) ; 

    den = (config_parabolic_bowl_b0 + config_parabolic_bowl_eta0)**2.0 + config_parabolic_bowl_b0**2.0 ;
    num = (config_parabolic_bowl_b0 + config_parabolic_bowl_eta0)**2.0 - config_parabolic_bowl_b0**2.0 ; 

    CC = num/den ;

    oneMC2 = 1.0_RKIND - CC**2.0 ;
    sqrtOneMC2 = sqrt(oneMC2) ; 
    oneMC = 1.0_RKIND - CC ;


    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

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
       if ( config_parabolic_bowl_adjust_domain_center ) then
          xCell = xCell - xshift ; 
          yCell = yCell - yshift ;

          xEdge = xEdge - xshift ;
          yEdge = yEdge - yshift ;

          xVertex = xVertex - xshift ;
          yVertex = yVertex - yshift ; 
       end if


       ! Initlialze vector
       call mpas_initialize_vectors(meshPool) ; 



       ! Bathymetry    
       !do iCell = 1, nCellsSolve
       !   call ocn_init_parabolic_bowl_bathymetry(xCell(iCell),yCell(iCell),bottomDepth(iCell))
       !end do


       minLevelCell(:) = 1
       do iCell = 1, nCellsSolve
         ! Set up vertical grid
         maxLevelCell(iCell) = nVertLevels ; ! sigma coordinates 
        end do


       ! Set velocity
       !!
       do iEdge = 1, nEdgesSolve

         RR = sqrt(xEdge(iEdge)**2.0 + yEdge(iEdge)**2.0) ;
         
         HH = (sqrtOneMC2/oneMC) - ( (RR**2.0)/(LL**2.0))*( (oneMC2/(oneMC**2.0))  ) ;
         HH = config_parabolic_bowl_b0*HH  ;
    
          
         uu =  - config_parabolic_bowl_Coriolis_parameter*yEdge(iEdge)*( sqrtOneMC2 + CC - 1.0_RKIND  ) ;
         vv =    config_parabolic_bowl_Coriolis_parameter*xEdge(iEdge)*( sqrtOneMc2 + CC - 1.0_RKIND  ) ; 

         uu = 0.5_RKIND*uu/oneMC ; 
         vv = 0.5_RKIND*vv/oneMC ; 
         IF ( HH < eps ) THEN
             uu = 0.0_RKIND ;
             vv = 0.0_RKIND ;
         end if

                                  
         do k = 1, nVertLevels 
            normalVelocity(k,iEdge) = uu*edgeNormalVectors(1,iEdge) + vv*edgeNormalVectors(2,iEdge) ;
         end do
         ! print*, maxval( abs(normalVelocity(:,iedge)) ), edgeNormalVectors(1,iEdge), edgeNormalVectors(2,iEdge), HH ;
       end do
       !!

       do iCell = 1, nCellsSolve
         
          ! Set temperature
          activeTracers(index_temperature, :, iCell) = 10.0_RKIND

          ! Set salinity
          activeTracers(index_salinity, :, iCell) = 30.0_RKIND

         ! Set Coriolis parameters, if other than zero
         fCell(iCell) = config_parabolic_bowl_coriolis_parameter ; 
      end do

      do iEdge = 1, nEdgesSolve
         fEdge(iEdge) = config_parabolic_bowl_coriolis_parameter ;  
      end do

      do iVertex = 1, nVerticesSolve
         fVertex(iVertex) =  config_parabolic_bowl_coriolis_parameter ;
      end do


      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Cells
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      print*, "Begin cells", nCellsSolve

      nSubgridTriPerSlice = config_parabolic_bowl_subgrid_level**2
      allocate(rSubgridPoints(3,maxEdges*nSubgridTriPerSlice), sSubgridPoints(3,maxEdges*nSubgridTriPerSlice))
      call ocn_init_define_subgrid_points(config_parabolic_bowl_subgrid_level, rSubgridPoints, sSubgridPoints)
      allocate(subgridBathymetryValues(maxEdges*nSubgridTriPerSlice), subgridAreas(maxEdges*nSubgridTriPerSlice))
      allocate(subgridSshValues(maxEdges*nSubgridTriPerSlice))

      do iCell = 1,nCellsSolve
        !print*, "iCell: ", iCell

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

          call ocn_init_evaluate_subgrid_data(x, y, nSubgridTriPerSlice, nSubgridCell, rSubgridPoints, sSubgridPoints, subgridBathymetryValues, subgridAreas, subgridSshValues)

        enddo
        

        ! Evaluate bounds of look-up table range
        !---------------------------------------------------------------

        subgridSshCellTableRange(1,iCell) = -maxval(subgridBathymetryValues(1:nSubgridCell)) + config_drying_min_cell_height + eps
        subgridSshCellTableRange(2,iCell) = -minval(subgridBathymetryValues(1:nSubgridCell))

        ! Evaluate subgrid bathymetry
        !---------------------------------------------------------------
        bottomDepth(iCell) = sum(subgridBathymetryValues(1:nSubgridCell))/real(nSubgridCell,RKIND)
        subgridCellBathymetryMin(iCell) = minval(subgridBathymetryValues(1:nSubgridCell))
        !subgridCellBathymetryMin(iCell) = bottomDepth(iCell)

        ! Vertical integration of wet fraction
        !---------------------------------------------------------------

        call ocn_init_vertical_integration(iCell,subgridSshCellTableRange, nSubgridCell, subgridBathymetryValues, subgridAreas, subgridWetVolumeCellTable)
        !print*, iCell, areaCell(iCell)-sum(subgridAreas(1:nSubgridCell))

        ! Evaluate wet layerThickness average
        !---------------------------------------------------------------
        call ocn_init_wet_average(nSubgridCell, subgridBathymetryValues, subgridSshValues, subgridAreas, layerThickness(1,iCell))

        if (iCell == 3030) then
          print*, subgridSshCellTableRange(1,iCell)
          print*, subgridSshCellTableRange(2,iCell)
          print*, nSubgridTableLevels
          deltaZ = (subgridSshCellTableRange(2,iCell) - subgridSshCellTableRange(1,iCell))/(nSubgridTableLevels-1)
          print*, deltaZ
          do i = 1,nSubgridTableLevels
            print*, (real(i,RKIND)-1.0_RKIND)*deltaZ + subgridSshCellTableRange(1,iCell), subgridWetVolumeCellTable(i,iCell)
          enddo
        endif

      enddo



      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Edges
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      print*, "Begin edges", nEdgesSolve

      do iEdge = 1,nEdgesSolve
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

        !print*, cellsOnEdge(1,iEdge), cellsOnEdge(2,iEdge), nSlice

        nSubgridEdge = 0  ! Counter for all subgrid triangles over edge slices
        do slice = 1,nSlice ! Loop over edge slices

          ! Edge slice coordinates
          x(1) = xVertex(verticesOnEdge(1,iEdge))
          y(1) = yVertex(verticesOnEdge(1,iEdge))

          x(2) = xVertex(verticesOnEdge(2,iEdge))
          y(2) = yVertex(verticesOnEdge(2,iEdge))

          x(3) = xCell(cellsOnEdge(slice,iEdge))
          y(3) = yCell(cellsOnEdge(slice,iEdge))

          call ocn_init_evaluate_subgrid_data(x, y, nSubgridTriPerSlice, nSubgridEdge, rSubgridPoints, sSubgridPoints, subgridBathymetryValues, subgridAreas)

        enddo

        ! Evaluate bounds of look-up table range
        !---------------------------------------------------------------

        subgridSshEdgeTableRange(1,iEdge) = -maxval(subgridBathymetryValues(1:nSubgridEdge)) + config_drying_min_cell_height + eps
        subgridSshEdgeTableRange(2,iEdge) = -minval(subgridBathymetryValues(1:nSubgridEdge))

        ! Evaluate subgrid bathymetry
        !---------------------------------------------------------------
        subgridEdgeBathymetryMean(iEdge) = sum(subgridBathymetryValues(1:nSubgridEdge))/real(nSubgridEdge,RKIND) 
        subgridEdgeBathymetryMin(iEdge)  = minval(subgridBathymetryValues(1:nSubgridEdge))

        ! Vertical integration of wet fraction
        !---------------------------------------------------------------

        call ocn_init_vertical_integration(iEdge,subgridSshEdgeTableRange, nSubgridEdge, subgridBathymetryValues, subgridAreas, subgridWetVolumeEdgeTable)

      enddo



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
        !print*, cellsOnVertex(1,iVertex), cellsOnVertex(2,iVertex), cellsOnVertex(3,iVertex), nSlice

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

          call ocn_init_evaluate_subgrid_data(x, y, nSubgridTriPerSlice, nSubgridVertex, rSubgridPoints, sSubgridPoints, subgridBathymetryValues, subgridAreas)
        enddo

        ! Evaluate bounds of look-up table range
        !---------------------------------------------------------------

        subgridSshVertexTableRange(1,iVertex) = -maxval(subgridBathymetryValues(1:nSubgridVertex)) + config_drying_min_cell_height + eps
        subgridSshVertexTableRange(2,iVertex) = -minval(subgridBathymetryValues(1:nSubgridVertex))

        ! Evaluate subgrid bathymetry
        !---------------------------------------------------------------
        subgridVertexBathymetryMean(iVertex) = sum(subgridBathymetryValues(1:nSubgridVertex))/real(nSubgridVertex,RKIND)
        subgridVertexBathymetryMin(iVertex)  = minval(subgridBathymetryValues(1:nSubgridVertex))

        ! Vertical integration of wet fraction
        !---------------------------------------------------------------

        call ocn_init_vertical_integration(iVertex,subgridSshVertexTableRange, nSubgridVertex, subgridBathymetryValues, subgridAreas, subgridWetVolumeVertexTable)

      enddo vertex

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

       if (config_use_wetting_drying) then

         do iCell = 1, nCellsSolve
           ! Set up vertical grid
           maxLevelCell(iCell) = nVertLevels ; ! sigma coordinates 
          end do

          do iCell = 1, nCellsSolve
            !
            ! make sure depth is thick enough via ssh = TOTAL_DEPTH - bottomDepth
            ! add a thin layer of nlayer*config_drying_min_cellhight 
            ! 
 
            if (config_use_subgrid_wetting_drying) then
              call ocn_subgrid_ssh_lookup(layerThickness(1,iCell),&
                                          subgridWetVolumeCellTable(:,iCell),&
                                          subgridSshCellTableRange(:,iCell),&
                                          bottomDepth(iCell),&
                                          subgridCellBathymetryMin(iCell),&
                                          ssh(iCell))

              !call ocn_init_parabolic_bowl_ssh(xCell(iCell),yCell(iCell),bottomDepth(iCell),ssh(iCell))
              !ssh(iCell) = - bottomDepth(iCell) + &
              !  max(ssh(iCell) + bottomDepth(iCell), &
              !  maxLevelCell(iCell)*(config_drying_min_cell_height + eps))
              !call ocn_subgrid_layer_thickness_lookup(ssh(iCell), & 
              !                                        subgridWetVolumeCellTable(:,iCell), &
              !                                        subgridSshCellTableRange(:,iCell),&
              !                                        bottomDepth(iCell),&
              !                                        layerThickness(1,iCell))
            else
              call ocn_init_parabolic_bowl_ssh(xCell(iCell),yCell(iCell),bottomDepth(iCell),ssh(iCell))
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
          end do
          
       end if 

       block_ptr => block_ptr % next
    end do

    do iCell = 1,nCellsSolve
      call ocn_subgrid_layer_thickness_lookup(ssh(iCell), & 
                                              subgridWetVolumeCellTable(:,iCell), &
                                              subgridSshCellTableRange(:,iCell),&
                                              bottomDepth(iCell),&
                                              subgridLayerThicknessDebug(iCell))
    enddo
      !do iCell = 1,nCellsSolve
      !  call ocn_subgrid_ssh_lookup(layerThickness(1,iCell),&
      !                              subgridWetVolumeCellTable(:,iCell),&
      !                              subgridSshCellTableRange(:,iCell),&
      !                              bottomDepth(iCell),&
      !                              subgridCellBathymetryMin(iCell),&
      !                              ssh(iCell))
      !enddo

    deallocate(interfaceLocations)
    !--------------------------------------------------------------------

    print*, "****** End parabolic bowl init *****" ;

    return ;  
  end subroutine ocn_init_setup_parabolic_bowl!}}}

!***********************************************************************
!
!  routine ocn_init_define_subgrid_points
!
!> \brief  
!> \author  Steven Brus 
!> \date    November 2022 
!> \details
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

          !print("(i5,1x,3(e10.3,1x))"), nSubgridTri, (rSubgridPoints(k,nSubgridTri), k=1,3)
          !print("(i5,1x,3(e10.3,1x))"), nSubgridTri, (sSubgridPoints(k,nSubgridTri), k=1,3)

          ! upper triangle in pair (see u in triangle above, doesn't occur next to hypotenuse)          
          if (i < nSubgridLevel+1 - j) then
            
            nSubgridTri = nSubgridTri + 1

            rSubgridPoints(1,nSubgridTri) = xPoints(i+1)
            sSubgridPoints(1,nSubgridTri) = xPoints(j)
            
            rSubgridPoints(2,nSubgridTri) = xPoints(i+1)
            sSubgridPoints(2,nSubgridTri) = xPoints(j+1)

            rSubgridPoints(3,nSubgridTri) = xPoints(i)
            sSubgridPoints(3,nSubgridTri) = xPoints(j+1)

            !print("(i5,1x,3(e10.3,1x))"), nSubgridTri, (rSubgridPoints(k,nSubgridTri), k=1,3)
            !print("(i5,1x,3(e10.3,1x))"), nSubgridTri, (sSubgridPoints(k,nSubgridTri), k=1,3)
            
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
!> \brief  
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_evaluate_subgrid_data(xTri, yTri, nSubgridTri, nSubgridCV, rSubgridPoints, sSubgridPoints, subgridBathymetryValues, subgridAreas, subgridSshValues)!{{{
           
      implicit none 
      real (kind=RKIND), intent(inout) :: xTri(3), yTri(3)
      integer, intent(in) :: nSubgridTri
      integer, intent(inout) :: nSubgridCV
      real (kind=RKIND), dimension(:,:), intent(in) :: rSubgridPoints, sSubgridPoints
      real (kind=RKIND), dimension(:), intent(inout) :: subgridBathymetryValues
      real (kind=RKIND), dimension(:), intent(inout) :: subgridAreas
      real (kind=RKIND), dimension(:), intent(inout),optional :: subgridSshValues
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
       x = xTri(2)
       y = yTri(2)
       xTri(2) = xTri(3)
       yTri(2) = yTri(3)
       xTri(3) = x
       yTri(3) = y
       !print*, area
       area = abs(area)
     endif

     !print("(3(f10.3,1x))"), (xTri(i)/1000, i=1,3)
     !print("(3(f10.3,1x))"), (yTri(i)/1000, i=1,3)

      do iPt = 1,nSubgridTri

        ! Counter over all subcells within cell/edge/vertex control volume
        nSubgridCV = nSubgridCV + 1

        ! Center sub-triangle (on reference triangle)
        rCenter = sum(rSubgridPoints(:,iPt))/3.0_RKIND
        sCenter = sum(sSubgridPoints(:,iPt))/3.0_RKIND

        ! Transformation of sub-triangle center to physical coordinates
        call ocn_init_tri_coordinate_transform(rCenter, sCenter, xTri, yTri, xCenter, yCenter)

        ! Evaluate bathymetry
        call ocn_init_parabolic_bowl_bathymetry(xCenter, yCenter, subgridBathymetryValues(nSubgridCV))

        ! Transformation of sub-triangle vertices to physical coordinates
        do i = 1,3
           call ocn_init_tri_coordinate_transform(rSubgridPoints(i,iPt), sSubgridPoints(i,iPt), xTri, yTri, xSubgridPoints(i), ySubgridPoints(i))
        enddo
 
        ! Calculate area of sub-triangle
        call ocn_init_tri_area(xSubgridPoints(:), ySubgridPoints(:), subgridAreas(nSubgridCV))

        ! Optinally evalulate ssh 
        if (present(subgridsshValues)) then
           call ocn_init_parabolic_bowl_ssh(xCenter, yCenter, subgridBathymetryValues(nSubgridCV), subgridSshValues(nSubgridCV))
        endif

        !print("(I5,I5,1x,5(f10.3,1x))"), nSubgridCV, iPt, x/1000, y/1000, subgridBathymetryValues(nSubgridCV), subgridAreas(nSubgridCV)

      enddo

   !--------------------------------------------------------------------

   end subroutine ocn_init_evaluate_subgrid_data!}}}

!***********************************************************************
!
!  routine ocn_init_parabolic_bowl_bathymetry
!
!> \brief   
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_parabolic_bowl_bathymetry(x, y, depth)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x, y
      real (kind=RKIND), intent(out) :: depth
      real (kind=RKIND) :: RR

   !--------------------------------------------------------------------

      RR = sqrt(x**2 + y**2) 
      depth  = config_parabolic_bowl_b0*(1.0_RKIND - (RR**2)/(LL**2))

   !--------------------------------------------------------------------

   end subroutine ocn_init_parabolic_bowl_bathymetry!}}}

!***********************************************************************
!
!  routine ocn_init_parabolic_bowl_ssh
!
!> \brief   
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_parabolic_bowl_ssh(x, y, bottomDepth, ssh)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x, y
      real (kind=RKIND), intent(in) :: bottomDepth
      real (kind=RKIND), intent(out) :: ssh
      real (kind=RKIND) :: RR

   !--------------------------------------------------------------------

      RR = sqrt(x**2 + y**2) ;
 
      ssh = (sqrtOneMC2/oneMC) - 1.0_RKIND - ((RR**2)/(LL**2))*( (oneMC2/(oneMC**2)) - 1.0_RKIND ) ;
      ssh = config_parabolic_bowl_b0*ssh ; 
      !ssh = - bottomDepth + max(ssh + bottomDepth, config_drying_min_cell_height + eps)

   !--------------------------------------------------------------------

   end subroutine ocn_init_parabolic_bowl_ssh!}}}

!***********************************************************************
!
!  routine ocn_init_tri_area
!
!> \brief   
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_tri_area(x, y, area)!{{{
           
      implicit none 

      real (kind=RKIND), intent(in) :: x(3), y(3)
      real (kind=RKIND), intent(out) :: area

   !--------------------------------------------------------------------

      !area = 0.5_RKIND*((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)))
      area = 0.5_RKIND*(x(2)*y(3) - y(2)*x(3) - x(1)*y(3) + y(1)*x(3) + x(1)*y(2) - y(1)*x(2))
      !area = abs(area)

   !--------------------------------------------------------------------

   end subroutine ocn_init_tri_area!}}}

!***********************************************************************
!
!  routine ocn_init_tri_coordinate_transform
!
!> \brief   
!> \author  
!> \date    
!> \details
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
!> \brief   
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_vertical_integration(iCV, subgridSshTableRange, nSubgridCV, subgridBathymetryValues, subgridAreas, subgridWetVolumeTable)!{{{
           
      implicit none 

      integer, intent(in) :: iCV
      real (kind=RKIND), dimension(:,:), intent(in) :: subgridSshTableRange
      integer, intent(in) :: nSubgridCV
      real (kind=RKIND), dimension(:), intent(in) :: subgridBathymetryValues, subgridAreas
      real (kind=RKIND), dimension(:,:), intent(inout) :: subgridWetVolumeTable

      real (kind=RKIND) :: deltaZ, ssh, pVal
      integer :: lev, tri
      

   !--------------------------------------------------------------------
   
      deltaZ = (subgridSshTableRange(2,iCV)-subgridSshTableRange(1,iCV))/real(nSubgridTableLevels-1,RKIND);

      subgridWetVolumeTable(1,iCV) = config_drying_min_cell_height + eps 
      ssh = subgridSshTableRange(1,iCV) + deltaZ
      subgridWetVolumeTable(2,iCV) = subgridWetVolumeTable(1,iCV)*sum(subgridAreas(1:nSubgridCV))
      do lev = 2,nSubgridTableLevels
        do tri = 1,nSubgridCV

          pVal = 0.0_RKIND
          if (subgridBathymetryValues(tri) + ssh >= 0.0_RKIND) then
            pVal = 1.0_RKIND
          endif

          subgridWetVolumeTable(lev,iCV) = subgridWetVolumeTable(lev,iCV) + pVal*deltaZ*subgridAreas(tri)
  
        enddo

        if (lev < nSubgridTableLevels) then
          subgridWetVolumeTable(lev+1,iCV) = subgridWetVolumeTable(lev,iCV)
        endif

        subgridWetVolumeTable(lev,iCV) = subgridWetVolumeTable(lev,iCV)/sum(subgridAreas(1:nSubgridCV))

        ssh = ssh + deltaZ
      enddo

   !--------------------------------------------------------------------

   end subroutine ocn_init_vertical_integration!}}}

!***********************************************************************
!
!  routine ocn_init_wet_average
!
!> \brief   
!> \author  
!> \date    
!> \details
!>  
!
!-----------------------------------------------------------------------

   subroutine ocn_init_wet_average(nSubgridCV, subgridBathymetryValues, subgridSshValues, subgridAreas, subgridWetAverage)!{{{
           
      implicit none 

      integer, intent(in) :: nSubgridCV
      real (kind=RKIND), dimension(:), intent(in) :: subgridBathymetryValues, subgridAreas
      real (kind=RKIND), dimension(:), intent(in) :: subgridSshValues
      real (kind=RKIND), intent(inout) :: subgridWetAverage

      real (kind=RKIND) :: deltaZ, ssh, pVal
      real (kind=RKIND) :: averageDepth, wetArea, layerThicknessValue
      integer :: lev, tri
      

   !--------------------------------------------------------------------
   
      averageDepth = sum(subgridBathymetryValues(1:nSubgridCV))/real(nSubgridCV,RKIND) 

      subgridWetAverage = 0.0_RKIND
      wetArea = 0.0_RKIND
      do tri = 1,nSubgridCV

        layerThicknessValue = subgridBathymetryValues(tri) + subgridSshValues(tri)
        if (layerThicknessValue < config_drying_min_cell_height + eps) then
          layerThicknessValue = config_drying_min_cell_height + eps
        endif
        subgridWetAverage = subgridWetAverage + layerThicknessValue*subgridAreas(tri)
        wetArea = wetArea + subgridAreas(tri)
      enddo

      subgridWetAverage = subgridWetAverage/wetArea
      

   !--------------------------------------------------------------------

   end subroutine ocn_init_wet_average!}}}

!***********************************************************************
!
!  routine ocn_init_validate_parabolic_bowl
!
!> \brief   Validation for this initial condition
!> \author  
!> \date   
!> \details
!>  This routine validates the configuration options for this case.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_parabolic_bowl(configPool, packagePool, iocontext, iErr)!{{{
           
      implicit none 

   !--------------------------------------------------------------------
      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_parabolic_bowl_vert_levels
   
      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)

      if(config_init_configuration .ne. trim('parabolic_bowl')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool, 'config_parabolic_bowl_vert_levels', config_parabolic_bowl_vert_levels)

      if(config_vert_levels <= 0 .and. config_parabolic_bowl_vert_levels > 0) then
         config_vert_levels = config_parabolic_bowl_vert_levels
      else if (config_vert_levels <= 0) then
         call mpas_log_write( 'Validation failed for para_bowl. Not given a usable value for vertical levels.', MPAS_LOG_CRIT)
         iErr = 1
      end if

      print*, "in ocn_init_validate_parabolic_bowl()" ; 
   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_parabolic_bowl!}}}


!***********************************************************************

end module ocn_init_parabolic_bowl

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
