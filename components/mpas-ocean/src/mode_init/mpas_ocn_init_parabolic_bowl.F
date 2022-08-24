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

    ! Define variable pointers
    logical, pointer :: on_a_sphere
    integer, dimension(:), pointer :: minLevelCell, maxLevelCell
    real (kind=RKIND), dimension(:), pointer :: xCell, yCell, refBottomDepth, refZMid, &
         vertCoordMovementWeights, bottomDepth, fCell, fEdge, fVertex, dcEdge
    real (kind=RKIND), dimension(:,:), pointer:: zMid

    real (kind=RKIND), dimension(:), pointer:: xEdge, yEdge, xVertex, yVertex
    ! DW
    real (kind=RKIND) :: minBottomDepth, maxBottomDepth, globalMaxBottomDepth, globalMinBottomDepth
    ! DW
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers
    
    ! DW:
    !   Define variable pointers
    real (kind=RKIND), dimension(:), pointer :: ssh
    real (kind=RKIND), dimension(:,:), pointer :: edgeNormalVectors
    real (kind=RKIND), dimension(:,:), pointer :: normalVelocity
  
    ! DW:
    real (kind=RKIND):: HH, uu, vv
    real (kind=RKIND):: LL, CC, RR, num, den
    real (kind=RKIND):: oneMC2, sqrtOneMC2, oneMC
    real (kind=RKIND),  parameter :: eps=1.0e-10
    real (kind=RKIND):: xshift = 0.0, yshift = 0.0 

    iErr = 0

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
       call mpas_pool_get_dimension( meshPool, 'nVerticesSolve', nVerticesSolve ) ;

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

    xshift = 0.5*(xMin + xMax) ;
    yshift = 0.5*(YMin + yMax) ;  
    
    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)

       call ocn_mark_north_boundary(meshPool, yMaxGlobal, dcEdgeMinGlobal, iErr)
       call ocn_mark_south_boundary(meshPool, yMinGlobal, dcEdgeMinGlobal, iErr)

       block_ptr => block_ptr % next
    end do

    !--------------------------------------------------------------------
    ! Use this section to set initial values
    !--------------------------------------------------------------------

    !
    den = ((config_parabolic_bowl_omega**2.0) - (config_parabolic_bowl_Coriolis_parameter**2.0)) ; 
    num = (8.0*config_parabolic_bowl_gravity*config_parabolic_bowl_b0) ;
    LL  = sqrt(num/den) ; 

    den = (config_parabolic_bowl_b0 + config_parabolic_bowl_eta0)**2.0 + config_parabolic_bowl_b0**2.0 ;
    num = (config_parabolic_bowl_b0 + config_parabolic_bowl_eta0)**2.0 - config_parabolic_bowl_b0**2.0 ; 

    CC = num/den ;

    oneMC2 = 1.0 - CC**2.0 ;
    sqrtOneMC2 = sqrt(oneMC2) ; 
    oneMC = 1.0 - CC ;
    !

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
      
       call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)

       call mpas_pool_get_array(meshPool, 'xCell', xCell)
       call mpas_pool_get_array(meshPool, 'yCell', yCell)
       call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
       call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'minLevelCell', minLevelCell)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
       call mpas_pool_get_array(meshPool, 'fCell', fCell)
       call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
       call mpas_pool_get_array(meshPool, 'fVertex', fVertex)
       
       ! DW
       call mpas_pool_get_array(meshPool, 'xEdge', xEdge ) 
       call mpas_pool_get_array(meshPool, 'yEdge', yEdge ) 
       call mpas_pool_get_array(meshPool, 'xVertex', xVertex ) 
       call mpas_pool_get_array(meshPool, 'yVertex', yVertex ) 

       ! DW
       call mpas_pool_get_array(statePool, 'zMid', zMid, 1) ;

       ! DW, for initial conditions
       call mpas_pool_get_array(statePool, 'ssh', ssh, 1)
       call mpas_pool_get_array(meshPool, 'edgeNormalVectors', edgeNormalVectors ) ;
       call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity ) ;

       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)

       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)


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



       ! print*,  "associated = ", associated( refBottomDepth ) ; 
       ! print*, "dim refBottomDepth = ", ubound( refBottomDepth ) ; 
       ! print*, "dim refZMid = ", ubound( refZMid ) ; 
      
       ! Initlialze vector
       call mpas_initialize_vectors(meshPool) ; 


       ! print*, "nVertLevels = ", nVertLevels ; 
       ! print*, "nCellsSolve = ", nCellsSolve ;
       ! print*, "nEdgesSolve = ", nEdgesSolve ;
       ! print*, "nVerticesSolve = ", nVerticesSolve ; 

       ! DW:
       ! Bathymetry    
       do iCell = 1, nCellsSolve

          ! bottomDepth(iCell) = -1.0_RKIND*bottomDepthObserved(iCell)
          RR = sqrt(xCell(iCell)**2.0 + yCell(icell)**2.0) ;
          bottomDepth(iCell) = config_parabolic_bowl_b0*(1.0 - (RR**2.0)/(LL**2.0)) ;         

          ! sea level rise adjustment
          ! bottomDepth(iCell) = bottomDepth(iCell) + config_hurricane_sea_level_rise_adjustment
          !
          !! Enforce minimum and maximum bottom depth
          ! bottomDepth(iCell) = max(bottomDepth(iCell), config_hurricane_min_depth)
          ! bottomDepth(iCell) = min(bottomDepth(iCell), globalMaxBottomDepth)
       end do

       ! Find max bottom depth
       ! 
       maxBottomDepth = maxval( bottomDepth ) ; 
       minBottomDepth = maxval( bottomDepth ) ; 
       call mpas_dmpar_max_real( domain % dminfo, maxBottomDepth, globalMaxBottomDepth ) ;
       call mpas_dmpar_min_real( domain % dminfo, minBottomDepth, globalMinBottomDepth ) ;
       ! DW
       !

       !
       ! Set refBottomDepth, bottomDepth, and maxLevelCell
       ! if (min(config_tidal_boundary_left_bottom_depth, config_tidal_boundary_right_bottom_depth) < 0.0_RKIND) then
       !   ! consider the case where there is wetting / drying and vertical mesh resolution is needed "on land"
       !   do k = 1, nVertLevels
       !    refBottomDepth(k) = config_tidal_boundary_left_bottom_depth + &
       !      (config_tidal_boundary_right_bottom_depth - config_tidal_boundary_left_bottom_depth)* interfaceLocations(k+1)
       !   end do
       ! else
       !  ! assumes we just need to build vertical mesh to deepest point (e.g., no "on land' vertical mesh)
       !  do k = 1, nVertLevels
       !    refBottomDepth(k) = &
       !    max(config_tidal_boundary_left_bottom_depth, config_tidal_boundary_right_bottom_depth) * interfaceLocations(k+1)
       !  end do
       ! end if
       !

       ! print*, ubound( refBottomDepth ) ;
       ! print*, ubound( refZMid ) ;

       ! Set refBottomDepth and refZMid
       do k = 1, nVertLevels
          ! refBottomDepth(k) = config_para_bowl_bottom_depth * interfaceLocations(k+1)
          ! refZMid(k) = - 0.5_RKIND * (interfaceLocations(k+1) + interfaceLocations(k)) * config_para_bowl_bottom_depth

          ! refBottomDepth(k) = globalMaxBottomDepth * interfaceLocations(k+1)
          ! refZMid(k) = - 0.5_RKIND * (interfaceLocations(k+1) + interfaceLocations(k)) * globalMaxBottomDepth
 !         refBottomDepth(k) = -2.0*config_parabolic_bowl_eta0 + &
 !    &         (globalMaxBottomDepth + 2.0*config_parabolic_bowl_eta0 )*interfaceLocations(k+1) ;


!! DW
!!          refBottomDepth(k) =  globalMinBottomDepth + &
!!     &         (globalMaxBottomDepth - globalMinBottomDepth )*interfaceLocations(k+1) ;
!!
!!          refZMid(k) =  globalMinBottomDepth + 0.5*( interfraceLocations(k+1) + interFaceLocations(k) )* &
!!                   (globalMaxBottomDepth - globalMinBottomDepth) ;
!!
!!          refZMid(k) = -refZMid(k)
!! DW
         refBottomDepth(k) = globalMaxBottomDepth*interfaceLocations(k+1) ; 
         refZMid(k) = -0.5*( interfaceLocations(k+1) + interfaceLocations(k))*globalMaxBottomDepth ; 
       end do

       ! Set vertCoordMovementWeights
       vertCoordMovementWeights(:) = 1.0_RKIND

       ! Set ssh
       do iCell = 1, nCellsSolve
         !  dlat = latCell(iCell) - config_hurricane_gaussian_lat_center*pii/180.0_RKIND
         !  dlon = abs(lonCell(iCell) - config_hurricane_gaussian_lon_center*pii/180.0_RKIND)
         !  dw = config_hurricane_gaussian_width*pii/180.0_RKIND
         !  if (dlon > pii) dlon = dlon - 2.0_RKIND**pii
         !  ssh(iCell) = config_hurricane_gaussian_slr_amp + &
         !  config_hurricane_gaussian_hump_amplitude * exp(-(dlat**2.0_RKIND + dlon**2.0_RKIND) / &
         !  (2.0_RKIND*dw**2.0_RKIND))

         RR = sqrt(xCell(iCell)**2.0 + yCell(icell)**2.0) ;
 
         ssh(iCell) = (sqrtOneMC2/oneMC) - 1.0 - ((RR**2.0)/(LL**2.0))*( (oneMC2/(oneMC**2.0)) - 1.0 ) ;
         ssh(iCell) = config_parabolic_bowl_b0*ssh(iCell) ; 
        
         HH = ssh(iCell) + bottomDepth(iCell) ; 
    
         IF ( HH < eps ) ssh(iCell) = -bottomDepth(iCell) - eps ; 
         !
       end do

       !
       minLevelCell(:) = 1

       if (config_use_wetting_drying) then
          !
         do iCell = 1, nCellsSolve
           ! Set up vertical grid
           maxLevelCell(iCell) = nVertLevels ; ! sigma coordinates 
 
           ! ensures only 3 layers can be obtained on mesh
 !           if ( bottomDepth(iCell) < 0 .or. bottomDepth(iCell) < refBottomDepth(3)) then
 !             maxLevelCell(iCell) = 3
 !           else
 !             ! snap to zstar coordinate
 !             do k = 1, nVertLevels
 !               if ( bottomDepth(iCell) < refBottomDepth(k) ) then
 !                 maxLevelCell(iCell) = k-1
 !                 bottomDepth(iCell) = refBottomDepth(k-1)
 !                 exit
 !               end if
 !             end do
 !             !
 !           end if
          end do
          !

          do iCell = 1, nCellsSolve
            !
            ! make sure depth is thick enough via ssh = TOTAL_DEPTH - bottomDepth
            ! add a thin layer of nlayer*config_drying_min_cellhight 
            ! 
            ssh(iCell) = - bottomDepth(iCell) + &
              max(ssh(iCell) + bottomDepth(iCell), &
              maxLevelCell(iCell)*(config_drying_min_cell_height + eps))
          
            do k = 1, maxLevelCell(iCell)
              layerThickness(k,iCell) = max(config_drying_min_cell_height + eps, &
                (ssh(iCell) + bottomDepth(iCell))/maxLevelCell(iCell))
             
              if (layerThickness(k,iCell) < config_drying_min_cell_height) then
                call mpas_log_write('layerThickness($i,$i)=$r', MPAS_LOG_CRIT, &
                  intArgs=(/k,iCell/), &
                  realArgs=(/layerThickness(k,iCell)/))
              end if
            end do
          
            do k = 1, maxLevelCell(iCell)
              restingThickness(k,iCell) = bottomDepth(iCell)/maxLevelCell(iCell)
            end do
          end do
          
          ! 
          ! Taken from Tidal boundary   
          !do iCell = 1, nCellsSolve
          !   ! restingThickness is z-level, with ssh = 0
          !   call ocn_compute_z_level_layerThickness(restingThickness(:,iCell), refBottomDepth, 0.0_RKIND, &
          !                                     bottomDepth(iCell), minLevelCell(iCell),              &
          !                                     maxLevelCell(iCell), nVertLevels, iErr)
          !
          !   ! stretch restingThickness to get the z-star layerThickness
          !   call ocn_compute_z_star_layerThickness(layerThickness(:,iCell), restingThickness(:,iCell),  &
          !                                       ssh(iCell), bottomDepth(iCell), minLevelCell(iCell), &
          !                                       maxLevelCell(iCell), nVertLevels, iErr)
          !
          !   ! compute zMid
          !   call ocn_compute_zMid_from_layerThickness(zMid(:,iCell), layerThickness(:,iCell), ssh(iCell), &
          !                                          minLevelCell(iCell), maxLevelCell(iCell), nVertLevels, iErr)
          !   do k = 1, maxLevelCell(iCell)
          !      restingThickness(k, iCell) = layerThickness(k, iCell)
          !   end do
          ! end do
          !

       end if 
       !!
      
       ! print*, "ssh dim = ", ubound( ssh, 1 ) ; 
       ! print*, "nEdge = ", nEdgesSolve ; 
       ! print*, "nvertLevel = ", nVertLevels ; 
       ! print*, ubound( normalVelocity, 1) ; 
       ! print*, ubound( normalVelocity, 2) ;  
       ! print*, ubound( edgeNormalVectors, 1) ;
       ! print*, ubound( edgeNormalVectors, 2) ; 

       ! Set velocity
       !!
       do iEdge = 1, nEdgesSolve

         RR = sqrt(xEdge(iEdge)**2.0 + yEdge(iEdge)**2.0) ;
         
         HH = (sqrtOneMC2/oneMC) - ( (RR**2.0)/(LL**2.0))*( (oneMC2/(oneMC**2.0))  ) ;
         HH = config_parabolic_bowl_b0*HH  ;
    
          
         uu =  - config_parabolic_bowl_Coriolis_parameter*yEdge(iEdge)*( sqrtOneMC2 + CC - 1.0  ) ;
         vv =    config_parabolic_bowl_Coriolis_parameter*xEdge(iEdge)*( sqrtOneMc2 + CC - 1.0  ) ; 

         uu = 0.5*uu/OneMC ; 
         vv = 0.5*vv/OneMC ; 
         IF ( HH < eps ) THEN
             uu = 0.0 ;
             vv = 0.0 ;
         end if

                                  
         do k = 1, nVertLevels 
            normalVelocity(k,iEdge) = uu*edgeNormalVectors(1,iEdge) + vv*edgeNormalVectors(2,iEdge) ;
         end do
         ! print*, maxval( abs(normalVelocity(:,iedge)) ), edgeNormalVectors(1,iEdge), edgeNormalVectors(2,iEdge), HH ;
       end do
       !!

       do iCell = 1, nCellsSolve
         !
         ! Set temperature
         ! idx = index_temperature
         ! do k = 1, nVertLevels
         !   ! activeTracers(idx, k, iCell) =
         ! end do
         !

         ! Set salinity
         ! idx = index_salinity
         ! do k = 1, nVertLevels
         !   ! activeTracers(idx, k, iCell) =
         ! end do

         ! Set layerThickness and restingThickness
         ! do k = 1, nVertLevels
         !   ! layerThickness(k, iCell) =
         !   ! restingThickness(k, iCell) =
         ! end do

         ! Set bottomDepth
         ! bottomDepth(iCell) =

         ! Set maxLevelCell
         ! maxLevelCell(iCell) =

         ! Set Coriolis parameters, if other than zero
         fCell(iCell) = config_parabolic_bowl_coriolis_parameter ; 
      end do

      do iEdge = 1, nEdgesSolve
         fEdge(iEdge) = config_parabolic_bowl_coriolis_parameter ;  
      end do

      do iVertex = 1, nVerticesSolve
         fVertex(iVertex) =  config_parabolic_bowl_coriolis_parameter ;
      end do

       block_ptr => block_ptr % next
    end do

    deallocate(interfaceLocations)
    !--------------------------------------------------------------------

    print*, "****** Exit gracefully *****" ;

    return ;  
  end subroutine ocn_init_setup_parabolic_bowl!}}}


!***********************************************************************
!
!  routine ocn_init_validate_TEMPLATE
!
!> \brief   Validation for this initial condition
!> \author  FILL_IN_AUTHOR
!> \date    FILL_IN_DATE
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
