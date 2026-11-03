!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing
!
!> \brief Module to handle time-varying forcing for the ocean core
!> \author Steven Brus, Xylar Asay-Davis, LANL
!> \date 2/3/2019 
!> \details
!>  Manages the wind and pressure forcing for the ocean.
!>  Based very closely on the seaice forcing module and
!>  framework written by Adrian Turner.
!
!-----------------------------------------------------------------------

module ocn_time_varying_forcing

  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_forcing
  use mpas_stream_manager
  use ocn_framework_forcing
  use ocn_constants
  use ocn_config
  use ocn_diagnostics_variables
  use mpas_log, only: mpas_log_write

  implicit none

  private
  public :: &
       ocn_time_varying_forcing_init, &
       ocn_time_varying_forcing_get, &
       ocn_time_varying_forcing_write_restart_times, &
       ocn_time_varying_forcing_advance_clocks

contains

!-----------------------------------------------------------------------
! initialization
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_init
!
!> \brief Driver for ocean core forcing initialization
!> \author Steven Brus, Xylar Asay-Davis, LANL
!> \date 2/3/2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_init(domain)!{{{

    type (domain_type) :: domain

    ! init the atmospheric forcing
    if (config_use_time_varying_atmospheric_forcing) then
       call init_atmospheric_forcing(domain)
    endif

    ! init other forcings, to be added later as needed
    if (config_use_time_varying_land_ice_forcing) then
      call init_land_ice_forcing(domain)
    endif

    ! init prescribed river inflow through specified mesh edges
    if (config_use_river_inflow_forcing) then
      call init_river_inflow_forcing(domain)
    endif

    ! init other forcings, to be added later as needed
    !if (config_use_time_varying_****_forcing) then
    !  call init_****_forcing(domain)
    !endif

  end subroutine ocn_time_varying_forcing_init!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_atmospheric_forcing
!
!> \brief
!> \author Steven Brus, LANL
!> \date 1/8/2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine init_atmospheric_forcing(domain)!{{{

    type (domain_type) :: domain

    select case (trim(config_time_varying_atmospheric_forcing_type))
      case ("WINDPRES")
         call init_atmospheric_forcing_WINDPRES(domain)
      case default
         call mpas_log_write("Atmospheric forcing type unknown: "//trim(config_time_varying_atmospheric_forcing_type), MPAS_LOG_CRIT)
    end select

  end subroutine init_atmospheric_forcing!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_atmospheric_forcing_WINDPRES
!
!> \brief Initialize the forcing objects for wind and pressure
!> \author Steven Brus, LANL
!> \date 1/8/2019
!> \details
!>  This routine calls the MPAS_forcing module subroutines that initializes
!>  the forcings type
!
!-----------------------------------------------------------------------

  subroutine init_atmospheric_forcing_WINDPRES(domain)!{{{

    type(domain_type) :: domain

    ! create the atmospheric forcing group
    call MPAS_forcing_init_group(&
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         domain, &                               ! domain
         config_time_varying_atmospheric_forcing_start_time, &            ! startTimeStr
         config_time_varying_atmospheric_forcing_cycle_start, &           ! forcingCycleStart
         config_time_varying_atmospheric_forcing_cycle_duration, &        ! forcingCycleDuration
         config_do_restart)!, &                  ! restart
         !.false.)                               ! forcingCycleStartInclusive

    ! windSpeedU init
    call MPAS_forcing_init_field(&
         domain % streamManager, &               ! streamManager
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         "windSpeedU", &                         ! forcingName
         "atmospheric_forcing", &                ! forcingStreamID
         "timeVaryingForcing", &                 ! poolname
         "windSpeedU", &                         ! fieldname
         "linear", &                             ! interpolationType
         config_time_varying_atmospheric_forcing_reference_time, &                 ! forcingReferenceTimeStr
         config_time_varying_atmospheric_forcing_interval)!,                      ! forcingIntervalStr
         !"next")                                ! forcingInitializationType

    ! windSpeedV init
    call MPAS_forcing_init_field(&
         domain % streamManager, &
         forcingGroupHead, &
         "ocn_atmospheric_forcing", &
         "windSpeedV", &
         "atmospheric_forcing", &
         "timeVaryingForcing", &
         "windSpeedV", &
         "linear", &
         config_time_varying_atmospheric_forcing_reference_time, &
         config_time_varying_atmospheric_forcing_interval)!, &
         !"next")

    ! atmosphericPressure init
    call MPAS_forcing_init_field(&
         domain % streamManager, &
         forcingGroupHead, &
         "ocn_atmospheric_forcing", &
         "atmosPressure", &
         "atmospheric_forcing", &
         "timeVaryingForcing", &
         "atmosPressure", &
         "linear", &
         config_time_varying_atmospheric_forcing_reference_time, &
         config_time_varying_atmospheric_forcing_interval)!, &
         !"next")

    call MPAS_forcing_init_field_data(&
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_atmospheric_forcing", &            ! forcingGroupName
         domain % streamManager, &               ! streamManager
         config_do_restart, &                    ! restart
         .false.)                                ! interpolateAtInit

  end subroutine init_atmospheric_forcing_WINDPRES!}}}

!-----------------------------------------------------------------------
! runtime
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_get
!
!> \brief Retrieve forcing data during time stepping
!> \author Steven Brus, Xylar Asay-Davis, LANL
!> \date 2/3/2019
!> \details
!>  This routine calls the MPAS_forcing routine that will perform the
!>  forcing data aquisition and interpolation during timestepping
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_get(streamManager, domain, simulationClock)!{{{

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    if (config_use_time_varying_atmospheric_forcing) then

        call atmospheric_forcing(&
             streamManager, &
             domain, &
             simulationClock)

    endif

    if (config_use_time_varying_land_ice_forcing) then

        call land_ice_forcing(&
             streamManager, &
             domain, &
             simulationClock)

    endif

    if (config_use_river_inflow_forcing) then

        call river_inflow_forcing(&
             streamManager, &
             domain, &
             simulationClock)

    endif

    ! call other forcings, to be added if needed
    ! if (config_use_****_forcing) then
    !   call ****_forcing(&
    !        streamManager, &
    !        domain, &
    !        simulationClock)
    ! endif


  end subroutine ocn_time_varying_forcing_get!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  atmospheric_forcing
!
!> \brief
!> \author Steven Brus, LANL
!> \date 1/8/2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine atmospheric_forcing(streamManager, domain, simulationClock)!{{{

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    type (MPAS_time_type) :: currentForcingTime

    type (block_type), pointer :: block

    type (MPAS_timeInterval_type) :: timeStepESMF

    character(len=StrKIND) :: timeStamp

    real (kind=RKIND) :: dtSim, dtSimReverse

    real (kind=RKIND), pointer :: forcingTimeIncrement

    type (mpas_pool_type), pointer :: forcingPool

    integer :: err

    call mpas_pool_get_subpool(domain % blocklist % structs, 'forcing', forcingPool)
    call mpas_pool_get_array(forcingPool, 'forcingTimeIncrement', forcingTimeIncrement)    

    ! convert config_dt to real
    call mpas_set_timeInterval(timeStepESMF, timeString=config_dt,ierr=err)
    dtSim = forcingTimeIncrement
    dtSimReverse = -dtSim

    ! use the forcing layer to get data
    call MPAS_forcing_get_forcing(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_atmospheric_forcing", &      ! forcingGroupName
         streamManager, &                  ! streamManager
         dtSim)                            ! dt

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_atmospheric_forcing", &      ! forcingGroupName
         currentForcingTime)               ! forcingTime
    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)
    !call mpas_log_write('Forcing time for atmospheric forcing' // trim(timeStamp))

    call mpas_advance_forcing_clock(forcingGroupHead, dtSimReverse)

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_atmospheric_forcing", &      ! forcingGroupName
         currentForcingTime)               ! forcingTime

    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)
    !call mpas_log_write('Forcing time reversed for atmospheric forcing' // trim(timeStamp))

    ! perform post forcing
    block => domain % blocklist
    do while (associated(block))

       ! apply drag law to compute wind stress from wind speed
       if (trim(config_time_varying_atmospheric_forcing_type) == "WINDPRES") then
          call post_atmospheric_forcing(block)
       endif

       block => block % next
    end do

  end subroutine atmospheric_forcing!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  post_atmospheric_forcing
!
!> \brief Calculate wind stress from wind speed and apply ramp
!> \author Steven Brus, LANL
!> \date 1/8/2019
!> \details
!>  Uses drag law from Garratt, J.R., 1977. Review of drag coefficients
!>  over oceans and continents. Monthly weather review, 105(7), pp.915-929.
!>
!
!-----------------------------------------------------------------------

  subroutine post_atmospheric_forcing(block)!{{{

    type (block_type), pointer :: block

    type (MPAS_time_type) :: currentForcingTime

    type (mpas_pool_type), pointer :: &
         mesh, &
         forcingPool, &
         timeVaryingForcingPool

    real(kind=RKIND), dimension(:), pointer :: &
         windSpeedU, &
         windSpeedV, &
         windSpeedMagnitude, &
         windStressZonal, &
         windStressMeridional, &
         atmosPressure, &
         atmosphericPressure

    real(kind=RKIND) :: &
         windStressCoefficient, &
         rhoAir, &
         ramp, &
         windStressCoefficientLimit, &
         t

    real(kind=RKIND), pointer :: forcingTimeIncrement

    integer, pointer :: &
         nCells

    integer :: &
         iCell

    call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
    call MPAS_pool_get_subpool(block % structs, "forcing", forcingPool)
    call MPAS_pool_get_subpool(block % structs, "timeVaryingForcing", timeVaryingForcingPool)

    call MPAS_pool_get_dimension(mesh, "nCells", nCells)

    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedU", windSpeedU)
    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedV", windSpeedV)
    call MPAS_pool_get_array(timeVaryingForcingPool, "windSpeedMagnitude", windSpeedMagnitude)
    call MPAS_pool_get_array(forcingPool, "windStressZonal", windStressZonal)
    call MPAS_pool_get_array(forcingPool, "windStressMeridional", windStressMeridional)

    call MPAS_pool_get_array(timeVaryingForcingPool, "atmosPressure", atmosPressure)
    call MPAS_pool_get_array(forcingPool, "atmosphericPressure", atmosphericPressure)
    call mpas_pool_get_array(forcingPool, 'forcingTimeIncrement', forcingTimeIncrement)

    rhoAir = 1.225_RKIND
    windStressCoefficientLimit = 0.0035_RKIND

    if (daysSinceStartOfSim >= config_time_varying_atmospheric_forcing_ramp_delay) then
      t = (daysSinceStartOfSim*86400_RKIND + forcingTimeIncrement)/86400.0_RKIND
      ramp = tanh((2.0_RKIND*(t-config_time_varying_atmospheric_forcing_ramp_delay)) &
                              /config_time_varying_atmospheric_forcing_ramp)
    else
      ramp = 0.0_RKIND
    end if

    !$omp parallel
    !$omp do schedule(runtime) private(windStressCoefficient)
    do iCell = 1, nCells

       windSpeedU(iCell) = ramp*windSpeedU(iCell)
       windSpeedV(iCell) = ramp*windSpeedV(iCell)
       windSpeedMagnitude(iCell) = sqrt(windSpeedU(iCell)**2 + windSpeedV(iCell)**2)
       windStressCoefficient = (0.75_RKIND + 0.067_RKIND * windSpeedMagnitude(iCell)) * 0.001_RKIND   ! Garratt 1977
       if (windStressCoefficient > windStressCoefficientLimit) then
         windStressCoefficient = windStressCoefficientLimit
       end if

       windStressZonal(iCell) =  rhoAir * windSpeedU(iCell) * windSpeedMagnitude(iCell) * windStressCoefficient
       windStressMeridional(iCell) = rhoAir * windSpeedV(iCell) * windSpeedMagnitude(iCell) * windStressCoefficient

       atmosphericPressure(iCell) = ((1.0_RKIND-ramp)*atm_ref_pressure + ramp*atmosPressure(iCell)) - atm_ref_pressure

    enddo
    !$omp end do
    !$omp end parallel

  end subroutine post_atmospheric_forcing!}}}


!-----------------------------------------------------------------------
! restart
!-----------------------------------------------------------------------
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_write_restart_times
!
!> \brief
!> \author Steven Brus, LANL
!> \date 1/8/2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_write_restart_times(domain)!{{{

    type(domain_type) :: domain

    call MPAS_forcing_write_restart_times(forcingGroupHead)

  end subroutine ocn_time_varying_forcing_write_restart_times!}}}

!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_land_ice_forcing
!
!> \brief
!> \author Xylar Asay-Davis, LANL
!> \date 2/3/2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine init_land_ice_forcing(domain)!{{{

    type (domain_type) :: domain

    ! create the land-ice forcing group
    call mpas_forcing_init_group(forcingGroupHead, "ocn_land_ice_forcing", domain, config_time_varying_land_ice_forcing_start_time, &
         config_time_varying_land_ice_forcing_cycle_start, config_time_varying_land_ice_forcing_cycle_duration, config_do_restart)

    call mpas_forcing_init_field(domain % streamManager, forcingGroupHead, "ocn_land_ice_forcing", &
         "landIcePressure", "land_ice_forcing", "timeVaryingForcing", "landIcePressureForcing", "linear", &
         config_time_varying_land_ice_forcing_reference_time, config_time_varying_land_ice_forcing_interval)

    call mpas_forcing_init_field(domain % streamManager, forcingGroupHead, "ocn_land_ice_forcing", &
         "landIceFraction", "land_ice_forcing", "timeVaryingForcing", "landIceFractionForcing", "linear", &
         config_time_varying_land_ice_forcing_reference_time, config_time_varying_land_ice_forcing_interval)

    call mpas_forcing_init_field(domain % streamManager, forcingGroupHead, "ocn_land_ice_forcing", &
         "landIceFloatingFraction", "land_ice_forcing", "timeVaryingForcing", "landIceFloatingFractionForcing", "linear", &
         config_time_varying_land_ice_forcing_reference_time, config_time_varying_land_ice_forcing_interval)

    call mpas_forcing_init_field(domain % streamManager, forcingGroupHead, "ocn_land_ice_forcing", &
         "landIceDraft", "land_ice_forcing", "timeVaryingForcing", "landIceDraftForcing", "linear", &
         config_time_varying_land_ice_forcing_reference_time, config_time_varying_land_ice_forcing_interval)

    call mpas_forcing_init_field_data(&
         forcingGroupHead, &                     ! forcingGroupHead
         "ocn_land_ice_forcing", &               ! forcingGroupName
         domain % streamManager, &               ! streamManager
         config_do_restart, &                    ! restart
         .true.)                                 ! interpolateAtInit

  end subroutine init_land_ice_forcing!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_river_inflow_forcing
!
!> \brief Initialize the forcing object for prescribed river inflow
!> \details
!>  The forcing field is dimensioned on nRiverInflowEdges rather than on
!>  the mesh, so it is not decomposed: the MPAS forcing framework reads
!>  it on the IO task and broadcasts the whole array to every task. The
!>  framework itself is dimension-agnostic, so no special handling is
!>  needed here.
!
!-----------------------------------------------------------------------

  subroutine init_river_inflow_forcing(domain)!{{{

    type(domain_type) :: domain

    ! create the river inflow forcing group
    call MPAS_forcing_init_group(&
         forcingGroupHead, &                                  ! forcingGroupHead
         "ocn_river_inflow_forcing", &                        ! forcingGroupName
         domain, &                                            ! domain
         config_river_inflow_forcing_start_time, &            ! startTimeStr
         config_river_inflow_forcing_cycle_start, &           ! forcingCycleStart
         config_river_inflow_forcing_cycle_duration, &        ! forcingCycleDuration
         config_do_restart)                                   ! restart

    ! riverInflowVolumeFlux init
    call MPAS_forcing_init_field(&
         domain % streamManager, &                            ! streamManager
         forcingGroupHead, &                                  ! forcingGroupHead
         "ocn_river_inflow_forcing", &                        ! forcingGroupName
         "riverInflowVolumeFlux", &                           ! forcingName
         "river_inflow_forcing", &                            ! forcingStreamID
         "timeVaryingForcing", &                              ! poolname
         "riverInflowVolumeFlux", &                           ! fieldname
         "linear", &                                          ! interpolationType
         config_river_inflow_forcing_reference_time, &        ! forcingReferenceTimeStr
         config_river_inflow_forcing_interval)                ! forcingIntervalStr

    ! riverInflowVelocity init, only when its package is active. It shares
    ! the stream with riverInflowVolumeFlux, so it must be registered too:
    ! an unregistered field in a forcing stream is overwritten by each raw
    ! read and never interpolated.
    if (config_use_river_inflow_momentum .and. &
        trim(config_river_inflow_velocity_source) == 'file') then
       call MPAS_forcing_init_field(&
            domain % streamManager, &                         ! streamManager
            forcingGroupHead, &                               ! forcingGroupHead
            "ocn_river_inflow_forcing", &                     ! forcingGroupName
            "riverInflowVelocity", &                          ! forcingName
            "river_inflow_forcing", &                         ! forcingStreamID
            "timeVaryingForcing", &                           ! poolname
            "riverInflowVelocity", &                          ! fieldname
            "linear", &                                       ! interpolationType
            config_river_inflow_forcing_reference_time, &     ! forcingReferenceTimeStr
            config_river_inflow_forcing_interval)             ! forcingIntervalStr
    endif

    call MPAS_forcing_init_field_data(&
         forcingGroupHead, &                                  ! forcingGroupHead
         "ocn_river_inflow_forcing", &                        ! forcingGroupName
         domain % streamManager, &                            ! streamManager
         config_do_restart, &                                 ! restart
         .true.)                                              ! interpolateAtInit

  end subroutine init_river_inflow_forcing!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  river_inflow_forcing
!
!> \brief Retrieve the prescribed river inflow for the current time
!> \details
!>  Interpolates riverInflowVolumeFlux, and riverInflowVelocity when it is
!>  in use, to the current forcing time. The interpolated arrays are
!>  consumed by ocn_river_inflow_build_array, so
!>  there is no post-forcing step here.
!
!-----------------------------------------------------------------------

  subroutine river_inflow_forcing(streamManager, domain, simulationClock)!{{{

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    type (MPAS_time_type) :: currentForcingTime

    type (MPAS_timeInterval_type) :: timeStepESMF

    type (mpas_pool_type), pointer :: forcingPool

    character(len=StrKIND) :: timeStamp

    real (kind=RKIND) :: dtSim, dtSimReverse

    real (kind=RKIND), pointer :: forcingTimeIncrement

    integer :: err

    call mpas_pool_get_subpool(domain % blocklist % structs, 'forcing', forcingPool)
    call mpas_pool_get_array(forcingPool, 'forcingTimeIncrement', forcingTimeIncrement)

    ! convert config_dt to real
    call mpas_set_timeInterval(timeStepESMF, timeString=config_dt, ierr=err)
    dtSim = forcingTimeIncrement
    dtSimReverse = -dtSim

    ! use the forcing layer to get data
    call MPAS_forcing_get_forcing(&
         forcingGroupHead, &                 ! forcingGroupHead
         "ocn_river_inflow_forcing", &       ! forcingGroupName
         streamManager, &                    ! streamManager
         dtSim)                              ! dt

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &                 ! forcingGroupHead
         "ocn_river_inflow_forcing", &       ! forcingGroupName
         currentForcingTime)                 ! forcingTime
    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)

    ! MPAS_forcing_get_forcing advanced this group's clock, so it is this
    ! group that must be stepped back. forcingGroupHead is the first group
    ! created, normally the atmospheric one, not this group.
    call advance_group_clock("ocn_river_inflow_forcing", dtSimReverse)

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &                 ! forcingGroupHead
         "ocn_river_inflow_forcing", &       ! forcingGroupName
         currentForcingTime)                 ! forcingTime
    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)

  end subroutine river_inflow_forcing!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_varying_forcing_advance_clocks
!
!> \brief Advance the clock of every forcing group by dt
!> \details
!>  mpas_advance_forcing_clock acts on the single group it is given, so
!>  passing forcingGroupHead advances only the first group in the list.
!>  When more than one group is active, each must be advanced.
!
!-----------------------------------------------------------------------

  subroutine ocn_time_varying_forcing_advance_clocks(dt)!{{{

    real (kind=RKIND), intent(in) :: dt

    type (mpas_forcing_group_type), pointer :: forcingGroup

    forcingGroup => forcingGroupHead
    do while (associated(forcingGroup))
       call mpas_advance_forcing_clock(forcingGroup, dt)
       forcingGroup => forcingGroup % next
    enddo

  end subroutine ocn_time_varying_forcing_advance_clocks!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  advance_group_clock
!
!> \brief Advance the clock of the named forcing group by dt
!
!-----------------------------------------------------------------------

  subroutine advance_group_clock(forcingGroupName, dt)!{{{

    character(len=*), intent(in) :: forcingGroupName

    real (kind=RKIND), intent(in) :: dt

    type (mpas_forcing_group_type), pointer :: forcingGroup

    forcingGroup => forcingGroupHead
    do while (associated(forcingGroup))
       if (trim(forcingGroup % forcingGroupName) == trim(forcingGroupName)) then
          call mpas_advance_forcing_clock(forcingGroup, dt)
          return
       endif
       forcingGroup => forcingGroup % next
    enddo

  end subroutine advance_group_clock!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  land_ice_forcing
!
!> \brief Compute land-ice pressure
!> \author Xylar Asay-Davis, LANL
!> \date 2/3/2019
!> \details Interpolate the pressure from overlying land ice
!>
!
!-----------------------------------------------------------------------

  subroutine land_ice_forcing(streamManager, domain, simulationClock)!{{{

    type (MPAS_streamManager_type), intent(inout) :: streamManager

    type (domain_type) :: domain

    type (MPAS_clock_type) :: simulationClock

    type (MPAS_time_type) :: currentForcingTime

    type (block_type), pointer :: block

    character(len=StrKIND) :: timeStamp

    type (MPAS_timeInterval_type) :: timeStepESMF

    real (kind=RKIND) :: dtSim, dtSimReverse

    real (kind=RKIND), pointer :: forcingTimeIncrement

    integer :: err

    type (mpas_pool_type), pointer :: &
         mesh, &
         forcingPool, &
         timeVaryingForcingPool

    real(kind=RKIND), dimension(:), pointer :: &
         landIceFractionForcing, &
         landIceFraction, &
         landIceFloatingFractionForcing, &
         landIceFloatingFraction, &
         landIcePressureForcing, &
         landIcePressure, &
         landIceDraftForcing, &
         landIceDraft

    integer, pointer :: &
         nCells

    integer :: &
         iCell

    call mpas_pool_get_subpool(domain % blocklist % structs, 'forcing', forcingPool)
    call mpas_pool_get_array(forcingPool, 'forcingTimeIncrement', forcingTimeIncrement)

    ! convert config_dt to real
    call mpas_set_timeInterval(timeStepESMF, timeString=config_dt,ierr=err)
    call mpas_get_timeInterval(timeStepESMF, dt=dtSim)

    dtSim = forcingTimeIncrement
    dtSimReverse = -dtSim

    ! use the forcing layer to get data
    call MPAS_forcing_get_forcing(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_land_ice_forcing", &         ! forcingGroupName
         streamManager, &                  ! streamManager
         dtSim)                            ! dt

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_land_ice_forcing", &         ! forcingGroupName
         currentForcingTime)               ! forcingTime

    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)
    call mpas_log_write('Forcing time for land ice forcing' // trim(timeStamp))


    call mpas_advance_forcing_clock(forcingGroupHead, dtSimReverse)

    call MPAS_forcing_get_forcing_time(&
         forcingGroupHead, &               ! forcingGroupHead
         "ocn_land_ice_forcing", &         ! forcingGroupName
         currentForcingTime)               ! forcingTime

    call mpas_get_time(curr_time=currentForcingTime, dateTimeString=timeStamp, ierr=err)
    call mpas_log_write('Forcing time reversed for land ice forcing' // trim(timeStamp))

    block => domain % blocklist
    do while (associated(block))

       call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
       call MPAS_pool_get_subpool(block % structs, "forcing", forcingPool)
       call MPAS_pool_get_subpool(block % structs, "timeVaryingForcing", timeVaryingForcingPool)
   
       call MPAS_pool_get_dimension(mesh, "nCells", nCells)
   
       call MPAS_pool_get_array(timeVaryingForcingPool, "landIceFractionForcing", landIceFractionForcing)
       call MPAS_pool_get_array(timeVaryingForcingPool, "landIceFloatingFractionForcing", landIceFloatingFractionForcing)
       call MPAS_pool_get_array(timeVaryingForcingPool, "landIcePressureForcing", landIcePressureForcing)
       call MPAS_pool_get_array(timeVaryingForcingPool, "landIceDraftForcing", landIceDraftForcing)
       call MPAS_pool_get_array(forcingPool, "landIceFraction", landIceFraction)
       call MPAS_pool_get_array(forcingPool, "landIceFloatingFraction", landIceFloatingFraction)
       call MPAS_pool_get_array(forcingPool, "landIcePressure", landIcePressure)
       call MPAS_pool_get_array(forcingPool, "landIceDraft", landIceDraft)
   
       do iCell = 1, nCells
   
          landIceFraction(iCell) = landIceFractionForcing(iCell)
          landIceFloatingFraction(iCell) = landIceFloatingFractionForcing(iCell)
          landIcePressure(iCell) = landIcePressureForcing(iCell)
          landIceDraft(iCell) = landIceDraftForcing(iCell)
   
       enddo
       block => block % next
    end do

  end subroutine land_ice_forcing!}}}

end module ocn_time_varying_forcing
! vim: foldmethod=marker
