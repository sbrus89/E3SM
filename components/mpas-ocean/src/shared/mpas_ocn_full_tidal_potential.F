! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_full_tidal_potential
!
!> \brief MPAS-Ocean full lunar-solar tidal potential module
!> \author D. Wirasaet, A. Cerrone
!> \date   2024
!> \details
!>  This module computes the full lunar-solar equilibrium tidal potential
!>  using different orders of approximation.
!>
!>  References:
!>    J.-M. Hervouet, Free surface flows: modelling with the finite
!>       element method. John Wiley & Sons, Ltd, 2007
!>
!>    Z. Kowalik and J. L. Luick, Modern theory and practice of tide
!>       analysis and tidal power, Austides consulting, 2019
!>       https://austides.com/downloads/
!
!-----------------------------------------------------------------------

module ocn_full_tidal_potential

   use mpas_kind_types, only: RKIND
   use mpas_log, only: mpas_log_write
   use astroformod, only: DEG2RAD, JULIANDAY
   use moon_sun_coors, only: HEAVENLY_OBJS_COORDS_JM, SET_NUTATION, &
                              GMST_DEG_FN
   use ephemri_module, only: HEAVENLY_OBJS_COORDS_FROM_TABLE

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_full_tidal_potential_init, &
             ocn_compute_full_tidal_potential

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   !*** configuration
   integer :: tip_order = 2
   character(len=24) :: moon_sun_position_method = 'JM'
   character(len=80) :: moon_sun_coord_file = 'none'
   real(kind=RKIND) :: k2_love_number = 0.302_RKIND
   real(kind=RKIND) :: h2_love_number = 0.609_RKIND

   !*** state
   logical :: initialized = .false.
   real(kind=RKIND) :: jde_beg

   !*** work arrays allocated on first init
   real(kind=RKIND), allocatable, dimension(:) :: &
      hour_angle,      &! local hour angle at each cell
      cos_zenith_angle, &! cosine of zenith angle at each cell
      work_arr,        &! general-purpose work array
      work_arr1         ! secondary work array (used for exact formula)

   !*** astronomical constants
   !*** mass ratios relative to Earth: (/ Moon, Sun /)
   real(kind=RKIND), parameter :: mass_ratio(2) = &
      (/ 1.0_RKIND/81.3005678_RKIND, 332946.0487_RKIND /)

   !*** Earth radius significant part (km for Moon, AU for Sun)
   real(kind=RKIND), parameter :: radius_earth_sig(2) = &
      (/ 6371.0088_RKIND, 4.258756338033895_RKIND /)

   !*** Earth radius exponent factor (unit conversion)
   real(kind=RKIND), parameter :: radius_earth_exp(2) = &
      (/ 1.0_RKIND, 1.0e-5_RKIND /)

   !*** Earth radius in meters (split into significant × exponent for precision)
   real(kind=RKIND), parameter :: earth_radius_m_sig = 6.3710088_RKIND
   real(kind=RKIND), parameter :: earth_radius_m_exp = 1.0e6_RKIND

   !*** time conversion
   real(kind=RKIND), parameter :: sec2day  = 1.0_RKIND / 86400.0_RKIND
   real(kind=RKIND), parameter :: min2day  = 1.0_RKIND / 1440.0_RKIND
   real(kind=RKIND), parameter :: hour2day = 1.0_RKIND / 24.0_RKIND

   !*** generic interface covering both Moon/Sun position methods
   interface heavenly_objs_coords
      module procedure HEAVENLY_OBJS_COORDS_JM
      module procedure HEAVENLY_OBJS_COORDS_FROM_TABLE
   end interface heavenly_objs_coords

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_full_tidal_potential_init
!
!> \brief   Initializes the full lunar-solar tidal potential module
!> \author  D. Wirasaet, A. Cerrone
!> \date    2024
!> \details
!>  Parses the simulation start date, computes the corresponding Julian
!>  Day, allocates work arrays, and configures the Moon/Sun position
!>  method and Love numbers.  Must be called before
!>  ocn_compute_full_tidal_potential.
!>
!>  The start_date_str argument must have the format
!>  'YYYY-MM-DD HH:MM:SS' (space separator between date and time).
!
!-----------------------------------------------------------------------

   subroutine ocn_full_tidal_potential_init(start_date_str, num_cells, &!{{{
                                            order, include_nutation,    &
                                            k2, h2, err)

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: &
         start_date_str    !< [in] simulation start date 'YYYY-MM-DD HH:MM:SS'

      integer, intent(in) :: &
         num_cells,        &!< [in] number of mesh cells
         order              !< [in] order of approximation (2, 3, or >=4 for exact)

      logical, intent(in) :: &
         include_nutation   !< [in] whether to include nutation correction

      real(kind=RKIND), intent(in) :: &
         k2,               &!< [in] k2 Love number
         h2                 !< [in] h2 Love number

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: year, month, day, hour, minute, second
      real(kind=RKIND) :: day_frac

      character(len=80) :: date_buf

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      err = 0

      !*** store configuration
      tip_order             = order
      k2_love_number        = k2
      h2_love_number        = h2

      !*** parse date string into integer components
      date_buf = adjustl(start_date_str)
      call extract_date_values(date_buf, '-', year, month, day)
      call extract_date_values(date_buf, ':', hour, minute, second)

      !*** compute fractional day and corresponding Julian Day
      day_frac = real(day, RKIND)    &
               + real(hour,   RKIND) * hour2day &
               + real(minute, RKIND) * min2day  &
               + real(second, RKIND) * sec2day

      jde_beg = JULIANDAY(day_frac, month, year)

      !*** configure nutation in the Moon/Sun coordinate module
      call SET_NUTATION(include_nutation)

      !*** allocate work arrays
      call allocate_work_arr(hour_angle,       num_cells)
      call allocate_work_arr(cos_zenith_angle, num_cells)
      call allocate_work_arr(work_arr,         num_cells)

      if (tip_order > 3) then
         call allocate_work_arr(work_arr1, num_cells)
      end if

      initialized = .true.

   end subroutine ocn_full_tidal_potential_init!}}}

!***********************************************************************
!
!  routine ocn_compute_full_tidal_potential
!
!> \brief   Computes the full lunar-solar equilibrium tidal potential
!> \author  D. Wirasaet, A. Cerrone
!> \date    2024
!> \details
!>  Computes the equilibrium tidal potential at every cell centre by
!>  summing Moon and Sun contributions.  Three formulations are
!>  available, selected by tip_order set in ocn_full_tidal_potential_init:
!>    tip_order == 2  : Legendre P2 term only
!>    tip_order == 3  : P2 + P3 terms
!>    tip_order >= 4  : exact (full) expression (Proudman)
!>
!>  The Love-number correction factor (1 + k2 - h2) is applied to the
!>  total potential before returning.
!>
!>  time_loc is measured in seconds since the date passed to
!>  ocn_full_tidal_potential_init.
!
!-----------------------------------------------------------------------

   subroutine ocn_compute_full_tidal_potential(tip_values, time_loc, &!{{{
                                               lon_rad, lat_rad)

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real(kind=RKIND), intent(in) :: &
         time_loc          !< [in] seconds elapsed since start date

      real(kind=RKIND), dimension(:), intent(in) :: &
         lon_rad,          &!< [in] cell longitudes  (radians)
         lat_rad            !< [in] cell latitudes   (radians)

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real(kind=RKIND), dimension(:), intent(inout) :: &
         tip_values        !< [inout] tidal potential at each cell (m^2 s^-2)

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: i_obj, i_exp, num_cells
      integer :: ierr

      real(kind=RKIND) :: jde_loc, gmst_deg
      real(kind=RKIND) :: moon_sun_coor(3,2)
      real(kind=RKIND) :: ra, dec, delta
      real(kind=RKIND) :: radius_div_delta, kp, c0

      ! End preamble
      !-----------------------------------------------------------------
      ! Begin code

      num_cells = size(tip_values)

      !*** Julian Day at the current time
      jde_loc = jde_beg + time_loc * sec2day

      !*** Moon and Sun positions (RA in deg, DEC in deg, distance)
      if (trim(moon_sun_position_method) == 'JM') then
         call heavenly_objs_coords(moon_sun_coor(:,1), &
                                   moon_sun_coor(:,2), jde_loc, ierr)
      else
         call heavenly_objs_coords(moon_sun_coor, jde_loc, &
                                   moon_sun_coord_file, ierr)
      end if

      !*** convert RA and DEC from degrees to radians
      moon_sun_coor(1:2,:) = moon_sun_coor(1:2,:) * DEG2RAD

      !*** Greenwich Mean Sidereal Time (degrees)
      gmst_deg = GMST_DEG_FN(jde_loc)

      tip_values = 0.0_RKIND

      do i_obj = 1, 2

         ra    = moon_sun_coor(1, i_obj) ! right ascension (rad)
         dec   = moon_sun_coor(2, i_obj) ! declination     (rad)
         delta = moon_sun_coor(3, i_obj) ! distance (km for Moon, AU for Sun)

         !*** local hour angle at each cell
         hour_angle = DEG2RAD * gmst_deg + lon_rad - ra

         !*** ratio of Earth's radius to celestial body distance
         radius_div_delta = radius_earth_sig(i_obj) / delta
         radius_div_delta = radius_div_delta * radius_earth_exp(i_obj)

         !*** cosine of the zenith angle
         cos_zenith_angle = sin(lat_rad) * sin(dec) &
                          + cos(lat_rad) * cos(dec) * cos(hour_angle)

         select case (tip_order)

         case (2, 3)

            !*** P2 term
            work_arr = 0.5_RKIND * (3.0_RKIND * cos_zenith_angle**2 - 1.0_RKIND)

            kp = aint_power(radius_div_delta, 3) &
               * earth_radius_m_sig * earth_radius_m_exp &
               * mass_ratio(i_obj)

            tip_values = tip_values + kp * work_arr

            !*** P3 term (added when tip_order == 3)
            if (tip_order == 3) then
               work_arr = 0.5_RKIND &
                        * (5.0_RKIND * cos_zenith_angle**3 &
                           - 3.0_RKIND * cos_zenith_angle)

               kp = aint_power(radius_div_delta, 4) &
                  * earth_radius_m_sig * earth_radius_m_exp &
                  * mass_ratio(i_obj)

               tip_values = tip_values + kp * work_arr
            end if

         case (22)

            !*** P2 separated into long-period, diurnal, and semi-diurnal species

            ! long-period species
            work_arr1 = sin(lat_rad)
            work_arr = 0.25_RKIND &
                     * (1.0_RKIND - 3.0_RKIND * work_arr1**2) &
                     * (1.0_RKIND - 3.0_RKIND * sin(dec)**2)

            ! diurnal species
            work_arr = work_arr &
                     + 0.75_RKIND * sin(2.0_RKIND * lat_rad) &
                                  * sin(2.0_RKIND * dec) &
                                  * cos(hour_angle)

            ! semi-diurnal species
            work_arr1 = cos(lat_rad)
            work_arr = work_arr &
                     + 0.75_RKIND * work_arr1**2 &
                                  * cos(dec)**2 &
                                  * cos(2.0_RKIND * hour_angle)

            kp = aint_power(radius_div_delta, 3) &
               * earth_radius_m_sig * earth_radius_m_exp &
               * mass_ratio(i_obj)

            tip_values = tip_values + kp * work_arr

         case default

            !*** exact expression (Proudman approach; Kowalik, p. 14)
            work_arr = 1.0_RKIND / sqrt(1.0_RKIND &
                     + aint_power(radius_div_delta, 2) &
                     - 2.0_RKIND * radius_div_delta * cos_zenith_angle) &
                     - radius_div_delta * cos_zenith_angle

            !*** constant of integration (offset)
            c0 = sqrt(1.0_RKIND + aint_power(radius_div_delta, 2) &
                       + 2.0_RKIND * radius_div_delta) &
               - sqrt(1.0_RKIND + aint_power(radius_div_delta, 2) &
                       - 2.0_RKIND * radius_div_delta)
            c0 = -0.5_RKIND * c0 / radius_div_delta

            work_arr = work_arr + c0

            kp = aint_power(radius_div_delta, 1) &
               * earth_radius_m_sig * earth_radius_m_exp &
               * mass_ratio(i_obj)

            tip_values = tip_values + kp * work_arr

         end select

      end do

      !*** apply Love-number correction
      tip_values = (1.0_RKIND + k2_love_number - h2_love_number) * tip_values

   end subroutine ocn_compute_full_tidal_potential!}}}

!***********************************************************************
!
!  function aint_power
!
!> \brief   Integer power of a scalar
!> \author  D. Wirasaet
!> \date    2024
!> \details
!>  Returns a^n computed by repeated multiplication.
!>  Elemental so it may be applied array-wide.
!
!-----------------------------------------------------------------------

   elemental function aint_power(a, n) result(ap) !{{{

      real(kind=RKIND), intent(in) :: a
      integer, intent(in) :: n
      real(kind=RKIND) :: ap

      integer :: i

      ap = 1.0_RKIND
      do i = 1, n
         ap = ap * a
      end do

   end function aint_power!}}}

!***********************************************************************
!
!  subroutine allocate_work_arr
!
!> \brief   Allocates or reallocates a 1-D work array
!> \author  D. Wirasaet
!> \date    2024
!> \details
!>  Deallocates arr if already allocated, then allocates it with
!>  n elements initialised to zero.
!
!-----------------------------------------------------------------------

   subroutine allocate_work_arr(arr, n) !{{{

      real(kind=RKIND), allocatable, intent(inout) :: arr(:)
      integer, intent(in) :: n

      if (allocated(arr)) deallocate(arr)
      allocate(arr(n))
      arr = 0.0_RKIND

   end subroutine allocate_work_arr!}}}

!***********************************************************************
!
!  subroutine extract_date_values
!
!> \brief   Parses three integer values from a delimited date string
!> \author  D. Wirasaet
!> \date    2024
!> \details
!>  Reads three consecutive tokens from buf, separated by delim_char or
!>  by a space when delim_char is not found.  buf is consumed in place
!>  (the parsed portion is discarded after each read).
!
!-----------------------------------------------------------------------

   subroutine extract_date_values(buf, delim_char, arg1, arg2, arg3) !{{{

      character(len=80), intent(inout) :: buf
      character, intent(in) :: delim_char
      integer, intent(out) :: arg1, arg2, arg3

      integer :: ii, pos, values(3)

      do ii = 1, 3
         pos = index(buf, delim_char)
         if (pos == 0) pos = index(buf, ' ')

         if (pos > 0) then
            read(buf(1:pos-1), *) values(ii)
            buf = adjustl(buf(pos+1:))
         else
            read(buf, *) values(ii)
         end if
      end do

      arg1 = values(1)
      arg2 = values(2)
      arg3 = values(3)

   end subroutine extract_date_values!}}}

!***********************************************************************

end module ocn_full_tidal_potential!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
