"""
Brooks, Pope, Marcolini broadband noise model.
Copyright (C) 2018 Eric Greenwood

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
import numpy as np
from acoustics.decibel import dbsum
from acoustics.standards.iec_61260_1_2014 import _nominal_center_frequency
from acoustics.standards.iec_61260_1_2014 import exact_center_frequency

exact_center_frequencies = np.array([exact_center_frequency(x, fraction=3) for x in range(-17, 14)])
nominal_third_octave_frequencies = np.array([_nominal_center_frequency(x,3) for x in exact_center_frequencies])


def directivity_function_high(theta, phi, mach, rotor=True, mach_convection=None):
    """
    Directivity function for high frequency sources
    :param theta: Angle from source streamwise axis to observer, deg
    :param phi: Angle from source lateral axis to observer, deg
    :param mach: Mach number
    :param rotor: Set to true to use rotor high-frequency directivity function from Burley & Brooks, JAHS 2004.
    :param mach_convection: Optional specification of Mach number of convecting sources, default None is 0.8*mach.  No effect when rotor=True.
    :return: D_h
    """
    theta = np.radians(theta)
    phi = np.radians(phi)
    if rotor:
        return 2.0 * np.sin(0.5 * theta) ** 2 * np.sin(phi) ** 2 / (1.0 + mach * np.cos(theta)) ** 4
    else:
        if mach_convection is None:
            mach_convection = 0.8 * mach
        return 2.0 * np.sin(0.5 * theta) ** 2 * np.sin(phi) ** 2 / (
                (1 + mach * np.cos(theta)) * (1.0 + (mach - mach_convection) * np.cos(theta)) ** 2)


def directivity_function_low(theta, phi, mach):
    """
    Directivity function for low frequency sources
    :param theta: Angle from source streamwise axis to observer, deg
    :param phi: Angle from source lateral axis to observer, deg
    :param mach: Mach number
    :return: D_l
    """
    theta = np.radians(theta)
    phi = np.radians(phi)
    return np.sin(theta) ** 2 * np.sin(phi) ** 2 / (1.0 + mach * np.cos(theta)) ** 4


def boundary_layer_model(alpha_e, chord_reynolds_number, tripped=False):
    """
    BPM empirical boundary layer model (originally developed from NACA 0012 in QFF)
    :param alpha_e: Effective angle of attack, deg, relative to zero-lift AoA
    :param chord_reynolds_number: Reynold's number based on chord
    :param tripped: Set true for tripped condition, false for natural
    :return: Suction side boundary layer thickness, suction side displacement thickness,
    suction side momentum thickness, pressure side boundary layer thickness, pressure side displacement thickness,
    pressure side momentum thickness, zero lift boundary layer thickness, zero lift displacement thickness, zero lift
    momentum thickness.  All given in units of chords.
    """
    # Don't care which side is suction or pressure side, so just take absolute value of AoA
    alpha_e = np.atleast_1d(np.fabs(alpha_e))
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    # Pre-compute log10(re) since it's used many times
    log_re = np.log10(chord_reynolds_number)
    # Calculate BL thicknesses at zero-lift-AoA (alpha_e = 0)
    if tripped:
        thickness_0 = 10.0 ** (1.892 - 0.9045 * log_re + 0.0596 * log_re ** 2)
        # Values for Re_c <= 0.3e6
        displacement_thickness_0 = 0.0601 * chord_reynolds_number ** -0.114
        momentum_thickness_0 = 0.0723 * chord_reynolds_number ** -0.1765
        # Correct for higher Re_c
        mask = chord_reynolds_number > 0.3e6
        displacement_thickness_0[mask] = 10.0 ** (3.411 - 1.5397 * log_re[mask] + 0.1059 * log_re[mask] ** 2)
        momentum_thickness_0[mask] = 10.0 ** (0.5578 - 0.7079 * log_re[mask] + 0.0404 * log_re[mask] ** 2)
    else:
        thickness_0 = 10.0 ** (1.6569 - 0.9045 * log_re + 0.0596 * log_re ** 2)
        displacement_thickness_0 = 10.0 ** (3.0187 - 1.5397 * log_re + 0.1059 * log_re ** 2)
        momentum_thickness_0 = 10.0 ** (0.2021 - 0.7079 * log_re + 0.0404 * log_re ** 2)

    # Correct pressure side for AoA
    thickness_pressure = thickness_0 * 10.0 ** (-0.04175 * alpha_e + 0.00106 * alpha_e ** 2)
    displacement_thickness_pressure = displacement_thickness_0 * 10.0 ** (-0.0432 * alpha_e + 0.00113 * alpha_e ** 2)
    momentum_thickness_pressure = momentum_thickness_0 * 10.0 ** (-0.04508 * alpha_e + 0.000873 * alpha_e ** 2)

    # Correct suction side for AoA; depends on whether flow is tripped
    if tripped:
        # Set up angle of attack ranges
        middle = np.logical_and(alpha_e <= 12.5, alpha_e > 5.0)
        low = alpha_e <= 5

        thickness_suction = 5.718 * 10.0 ** (0.0258 * alpha_e)
        displacement_thickness_suction = 14.296 * 10.0 ** (0.0258 * alpha_e)
        momentum_thickness_suction = 4.0846 * 10.0 ** (0.0258 * alpha_e)

        thickness_suction[middle] = 0.3468 * 10.0 ** (0.1231 * alpha_e[middle])
        displacement_thickness_suction[middle] = 0.381 * 10.0 ** (0.1516 * alpha_e[middle])
        momentum_thickness_suction[middle] = 0.6984 * 10.0 ** (0.0869 * alpha_e[middle])

        thickness_suction[low] = 10.0 ** (0.0311 * alpha_e[low])
        displacement_thickness_suction[low] = 10.0 ** (0.0679 * alpha_e[low])
        momentum_thickness_suction[low] = 10.0 ** (0.0559 * alpha_e[low])
    else:
        # Set up angle of attack ranges
        middle = np.logical_and(alpha_e <= 12.5, alpha_e > 7.5)
        low = alpha_e <= 7.5

        thickness_suction = 12.0 * 10.0 ** (0.0258 * alpha_e)
        displacement_thickness_suction = 52.42 * 10.0 ** (0.0258 * alpha_e)
        momentum_thickness_suction = 14.977 * 10.0 ** (0.0258 * alpha_e)

        thickness_suction[middle] = 0.0303 * 10.0 ** (0.2336 * alpha_e[middle])
        displacement_thickness_suction[middle] = 0.0162 * 10.0 ** (0.3066 * alpha_e[middle])
        momentum_thickness_suction[middle] = 0.0633 * 10.0 ** (0.2157 * alpha_e[middle])

        thickness_suction[low] = 10.0 ** (0.03114 * alpha_e[low])
        displacement_thickness_suction[low] = 10.0 ** (0.0679 * alpha_e[low])
        momentum_thickness_suction[low] = 10.0 ** (0.0559 * alpha_e[low])

    thickness_suction = thickness_suction * thickness_0
    displacement_thickness_suction = displacement_thickness_suction * displacement_thickness_0
    momentum_thickness_suction = momentum_thickness_suction * momentum_thickness_0
    return (thickness_suction, displacement_thickness_suction, momentum_thickness_suction, thickness_pressure,
            displacement_thickness_pressure, momentum_thickness_pressure, thickness_0, displacement_thickness_0,
            momentum_thickness_0)


def RP1218_BL_plots():
    """
    Plots boundary layer model validation plots from Section 3 of RP1218.
    """
    chord_reynolds_numbers = np.geomspace(.04e6, 3e6, 100)
    angle_of_attack = 0.0
    btsu0, dtsu0, mtsu0, btpu0, dtpu0, mtpu0, _, _, _ = boundary_layer_model(angle_of_attack, chord_reynolds_numbers,
                                                                             tripped=False)
    btst0, dtst0, mtst0, btpt0, dtpt0, mtpt0, _, _, _ = boundary_layer_model(angle_of_attack, chord_reynolds_numbers,
                                                                             tripped=True)
    # Replicate figure 6
    _, ax = plt.subplots(nrows=3)
    ax[0].loglog(chord_reynolds_numbers, btsu0, 'r-')
    ax[0].loglog(chord_reynolds_numbers, btpu0, 'k-')
    ax[0].loglog(chord_reynolds_numbers, btst0, 'r--')
    ax[0].loglog(chord_reynolds_numbers, btpt0, 'k--')
    ax[0].set_xlim((0.04e6, 3e6))
    ax[0].set_ylim((0.01, .2))
    ax[0].set_ylabel('$\delta_0/c$')

    ax[1].loglog(chord_reynolds_numbers, dtsu0, 'r-')
    ax[1].loglog(chord_reynolds_numbers, dtpu0, 'k-')
    ax[1].loglog(chord_reynolds_numbers, dtst0, 'r--')
    ax[1].loglog(chord_reynolds_numbers, dtpt0, 'k--')
    ax[1].set_xlim((0.04e6, 3e6))
    ax[1].set_ylim((0.001, .03))
    ax[1].set_ylabel('$\delta^*_0/c$')

    ax[2].loglog(chord_reynolds_numbers, mtsu0, 'r-')
    ax[2].loglog(chord_reynolds_numbers, mtpu0, 'k-')
    ax[2].loglog(chord_reynolds_numbers, mtst0, 'r--')
    ax[2].loglog(chord_reynolds_numbers, mtpt0, 'k--')
    ax[2].set_xlim((0.04e6, 3e6))
    ax[2].set_ylim((0.001, .02))
    ax[2].set_ylabel(r'$\theta_0/c$')
    ax[2].set_xlabel(r'$Re_c$')

    # Figure 7
    _, ax = plt.subplots(nrows=3)
    chord_reynolds_number = .5e6
    angles_of_attack = np.linspace(0, 25, 100)
    btst, dtst, mtst, btpt, dtpt, mtpt, bt0, dt0, mt0 = boundary_layer_model(angles_of_attack, chord_reynolds_number,
                                                                             tripped=True)
    ax[0].semilogy(angles_of_attack, btst / bt0, 'k--')
    ax[0].semilogy(angles_of_attack, btpt / bt0, 'k-')
    ax[0].set_xlim((0, 25))
    ax[0].set_ylim((.2, 20))
    ax[0].set_ylabel('$\delta/\delta_0$')

    ax[1].semilogy(angles_of_attack, dtst / dt0, 'k--')
    ax[1].semilogy(angles_of_attack, dtpt / dt0, 'k-')
    ax[1].set_xlim((0, 25))
    ax[1].set_ylim((.2, 50))
    ax[1].set_ylabel('$\delta^*/\delta^*_0$')

    ax[2].semilogy(angles_of_attack, mtst / mt0, 'k--')
    ax[2].semilogy(angles_of_attack, mtpt / mt0, 'k-')
    ax[2].set_xlim((0, 25))
    ax[2].set_ylim((.1, 20))
    ax[2].set_ylabel(r'$\theta/\theta_0$')
    ax[2].set_xlabel(r'$\alpha_*$')

    # Figure 8
    _, ax = plt.subplots(nrows=3)
    chord_reynolds_number = .5e6
    angles_of_attack = np.linspace(0, 25, 100)
    btst, dtst, mtst, btpt, dtpt, mtpt, bt0, dt0, mt0 = boundary_layer_model(angles_of_attack, chord_reynolds_number,
                                                                             tripped=False)
    ax[0].semilogy(angles_of_attack, btst / bt0, 'k--')
    ax[0].semilogy(angles_of_attack, btpt / bt0, 'k-')
    ax[0].set_xlim((0, 25))
    ax[0].set_ylim((.2, 40))
    ax[0].set_ylabel('$\delta/\delta_0$')

    ax[1].semilogy(angles_of_attack, dtst / dt0, 'k--')
    ax[1].semilogy(angles_of_attack, dtpt / dt0, 'k-')
    ax[1].set_xlim((0, 25))
    ax[1].set_ylim((.2, 200))
    ax[1].set_ylabel('$\delta^*/\delta^*_0$')

    ax[2].semilogy(angles_of_attack, mtst / mt0, 'k--')
    ax[2].semilogy(angles_of_attack, mtpt / mt0, 'k-')
    ax[2].set_xlim((0, 25))
    ax[2].set_ylim((.1, 50))
    ax[2].set_ylabel(r'$\theta/\theta_0$')
    ax[2].set_xlabel(r'$\alpha_*$')


def tbl_a_min(a):
    """
    Minimum Reynolds number spectrum A
    :param a: Absolute value of log10 the pressure side Strouhal ratio
    :return: A_min(a)
    """
    a = np.atleast_1d(a)
    a_min = np.empty_like(a)
    low = a < 0.204
    mid = np.logical_and(a >= 0.204, a <= 0.244)
    high = a > 0.244
    a_min[low] = np.sqrt(67.552 - 886.788 * a[low] ** 2) - 8.219
    a_min[mid] = -32.665 * a[mid] + 3.981
    a_min[high] = -142.795 * a[high] ** 3 + 103.656 * a[high] ** 2 - 57.757 * a[high] + 6.006
    return a_min


def tbl_a_max(a):
    """
    Maximum Reynolds number spectrum A
    :param a: Absolute value of log10 the pressure side Strouhal ratio
    :return: A_max(a)
    """
    a = np.atleast_1d(a)
    a_max = np.empty_like(a)
    low = a < 0.13
    mid = np.logical_and(0.13 <= a, a <= 0.321)
    high = a > 0.321
    a_max[low] = np.sqrt(67.552 - 886.788 * a[low] ** 2) - 8.219
    a_max[mid] = -15.901 * a[mid] + 1.098
    a_max[high] = -4.669 * a[high] ** 3 + 3.491 * a[high] ** 2 - 16.99 * a[high] + 1.149
    return a_max


def tbl_a0(chord_reynolds_number):
    """
    Scaled spectrum A horizontal intercept
    :param chord_reynolds_number: Reynolds number reference to section chord
    :return: a_0(Re_c)
    """
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    a0 = np.empty_like(chord_reynolds_number)
    low = chord_reynolds_number < 9.52e4
    mid = np.logical_and(9.52e4 <= chord_reynolds_number, chord_reynolds_number <= 8.57e5)
    high = chord_reynolds_number > 8.57e5
    a0[low] = 0.57
    a0[mid] = -9.57e-13 * (chord_reynolds_number[mid] - 8.57e5) ** 2 + 1.13
    a0[high] = 1.13
    return a0


def tbl_interpolating_function_a(a, chord_reynolds_number):
    """
     Interpolating function A
     :param a: Absolute value of log10 the pressure side Strouhal ratio
     :param chord_reynolds_number: Reynolds number reference to section chord
     :return: A(a)
     """
    a0 = tbl_a0(chord_reynolds_number)
    tbl_a_min_a0 = tbl_a_min(a0)
    ar = (-20 - tbl_a_min_a0) / (tbl_a_max(a0) - tbl_a_min_a0)
    tbl_a_min_a = tbl_a_min(a)
    return tbl_a_min_a + ar * (tbl_a_max(a) - tbl_a_min_a)


def tbl_b_min(b):
    """
    Minimum Reynolds number spectrum B
    :param b: Absolute value of log10 the suction side Strouhal ratio
    :return: B_min(b)
    """
    b = np.atleast_1d(b)
    b_min = np.empty_like(b)
    low = b < 0.13
    mid = np.logical_and(b >= 0.13, b <= 0.145)
    high = b > 0.145
    b_min[low] = np.sqrt(16.888 - 886.788 * b[low] ** 2) - 4.109
    b_min[mid] = -83.607 * b[mid] + 8.139
    b_min[high] = -817.81 * b[high] ** 3 + 355.210 * b[high] ** 2 - 135.024 * b[high] + 10.619
    return b_min


def tbl_b_max(b):
    """
    Maximum Reynolds number spectrum B
    :param b: Absolute value of log10 the suction side Strouhal ratio
    :return: B_max(b)
    """
    b = np.atleast_1d(b)
    b_max = np.empty_like(b)
    low = b < 0.1
    mid = np.logical_and(0.1 <= b, b <= 0.187)
    high = b > 0.187
    b_max[low] = np.sqrt(16.888 - 886.788 * b[low] ** 2) - 4.109
    b_max[mid] = -31.33 * b[mid] + 1.854
    b_max[high] = -80.541 * b[high] ** 3 + 44.174 * b[high] ** 2 - 39.38 * b[high] + 2.344
    return b_max


def tbl_b0(chord_reynolds_number):
    """
    Scaled spectrum B horizontal intercept
    :param chord_reynolds_number: Reynolds number reference to section chord
    :return: b_0(Re_c)
    """
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    b0 = np.empty_like(chord_reynolds_number)
    low = chord_reynolds_number < 9.52e4
    mid = np.logical_and(9.52e4 <= chord_reynolds_number, chord_reynolds_number <= 8.57e5)
    high = chord_reynolds_number > 8.57e5
    b0[low] = 0.3
    b0[mid] = -4.48e-13 * (chord_reynolds_number[mid] - 8.57e5) ** 2 + 0.56
    b0[high] = 0.56
    return b0


def tbl_interpolating_function_b(b, chord_reynolds_number):
    """
    Interpolating function B
    :param b: Absolute value of log10 the suction side Strouhal ratio
    :param chord_reynolds_number: Reynolds number reference to section chord
    :return: B(b)
    """
    b0 = tbl_b0(chord_reynolds_number)
    tbl_b_min_b0 = tbl_b_min(b0)
    br = (-20 - tbl_b_min_b0) / (tbl_b_max(b0) - tbl_b_min_b0)
    tbl_b_min_b = tbl_b_min(b)
    return tbl_b_min_b + br * (tbl_b_max(b) - tbl_b_min_b)


def tbl_amplitude_function_k1(chord_reynolds_number):
    """
    TBL amplitude function K1
    :param chord_reynolds_number: Reynolds number reference to section chord
    :return: K1
    """
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    k1 = np.empty_like(chord_reynolds_number)
    low = chord_reynolds_number < 2.47e5
    mid = np.logical_and(2.47e5 <= chord_reynolds_number, 8.0e5 >= chord_reynolds_number)
    high = chord_reynolds_number > 8.0e5
    k1[low] = -4.31 * np.log10(chord_reynolds_number[low]) + 156.3
    k1[mid] = -9.00 * np.log10(chord_reynolds_number[mid]) + 181.6
    k1[high] = 128.5
    return k1


def tbl_amplitude_function_delta_k1(alpha_e, displacement_thickness_pressure_reynolds_number):
    """
    Pressure side level adjustment for nonzero effective angles of attack, delta K1
    :param alpha_e: Effective angle of attack, deg
    :param displacement_thickness_pressure_reynolds_number: Reynolds number referenced to displacement thickness on the
    pressure side
    :return: delta K1
    """
    alpha_e = np.atleast_1d(alpha_e)
    displacement_thickness_pressure_reynolds_number = np.atleast_1d(displacement_thickness_pressure_reynolds_number)
    delta_k1 = alpha_e * (1.43 * np.log10(displacement_thickness_pressure_reynolds_number) - 5.29)
    delta_k1[displacement_thickness_pressure_reynolds_number < 5000] = 0.0
    return delta_k1


def tbl_gamma0(mach):
    """
    Peak of angle of attack for amplitude function K2
    :param mach: Mach number
    :return: Peak gamma_0, deg.
    """
    return 23.43 * mach + 4.651


def tbl_amplitude_function_k2(alpha_e, mach, chord_reynolds_number):
    """
    Angle of attack amplitude function K2
    :param alpha_e: Effective angle of attack, deg
    :param mach: Mach number
    :param chord_reynolds_number: Reynolds number for section chord
    :return: value of K2
    """
    alpha_e = np.atleast_1d(alpha_e)
    mach = np.atleast_1d(mach)

    gamma = 27.094 * mach + 3.31
    gamma0 = tbl_gamma0(mach)
    beta = 72.65 * mach + 10.74
    beta0 = -34.19 * mach - 13.82
    # Disable warnings for calculating K2 out of range
    np_error_settings = np.seterr()
    np.seterr(invalid='ignore')
    k2 = np.sqrt(beta ** 2 - (beta / gamma) ** 2 * (alpha_e - gamma0) ** 2) + beta0
    k2[alpha_e < gamma0 - gamma] = -1000
    k2[np.logical_or(alpha_e > (gamma0 + gamma), np.logical_and(alpha_e > gamma0, k2 < -12))] = -12
    np.seterr(**np_error_settings)
    k1 = tbl_amplitude_function_k1(chord_reynolds_number)
    k2 = k2 + k1
    return k2


def RP1218_TBL_plots():
    """
    Plot the interpolating functions A and B, and amplitude functions K1, K2 for comparison with RP1218 plots
    """
    # Figure 78 + moderate Reynolds value in red
    st_ratio = np.geomspace(.1, 20, 100)
    st = np.fabs(np.log10(st_ratio))

    a_max = tbl_a_max(st)
    a_min = tbl_a_min(st)
    a_mod = tbl_interpolating_function_a(st, chord_reynolds_number=.3e6)

    b_max = tbl_b_max(st)
    b_min = tbl_b_min(st)
    b_mod = tbl_interpolating_function_b(st, chord_reynolds_number=.3e6)

    _, ax = plt.subplots(nrows=2)
    ax[0].semilogx(st_ratio, a_min, 'k')
    ax[0].semilogx(st_ratio, a_max, 'k')
    ax[0].semilogx(st_ratio, a_mod, 'r')
    ax[0].set_xlim((.1, 20))
    ax[0].set_ylim((-20, 0))
    ax[0].set_ylabel('A')

    ax[1].semilogx(st_ratio, b_min, 'k')
    ax[1].semilogx(st_ratio, b_max, 'k')
    ax[1].semilogx(st_ratio, b_mod, 'r')
    ax[1].set_xlim((.1, 20))
    ax[1].set_ylim((-20, 0))
    ax[1].set_ylabel('B')
    ax[1].set_xlabel('Strouhal Number Ratio')

    # Figure 77
    re = np.geomspace(1e4, 1e7, 100)
    k1 = tbl_amplitude_function_k1(re)
    _, ax = plt.subplots()
    ax.semilogx(re, k1)
    ax.set_xlim((1e4, 1e7))
    ax.set_ylim((110, 150))
    ax.set_xlabel('Reynolds Number, $R_c$')
    ax.set_ylabel('$K_1$')

    # Figure 82
    alpha_e = np.linspace(0, 25, 100)
    re = 1e6
    k1 = tbl_amplitude_function_k1(re)
    k2m1 = tbl_amplitude_function_k2(alpha_e, mach=0.093, chord_reynolds_number=re) - k1
    k2m2 = tbl_amplitude_function_k2(alpha_e, mach=0.116, chord_reynolds_number=re) - k1
    k2m3 = tbl_amplitude_function_k2(alpha_e, mach=0.163, chord_reynolds_number=re) - k1
    k2m4 = tbl_amplitude_function_k2(alpha_e, mach=0.209, chord_reynolds_number=re) - k1
    _, ax = plt.subplots()
    ax.plot(alpha_e, k2m1, alpha_e, k2m2, alpha_e, k2m3, alpha_e, k2m4)
    ax.set_xlim((0, 25))
    ax.set_ylim((-20, 20))
    ax.set_ylabel('$K_2 - K_1$')
    ax.set_xlabel(r'$\alpha_*$')


def strouhal_peaks(alpha_e, mach):
    """
    Compute the Strouhal number for the peaks of the scaled spectra A, B
    :param alpha_e: Effective angle of attack, deg
    :param mach: Mach number
    :return: St_1 and St_2
    """
    alpha_e = np.atleast_1d(alpha_e)
    mach = np.atleast_1d(mach)
    strouhal_1 = 0.02 * mach ** (-0.6)
    strouhal_2 = strouhal_1 * 10.0 ** (0.0054 * (alpha_e - 1.33) ** 2)
    strouhal_2[alpha_e < 1.33] = 1.0
    strouhal_2[alpha_e > 12.5] = 4.72
    return strouhal_1, strouhal_2


def single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped=False, direct=False,
                               theta=0.0, phi=0.0, speed_of_sound=340.3):
    """
    Calculate turbulent boundary layer (TBL) Sound Pressure Level (SPL) for 1/3 octave center frequency f
    :param f: 1/3 octave center frequency, Hz
    :param chord: Section chord, m
    :param span: Section span, m
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :return: Sound pressure level for 1/3 octave band, dB
    """
    bts, dts, mts, btp, dtp, mtp, _, _, _ = boundary_layer_model(alpha_e, chord_reynolds_number, tripped)

    chord = np.atleast_1d(chord)
    span = np.atleast_1d(span)

    alpha_e = np.atleast_1d(alpha_e)
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    speed = np.atleast_1d(speed)
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    strouhal_pressure = f * dtp * chord / speed
    strouhal_suction = f * dts * chord / speed
    mach = speed / speed_of_sound

    if direct:
        dh = directivity_function_high(theta, phi, mach)
        dl = directivity_function_low(theta, phi, mach)
    else:
        dh = np.ones_like(mach)
        dl = np.ones_like(mach)
    strouhal_1, strouhal_2 = strouhal_peaks(alpha_e, mach)
    displacement_thickness_pressure_reynolds_number = chord_reynolds_number * dtp

    str_p_1 = np.fabs(np.log10(strouhal_pressure / strouhal_1))
    spl_pressure_side = 10.0 * np.log10(
        dtp * chord * mach ** 5 * span * dh / (r ** 2)) + tbl_interpolating_function_a(
        str_p_1, chord_reynolds_number) + tbl_amplitude_function_k1(
        chord_reynolds_number) - 3.0 + tbl_amplitude_function_delta_k1(alpha_e,
                                                                       displacement_thickness_pressure_reynolds_number)
    str_s_1 = np.fabs(np.log10(strouhal_suction / strouhal_1))
    spl_suction_side = 10.0 * np.log10(
        dts * chord * mach ** 5 * span * dh / (r ** 2)) + tbl_interpolating_function_a(str_s_1,
                                                                                       chord_reynolds_number) + \
                       tbl_amplitude_function_k1(chord_reynolds_number) - 3.0

    str_s_2 = np.fabs(np.log10(strouhal_suction / strouhal_2))

    spl_alpha = 10.0 * np.log10(
        dts * chord * mach ** 5 * span * dh / (r ** 2)) + tbl_interpolating_function_b(str_s_2,
                                                                                       chord_reynolds_number) + \
                tbl_amplitude_function_k2(alpha_e, mach, chord_reynolds_number)

    # If angle of attack exceeds a threshhold, use separation noise model
    alpha_e_0 = np.where(tbl_gamma0(mach) < 12.5, tbl_gamma0(mach), 12.5)

    spl_pressure_side[alpha_e > alpha_e_0] = -np.Inf
    spl_suction_side[alpha_e > alpha_e_0] = -np.Inf
    # Expand r, span, and chord, if necessary, to allow alpha indexing
    if len(span) == 1:
        span = np.full_like(alpha_e, span[0])
    if len(chord) == 1:
        chord = np.full_like(alpha_e, chord[0])
    if len(r) == 1:
        r = np.full_like(alpha_e, r[0])
    spl_alpha[alpha_e > alpha_e_0] = 10.0 * np.log10(
        dts[alpha_e > alpha_e_0] * chord[alpha_e > alpha_e_0] * mach[alpha_e > alpha_e_0] ** 5 * span[
            alpha_e > alpha_e_0] * dl[alpha_e > alpha_e_0] / (
                r[alpha_e > alpha_e_0] ** 2)) + tbl_interpolating_function_a(str_s_2[alpha_e > alpha_e_0],
                                                                             3.0 * chord_reynolds_number[
                                                                                 alpha_e > alpha_e_0]) + \
                                     tbl_amplitude_function_k2(alpha_e[alpha_e > alpha_e_0], mach[alpha_e > alpha_e_0],
                                                               chord_reynolds_number[alpha_e > alpha_e_0])

    return 10.0 * np.log10(
        10.0 ** (0.1 * spl_pressure_side) + 10.0 ** (0.1 * spl_suction_side) + 10.0 ** (0.1 * spl_alpha))


def tbl_noise_spectra(chord, span, alpha_e, chord_reynolds_number, speed, r, tripped=False, direct=False, theta=0.0,
                      phi=0.0, speed_of_sound=340.3, frequencies=exact_center_frequencies):
    """
    Calculate turbulent boundary layer (TBL) Sound Pressure Level (SPL) for 1/3 octave spectra
    :param chord: Section chord, m
    :param span: Section span, m
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :param frequencies: 1/3 Octave center frequencies, Hz.  Default exact center frequencies from 20Hz -20kHz
    :return: Sound pressure levels for 1/3 octave band, dB, frequencies x panels
    """
    spectra = []
    for f in frequencies:
        L = single_frequency_tbl_noise(f=f, chord=chord, span=span, alpha_e=alpha_e,
                                       chord_reynolds_number=chord_reynolds_number, speed=speed, r=r, tripped=tripped,
                                       direct=direct, theta=theta, phi=phi, speed_of_sound=speed_of_sound)
        spectra.append(L)
    return np.array(spectra)


def tbl_noise_spectrum(chord, span, alpha_e, chord_reynolds_number, speed, r, tripped=False, direct=False, theta=0.0,
                       phi=0.0, speed_of_sound=340.3, frequencies=exact_center_frequencies):
    """
    Calculate turbulent boundary layer (TBL) Sound Pressure Level (SPL) for 1/3 octave spectra
    :param chord: Section chord, m
    :param span: Section span, m
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :param frequencies: 1/3 Octave center frequencies, Hz.  Default exact center frequencies from 20Hz -20kHz
    :return: Total sound pressure level for each 1/3 octave band, dB
    """
    return dbsum(tbl_noise_spectra(chord=chord, span=span, alpha_e=alpha_e, chord_reynolds_number=chord_reynolds_number,
                                   speed=speed, r=r, tripped=tripped, direct=direct, theta=theta, phi=phi,
                                   speed_of_sound=speed_of_sound, frequencies=frequencies), 1)


def RP1218_TBL_predict_plots():
    """
    Calculate TBL noise and plot against several test cases from RP1218
    """
    # Figure 11
    chord = 30.48 / 100.0
    span = 45.72 / 100.0
    alpha_e = 0.0
    theta = 90.0
    phi = 90.0
    r = 1.22
    viscosity = 1.460e-5
    tripped = True
    direct = True

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Figure 11')

    speed = 71.3
    chord_reynolds_number = speed * chord / viscosity
    spla = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 55.5
    chord_reynolds_number = speed * chord / viscosity
    splb = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 39.6
    chord_reynolds_number = speed * chord / viscosity
    splc = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 31.7
    chord_reynolds_number = speed * chord / viscosity
    spld = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])

    ax[0, 0].semilogx(exact_center_frequencies, spla)
    ax[0, 0].set_ylim((40, 80))
    ax[0, 0].set_xlim((.2e3, 20e3))
    ax[0, 0].set_title('U = 71.3 m/s')
    ax[0, 0].set_ylabel('SPL 1/3, dB')
    ax[0, 1].semilogx(exact_center_frequencies, splb)
    ax[0, 1].set_ylim((30, 70))
    ax[0, 1].set_xlim((.2e3, 20e3))
    ax[0, 1].set_title('U = 55.5 m/s')
    ax[1, 0].semilogx(exact_center_frequencies, splc)
    ax[1, 0].set_ylim((20, 60))
    ax[1, 0].set_xlim((.2e3, 20e3))
    ax[1, 0].set_title('U = 39.6 m/s')
    ax[1, 0].set_xlabel('Frequency, Hz')
    ax[1, 0].set_ylabel('SPL 1/3, dB')
    ax[1, 1].semilogx(exact_center_frequencies, spld)
    ax[1, 1].set_ylim((20, 60))
    ax[1, 1].set_xlim((.2e3, 20e3))
    ax[1, 1].set_title('U = 31.7 m/s')
    ax[1, 1].set_xlabel('Frequency, Hz')

    plt.tight_layout()

    # Figure 13
    alpha_e = 3.0

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Figure 13')

    speed = 71.3
    chord_reynolds_number = speed * chord / viscosity
    spla = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 55.5
    chord_reynolds_number = speed * chord / viscosity
    splb = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 39.6
    chord_reynolds_number = speed * chord / viscosity
    splc = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 31.7
    chord_reynolds_number = speed * chord / viscosity
    spld = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])

    ax[0, 0].semilogx(exact_center_frequencies, spla)
    ax[0, 0].set_ylim((40, 80))
    ax[0, 0].set_xlim((.2e3, 20e3))
    ax[0, 0].set_title('U = 71.3 m/s')
    ax[0, 0].set_ylabel('SPL 1/3, dB')
    ax[0, 1].semilogx(exact_center_frequencies, splb)
    ax[0, 1].set_ylim((30, 70))
    ax[0, 1].set_xlim((.2e3, 20e3))
    ax[0, 1].set_title('U = 55.5 m/s')
    ax[1, 0].semilogx(exact_center_frequencies, splc)
    ax[1, 0].set_ylim((20, 60))
    ax[1, 0].set_xlim((.2e3, 20e3))
    ax[1, 0].set_title('U = 39.6 m/s')
    ax[1, 0].set_xlabel('Frequency, Hz')
    ax[1, 0].set_ylabel('SPL 1/3, dB')
    ax[1, 1].semilogx(exact_center_frequencies, spld)
    ax[1, 1].set_ylim((20, 60))
    ax[1, 1].set_xlim((.2e3, 20e3))
    ax[1, 1].set_title('U = 31.7 m/s')
    ax[1, 1].set_xlabel('Frequency, Hz')

    plt.tight_layout()

    # Figure 19
    chord = 22.86 / 100.0
    alpha_e = 7.3

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Figure 19')

    speed = 71.3
    chord_reynolds_number = speed * chord / viscosity
    spla = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 55.5
    chord_reynolds_number = speed * chord / viscosity
    splb = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 39.6
    chord_reynolds_number = speed * chord / viscosity
    splc = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])
    speed = 31.7
    chord_reynolds_number = speed * chord / viscosity
    spld = np.array([single_frequency_tbl_noise(f, chord, span, alpha_e, chord_reynolds_number, speed, r, tripped,
                                                direct, theta, phi) for f in
                     exact_center_frequencies])

    ax[0, 0].semilogx(exact_center_frequencies, spla)
    ax[0, 0].set_ylim((50, 90))
    ax[0, 0].set_xlim((.2e3, 20e3))
    ax[0, 0].set_title('U = 71.3 m/s')
    ax[0, 0].set_ylabel('SPL 1/3, dB')
    ax[0, 1].semilogx(exact_center_frequencies, splb)
    ax[0, 1].set_ylim((40, 80))
    ax[0, 1].set_xlim((.2e3, 20e3))
    ax[0, 1].set_title('U = 55.5 m/s')
    ax[1, 0].semilogx(exact_center_frequencies, splc)
    ax[1, 0].set_ylim((30, 70))
    ax[1, 0].set_xlim((.2e3, 20e3))
    ax[1, 0].set_title('U = 39.6 m/s')
    ax[1, 0].set_xlabel('Frequency, Hz')
    ax[1, 0].set_ylabel('SPL 1/3, dB')
    ax[1, 1].semilogx(exact_center_frequencies, spld)
    ax[1, 1].set_ylim((20, 60))
    ax[1, 1].set_xlim((.2e3, 20e3))
    ax[1, 1].set_title('U = 31.7 m/s')
    ax[1, 1].set_xlabel('Frequency, Hz')

    plt.tight_layout()


def single_frequency_vortex_noise(f, chord, alpha_tip, speed, r, round_tip=False, slope_correction=1.0, direct=False,
                                  theta=0.0, phi=0.0, speed_of_sound=340.3):
    """
      Calculate tip vortex formation Sound Pressure Level (SPL) for 1/3 octave center frequency f
      :param f: 1/3 octave center frequency, Hz
      :param chord: Section chord, m
      :param alpha_tip: Effective angle of attack at the tip, deg
      :param speed: Flow speed, m/s
      :param r: Propagation distance, m
      :param round_tip: Set to true for rounded tip, false for square tip.  Default false
      :param slope_correction: Correction factor for spanwise tip loading slope.  Default 1.
      :param direct: Set to true to include directivity function effects. Default false
      :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
      :param phi: Angle from source lateral axis to observer, deg. Default 0.0
      :param speed_of_sound: Speed of sound, m/s.  Default 340.3
      :return: Sound pressure level for 1/3 octave band, dB
      """
    alpha_tip = slope_correction * np.fabs(alpha_tip)

    speed_max = (1.0 + 0.036 * alpha_tip) * speed
    mach = speed / speed_of_sound
    mach_max = speed_max / speed_of_sound

    if round_tip:
        vortex_span = 0.008 * alpha_tip * chord
    elif alpha_tip < 2.0:
        vortex_span = (0.023 + 0.0169 * alpha_tip) * chord
    else:
        vortex_span = (0.0378 + 0.0095 * alpha_tip) * chord
    strouhal_vortex = f * vortex_span / speed_max
    if direct:
        dh = directivity_function_high(theta, phi, mach)
    else:
        dh = np.ones_like(speed)

    return 10.0 * np.log10(mach ** 2 * mach_max ** 3 * vortex_span ** 2 * dh / (r ** 2)) - 30.5 * (
            np.log10(strouhal_vortex) + 0.3) ** 2 + 126


def vortex_noise_spectrum(chord, alpha_tip, speed, r, round_tip=False, slope_correction=1.0, direct=False, theta=0.0,
                          phi=0.0, speed_of_sound=340.3, frequencies=exact_center_frequencies):
    """
      Calculate tip vortex formation Sound Pressure Level (SPL) spectrum for 1/3 octave bands
      :param chord: Section chord, m
      :param alpha_tip: Effective angle of attack at the tip, deg
      :param speed: Flow speed, m/s
      :param r: Propagation distance, m
      :param round_tip: Set to true for rounded tip, false for square tip.  Default false
      :param slope_correction: Correction factor for spanwise tip loading slope.  Default 1.
      :param direct: Set to true to include directivity function effects. Default false
      :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
      :param phi: Angle from source lateral axis to observer, deg. Default 0.0
      :param speed_of_sound: Speed of sound, m/s.  Default 340.3
      :param frequencies: 1/3 octave center frequencies, Hz.  Default exact center frequencies from 20Hz to 20 kHz
      :return: 1/3 octave spectrum, dB.
      """
    spectrum = []
    for f in frequencies:
        L = single_frequency_vortex_noise(f=f, chord=chord, alpha_tip=alpha_tip, speed=speed, r=r, round_tip=round_tip,
                                          slope_correction=slope_correction, direct=direct, theta=theta, phi=phi,
                                          speed_of_sound=speed_of_sound)
        spectrum.append(L)
    return np.array(spectrum)


def RP1218_tip_vortex_plot():
    """
    Calculate and plot tip vortex noise test case from RP1218/
    """
    # Figure 91
    chord = 15.24 / 100.0
    alpha_tip = 8.4  # Effective, but not listed in figure caption
    theta = 90.0
    phi = 90.0
    r = 1.22
    round_tip = True
    direct = True

    fig, ax = plt.subplots()
    plt.suptitle('Figure 91')

    speed = 71.3

    spl = np.array(
        [single_frequency_vortex_noise(f, chord, alpha_tip, speed, r, round_tip, 1.0, direct, theta, phi) for f in
         exact_center_frequencies])
    ax.semilogx(exact_center_frequencies, spl)
    ax.set_ylim((40, 90))
    ax.set_xlim((.2e3, 20e3))
    ax.set_title('U = 71.3 m/s')
    ax.set_ylabel('SPL 1/3, dB')


def teb_g4(bluntness_ratio, solid_angle):
    """
    Trailing edge bluntness function G4
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :return: Value of scaling function, G4
    """

    solid_angle = solid_angle * np.ones_like(bluntness_ratio)
    g4 = 169.7 - 1.114 * solid_angle
    mask = bluntness_ratio <= 5.0
    g4[mask] = 17.5 * np.log10(bluntness_ratio[mask]) + 157.5 - 1.114 * solid_angle[mask]
    return g4


def teb_g5_mu(bluntness_ratio):
    """
    Parameter mu in function G5
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :return:  mu
    """
    low = np.logical_and(bluntness_ratio >= 0.25, bluntness_ratio < 0.62)
    med = np.logical_and(bluntness_ratio >= 0.62, bluntness_ratio < 1.15)
    high = bluntness_ratio >= 1.15
    mu = np.full_like(bluntness_ratio, 0.1221)
    mu[low] = -0.2175 * bluntness_ratio[low] + 0.1755
    mu[med] = -0.0308 * bluntness_ratio[med] + 0.0596
    mu[high] = 0.0242
    return mu


def teb_g5_m(bluntness_ratio):
    """
    Parameter m in function G5
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :return:  m
    """
    m = np.zeros_like(bluntness_ratio)
    lower = np.logical_and(0.02 < bluntness_ratio, bluntness_ratio <= 0.5)
    low = np.logical_and(0.5 < bluntness_ratio, bluntness_ratio <= 0.62)
    med = np.logical_and(0.62 < bluntness_ratio, bluntness_ratio <= 1.15)
    high = np.logical_and(1.15 < bluntness_ratio, bluntness_ratio <= 1.2)
    higher = bluntness_ratio > 1.2
    m[lower] = 68.724 * bluntness_ratio[lower] - 1.35
    m[low] = 308.475 * bluntness_ratio[low] - 121.23
    m[med] = 224.811 * bluntness_ratio[med] - 69.35
    m[high] = 1583.28 * bluntness_ratio[high] - 1631.59
    m[higher] = 268.344
    return m


def teb_g5_14(bluntness_ratio, strouhal_ratio):
    """
    Trailing edge bluntness function G5 for NACA 0012 (solid_angle = 14)
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :param strouhal_ratio: ratio between TEB Strouhal number and peak Strouhal for TEB scaled spectrum
    :return: Value of scaling function, G5_14
    """

    m = teb_g5_m(bluntness_ratio)
    mu = teb_g5_mu(bluntness_ratio)
    eta = np.log10(strouhal_ratio)
    eta0 = -np.sqrt(m ** 2 * mu ** 4 / (6.25 + m ** 2 * mu ** 2))
    k = 2.5 * np.sqrt(1.0 - (eta0 / mu) ** 2) - 2.5 - m * eta0

    g5_14 = np.full_like(bluntness_ratio, m * eta + k)
    low = np.logical_and(eta >= eta0, eta < 0.0)
    med = np.logical_and(eta >= 0.0, eta < 0.03616)
    high = eta >= 0.03616

    g5_14[low] = 2.5 * np.sqrt(1.0 - (eta[low] / mu[low]) ** 2) - 2.5
    g5_14[med] = np.sqrt(1.5625 - 1194.99 * eta[med] ** 2) - 1.25
    g5_14[high] = -155.543 * eta[high] + 4.375

    return g5_14


def teb_g5_0(bluntness_ratio, strouhal_ratio):
    """
    Trailing edge bluntness function G5 for flat plate
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :param strouhal_ratio: ratio between TEB Strouhal number and peak Strouhal for TEB scaled spectrum
    :return: Value of scaling function, G5_0
    """
    bluntness_ratio_0 = 6.724 * bluntness_ratio ** 2 - 4.019 * bluntness_ratio + 1.107
    return teb_g5_14(bluntness_ratio_0, strouhal_ratio)


def teb_g5(bluntness_ratio, solid_angle, strouhal_ratio):
    """
    Trailing edge bluntness function G5
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :param strouhal_ratio: ratio between TEB Strouhal number and peak Strouhal for TEB scaled spectrum
    :return: Value of scaling function, G5
    """

    teb_g5_zero = teb_g5_0(bluntness_ratio, strouhal_ratio)
    return teb_g5_zero + 0.0714 * solid_angle * (
            teb_g5_14(bluntness_ratio, strouhal_ratio) - teb_g5_zero)


def teb_strouhal_peak(bluntness_ratio, solid_angle):
    """
    Trailing edge bluntness spectrum Strouhal peak
    :param bluntness_ratio: ratio of trailing edge thickness to average boundary-layer displacement thickness
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :return: Strouhal of scaled spectrum peak
    """
    solid_angle = solid_angle * np.ones_like(bluntness_ratio)
    strouhal_peak = 0.1 * bluntness_ratio + 0.095 - 0.00243 * solid_angle
    mask = bluntness_ratio >= 0.2
    strouhal_peak[mask] = (0.212 - 0.0045 * solid_angle[mask]) / (
            1.0 + 0.235 / bluntness_ratio[mask] - 0.0132 / (bluntness_ratio[mask] ** 2))
    return strouhal_peak


def single_frequency_bluntness_noise(f, chord, span, trailing_edge_thickness, solid_angle, alpha_e,
                                     chord_reynolds_number, speed, r, tripped=False, direct=False, theta=0.0, phi=0.0,
                                     speed_of_sound=340.3):
    """
    Calculate trailing edge bluntness (TEB) Sound Pressure Level (SPL) for 1/3 octave center frequency f
    param f: 1/3 octave center frequency, Hz
    :param chord: Section chord, m
    :param span: Section span, m
    :param trailing_edge_thickness: trailing edge thickness, m
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :return: Sound pressure level for 1/3 octave band, dB
    """
    # Convert scalars to arrays
    span = np.atleast_1d(span)
    trailing_edge_thickness = np.atleast_1d(trailing_edge_thickness)
    # Set minimum TE thickness
    trailing_edge_thickness[trailing_edge_thickness < 1.0e-6] = 1e-6
    solid_angle = np.atleast_1d(solid_angle)
    alpha_e = np.atleast_1d(alpha_e)
    chord_reynolds_number = np.atleast_1d(chord_reynolds_number)
    speed = np.atleast_1d(speed)
    r = np.atleast_1d(r)
    # Calculate average displacement thickness
    _, dts, _, _, dtp, _, _, _, _ = boundary_layer_model(alpha_e, chord_reynolds_number, tripped)
    displacement_thickness = 0.5 * (dts + dtp) * chord
    # Compute ratios
    bluntness_ratio = trailing_edge_thickness / displacement_thickness
    strouhal = f * trailing_edge_thickness / speed
    strouhal_peak = teb_strouhal_peak(bluntness_ratio, solid_angle)
    strouhal_ratio = strouhal / strouhal_peak
    mach = speed / speed_of_sound
    if direct:
        df = directivity_function_high(theta, phi, mach)
    else:
        df = 1.0
    # Add scaled source contributions
    return 10.0 * np.log10(trailing_edge_thickness * mach ** 5.5 * span * df / (r ** 2)) + teb_g4(bluntness_ratio,
                                                                                                  solid_angle) + teb_g5(
        bluntness_ratio, solid_angle, strouhal_ratio)


def bluntness_noise_spectra(chord, span, trailing_edge_thickness, solid_angle, alpha_e, chord_reynolds_number, speed, r,
                            tripped=False, direct=False, theta=0.0, phi=0.0, speed_of_sound=340.3,
                            frequencies=exact_center_frequencies):
    """
    Calculate trailing edge bluntness (TEB) Sound Pressure Level (SPL) for 1/3 octave spectra
    :param chord: Section chord, m
    :param span: Section span, m
    :param trailing_edge_thickness: trailing edge thickness, m
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :param frequencies: 1/3 Octave center frequencies, Hz.  Default exact center frequencies from 20Hz -20kHz
    :return: Sound pressure levels for 1/3 octave band, dB, frequencies x panels
    """
    spectra = []
    for f in frequencies:
        L = single_frequency_bluntness_noise(f=f, chord=chord, span=span,
                                             trailing_edge_thickness=trailing_edge_thickness, solid_angle=solid_angle,
                                             alpha_e=alpha_e, chord_reynolds_number=chord_reynolds_number, speed=speed,
                                             r=r, tripped=tripped, direct=direct, theta=theta, phi=phi,
                                             speed_of_sound=speed_of_sound)
        spectra.append(L)
    return np.array(spectra)


def bluntness_noise_spectrum(chord, span, trailing_edge_thickness, solid_angle, alpha_e, chord_reynolds_number, speed,
                             r, tripped=False, direct=False, theta=0.0, phi=0.0, speed_of_sound=340.3,
                             frequencies=exact_center_frequencies):
    """
    Calculate trailing edge bluntness (TEB) Sound Pressure Level (SPL) for 1/3 octave spectra
    :param chord: Section chord, m
    :param span: Section span, m
    :param trailing_edge_thickness: trailing edge thickness, m
    :param solid_angle: solid angle between sloping surfaces upstream of trailing edge, deg
    :param alpha_e: Effective angle of attack, deg
    :param chord_reynolds_number: Reynolds number referenced to section chord
    :param speed: Flow speed, m/s
    :param r: Propagation distance, m
    :param tripped: Set to true for tripped flow.  Default false
    :param direct: Set to true to include directivity function effects. Default false
    :param theta: Angle from source streamwise axis to observer, deg.  Default 0.0
    :param phi: Angle from source lateral axis to observer, deg. Default 0.0
    :param speed_of_sound: Speed of sound, m/s.  Default 340.3
    :param frequencies: 1/3 Octave center frequencies, Hz.  Default exact center frequencies from 20Hz -20kHz
    :return: Total sound pressure level for each 1/3 octave band, dB
    """
    return dbsum(
        bluntness_noise_spectra(chord=chord, span=span, trailing_edge_thickness=trailing_edge_thickness,
                                solid_angle=solid_angle,
                                alpha_e=alpha_e, chord_reynolds_number=chord_reynolds_number, speed=speed, r=r,
                                tripped=tripped, direct=direct, theta=theta, phi=phi, speed_of_sound=speed_of_sound,
                                frequencies=frequencies), 1)


def RP1218_TEB_plots():
    """
    Replicate some plots from RP1218 for trailing edge bluntness
    """
    # Figure 97
    bluntness_ratios = np.array([.25, .43, .5, .54, .62, 1.2])
    strouhal_ratios = np.geomspace(.1, 10, 100)

    fig, ax = plt.subplots(nrows=2)
    fig.suptitle('Figure 97')

    for bluntness_ratio in bluntness_ratios:
        bluntness_ratio = np.atleast_1d(bluntness_ratio)
        tebg514 = list(map(lambda sr: teb_g5(bluntness_ratio, solid_angle=14.0, strouhal_ratio=np.atleast_1d(sr)),
                           strouhal_ratios))
        tebg500 = list(
            map(lambda sr: teb_g5(bluntness_ratio, solid_angle=0.0, strouhal_ratio=np.atleast_1d(sr)), strouhal_ratios))
        ax[0].semilogx(strouhal_ratios, tebg514, label=str(bluntness_ratio))
        ax[1].semilogx(strouhal_ratios, tebg500, label=str(bluntness_ratio))

    ax[0].set_xlim((.1, 10))
    ax[0].set_xlabel('Strouhal ratio, $St/St_{peak}$')
    ax[0].set_ylim((-30, 10))
    ax[0].set_ylabel('$(G_5)_{\Psi=14}$')
    ax[1].set_xlim((.1, 10))
    ax[1].set_xlabel('Strouhal ratio, $St/St_{peak}$')
    ax[1].set_ylim((-30, 10))
    ax[1].set_ylabel('$(G_5)_{\Psi=0}$')

    # Figure 96
    fig, ax = plt.subplots()
    fig.suptitle('Figure 96')
    bluntness_ratios = np.geomspace(0.1, 10, 50)
    g414 = teb_g4(bluntness_ratios, np.atleast_1d(14.0))
    g400 = teb_g4(bluntness_ratios, np.atleast_1d(0.0))
    ax.semilogx(bluntness_ratios, g414, bluntness_ratios, g400)
    ax.set_xlim((0.1, 10))
    ax.set_ylim((110, 180))
    ax.set_xlabel('Bluntness ratio, $h/\delta^*$')
    ax.set_ylabel('Scaled peak SPL, dB')
    plt.tight_layout()

    # Figure 95
    fig, ax = plt.subplots()
    fig.suptitle('Figure 95')
    bluntness_ratios = np.geomspace(0.2, 10, 50)
    stp14 = teb_strouhal_peak(bluntness_ratios, 14.0)
    stp00 = teb_strouhal_peak(bluntness_ratios, 0.0)
    ax.loglog(bluntness_ratios, stp14, bluntness_ratios, stp00)
    ax.set_xlim((0.2, 10))
    ax.set_ylim((.05, .3))
    ax.set_xlabel('Bluntness ratio, $h/\delta^*$')
    ax.set_ylabel('Peak Stouhal number, $St_{peak}$')
    plt.tight_layout()

    # Figure 98
    chord = 60.96 / 100.0
    span = 45.72 / 100.0
    alpha_e = 0.0
    theta = 90.0
    phi = 90.0
    r = 1.22
    viscosity = 1.460e-5
    speed = 69.5
    tripped = True
    direct = True
    solid_angle = 14.0
    chord_reynolds_number = chord * speed / viscosity

    bnh00 = bluntness_noise_spectra(chord, span, 0.0, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh11 = bluntness_noise_spectra(chord, span, 1.1e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh19 = bluntness_noise_spectra(chord, span, 1.9e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh25 = bluntness_noise_spectra(chord, span, 2.5e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Figure 98')
    ax[0, 0].semilogx(exact_center_frequencies, bnh00)
    ax[0, 0].set_xlim((.2e3, 20e3))
    ax[0, 0].set_ylim((40, 80))
    ax[0, 1].semilogx(exact_center_frequencies, bnh11)
    ax[0, 1].set_xlim((.2e3, 20e3))
    ax[0, 1].set_ylim((40, 80))
    ax[1, 0].semilogx(exact_center_frequencies, bnh19)
    ax[1, 0].set_xlim((.2e3, 20e3))
    ax[1, 0].set_ylim((40, 80))
    ax[1, 1].semilogx(exact_center_frequencies, bnh25)
    ax[1, 1].set_xlim((.2e3, 20e3))
    ax[1, 1].set_ylim((40, 80))

    # Figure 99
    speed = 38.6
    chord_reynolds_number = chord * speed / viscosity
    bnh00 = bluntness_noise_spectra(chord, span, 0.0, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh11 = bluntness_noise_spectra(chord, span, 1.1e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh19 = bluntness_noise_spectra(chord, span, 1.9e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    bnh25 = bluntness_noise_spectra(chord, span, 2.5e-3, solid_angle, alpha_e, chord_reynolds_number, speed, r, tripped,
                                    direct, theta, phi)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Figure 99')
    ax[0, 0].semilogx(exact_center_frequencies, bnh00)
    ax[0, 0].set_xlim((.2e3, 20e3))
    ax[0, 0].set_ylim((30, 70))
    ax[0, 1].semilogx(exact_center_frequencies, bnh11)
    ax[0, 1].set_xlim((.2e3, 20e3))
    ax[0, 1].set_ylim((30, 70))
    ax[1, 0].semilogx(exact_center_frequencies, bnh19)
    ax[1, 0].set_xlim((.2e3, 20e3))
    ax[1, 0].set_ylim((30, 70))
    ax[1, 1].semilogx(exact_center_frequencies, bnh25)
    ax[1, 1].set_xlim((.2e3, 20e3))
    ax[1, 1].set_ylim((30, 70))


if __name__ == "__main__":
    RP1218_BL_plots()
    RP1218_TBL_plots()
    RP1218_TBL_predict_plots()
    RP1218_tip_vortex_plot()
    RP1218_TEB_plots()
    plt.show(block=True)
