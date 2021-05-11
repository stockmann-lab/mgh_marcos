#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))
sys.path.append(os.path.abspath("../config"))
import scanner_config as sc # pylint: disable=import-error

from examples import turbo_spin_echo # pylint: disable=import-error

def grad_max_cal(phantom_width=10, larmor_freq=sc.LARMOR_FREQ, calibration_power=0.8,
                 trs=2, tr_spacing=2e6, echo_duration=5000, 
                 readout_duration=500, rx_period=25/3, gradient_overshoot=100,
                 rf_pi2_duration=50, rf_max=sc.RF_MAX,
                 plot=True):
    """
    Calibrate gradient maximum using a phantom of known width

    Args:
        phantom_width (float): [mm] Phantom width 
        larmor_freq (float): [MHz] Scanner larmor frequency
        calibration_power (float): [arb.] Fractional power to evaluate at
        trs (int): [arb.] Number of times to repeat for averaging
        tr_spacing (float): [us] Time between repetitions
        echo_duration (float): [us] Time between echo peaks
        readout_duration (float): [us] Readout window around echo peak
        rx_period (float): [us] Readout dwell time
        gradient_overshoot (float): [us] Amount of time to hold the readout gradient on for longer than readout_duration
        rf_pi2_duration (float): [us] RF pi/2 pulse duration
        rf_max (float): [Hz] System RF max
        plot (bool): Default True, plot final data

    Returns:
        float: Estimated gradient max in Hz/m
    """

    rf_scaling = .25 / (rf_max * 1e-6 * rf_pi2_duration)

    # Run the experiment, hardcoded from marcos_client/examples
    rxd = turbo_spin_echo(lo_freq=larmor_freq,
        trs=trs, echos_per_tr=1,
        rf_amp=rf_scaling,
        rx_period=rx_period,
        echo_duration=echo_duration,
        readout_duration=readout_duration,
        readout_amp=calibration_power,
        readout_grad_duration=readout_duration + gradient_overshoot,
        phase_start_amp=0,
        slice_start_amp=0,
        tr_pause_duration=tr_spacing,
        init_gpa=True, plot_rx=False)

    rx_arr = np.reshape(rxd, (trs, -1))
    rx_arr_av = np.average(rx_arr, axis=0)
    rxd_av = np.reshape(rx_arr_av, (-1))

    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rxd_av)))

    peaks, _ = sig.find_peaks(np.abs(rx_fft), width=2)
    peak_results = sig.peak_widths(np.abs(rx_fft), peaks, rel_height=0.95)
    fwhm = peak_results[0][0]

    fft_scale = 1e6/(rx_period * rx_fft.shape[0]) # [Hz/index] Need to know what the distance on x is

    fft_bw = 1/(rx_period)
    fft_x = np.linspace(larmor_freq - fft_bw/2, larmor_freq + fft_bw/2, num=rx_fft.shape[0])

    grad_max = fwhm * fft_scale / (phantom_width * 1e-3 * calibration_power) # [Hz/m]
    print(f'Gradient value: {(grad_max * calibration_power * 1e-3):.4f} KHz/m')
    print(f'Estimated gradient max: {(grad_max*1e-3):.4f} KHz/m')

    if plot:
        _, axs = plt.subplots(4, 1, constrained_layout=True)
        axs[0].plot(np.real(rxd))
        axs[0].set_title('Concatenated signal -- Real')
        axs[1].plot(np.abs(rxd))
        axs[1].set_title('Concatenated signal -- Magnitude')
        axs[2].plot(np.abs(rxd_av))
        axs[2].set_title('Averaged TRs -- Magnitude')
        axs[3].plot(fft_x, np.abs(rx_fft))
        axs[3].hlines(*peak_results[1:], 'r')
        axs[3].set_title(f'FFT -- Magnitude ({(grad_max*1e-3):.4f} KHz/m gradient max)')
        plt.show()
    
    return grad_max

if __name__ == "__main__":
    phantom_width = 10
    grad_max_cal(phantom_width=10)