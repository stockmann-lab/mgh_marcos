#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import scipy.signal as sig
sys.path.append(os.path.abspath("../marcos_client"))

from examples import turbo_spin_echo

if __name__ == "__main__":

    # Change these variables to change the sequence
    # Optionally test values with a spin echo using spin_echo_train.py

    larmor_freq = 15.454 # Scanner larmor frequency [MHz], will need to change for drift
    rf_scaling = 0.48 # RF power [Arbitrary], higher for more power

    echo_duration = 5000 # Length of time between echo peaks [us]

    readout_duration = 500 # Length of time for acquisition around echo peak [us]
    rx_period = 25/3 # Receive period [us]

    tr_spacing = 2e6 # Time between repetitions [us]
    trs = 4 # Number of times to average the image
    
    calibration_power = 0.4 # [Arb.] Fractional system power for calibration
    gradient_overshoot = 100 # [us] Total duration longer than readout_duration the gradienbt will run

    phantom_width = 9 # [mm]

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

    peaks, _ = sig.find_peaks(rx_fft)
    peak_results = sig.peak_widths(rx_fft, peaks)
    fwhm = peak_results[0][0]

    fft_scale = 1 # [Hz/index] Need to know what the distance on x is
    grad_max = fwhm * fft_scale / (phantom_width * 1e-3 * calibration_power) # [Hz/m]
    print(f'Gradient max: {grad_max:.4f} Hz/m')

    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rxd_av))
    axs[2].set_title('Averaged TRs -- Magnitude')
    axs[3].plot(np.abs(rx_fft))
    axs[3].hlines(*peak_results[1:])
    axs[3].set_title(f'FFT -- Magnitude ({grad_max:.4f} gradient max)')
    plt.show()