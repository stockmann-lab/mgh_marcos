#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))
sys.path.append(os.path.abspath("../config"))
import scanner_config as sc # pylint: disable=import-error

from examples import turbo_spin_echo # pylint: disable=import-error

if __name__ == "__main__":

    # Change these variables to change the sequence
    # Optionally test values with a spin echo using spin_echo_train.py

    larmor_freq = 15.454 # [MHz] Scanner larmor frequency, will need to change for drift
    rf_scaling = 0.48 # [Arb.] RF power, higher for more power
    
    echo_count = 64 # Echoes per train
    echo_duration = 5000 # [us] Length of time between echo peaks

    readout_duration = 500 # [us] Length of time for acquisition around echo peak
    rx_period = 25/3 # [us] Receive period

    readout_gradient_scaling=0.4 # [Arb.] Fractional readout gradient power, 1 is full strength
    readout_gradient_duration=2600 # [us] Readout gradient length 
    # ^ must at least be longer than readout_duration + trap_ramp_duration (default 50us)

    phase_gradient_scaling=0.4 # Fractional phase gradient power, 1 is full strength
    phase_gradient_duration=150 # [us] Phase gradient length
    phase_gradient_interval=600 # [us] Interval between positive and negative phase gradients within one echo
    # ^ must at most be shorter than echo_duration - readount_gradient_duration - phase_gradient_duration

    tr_spacing = 750000 # [us] Time between repetitions
    trs = 4 # Number of times to average the image


    # Run the experiment, hardcoded from marcos_client/examples
    rxd = turbo_spin_echo(lo_freq=larmor_freq,
        trs=trs, echos_per_tr=echo_count,
        rf_amp=rf_scaling,
        rx_period=rx_period,
        echo_duration=echo_duration,
        readout_duration=readout_duration,
        readout_amp=readout_gradient_scaling,
        readout_grad_duration=readout_gradient_duration,
        phase_start_amp=phase_gradient_scaling,
        phase_grad_duration=phase_gradient_duration,
        phase_grad_interval=phase_gradient_interval,
        slice_start_amp=0,
        tr_pause_duration=tr_spacing,
        init_gpa=True, plot_rx=False)

    rx_arr = np.reshape(rxd, (trs, echo_count, -1))
    rx_arr_av = np.average(rx_arr, axis=0)
    rxd_av = np.reshape(rx_arr_av, (-1))

    rx_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rx_arr_av)))

    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rxd_av))
    axs[2].set_title('Averaged TRs -- Magnitude')
    axs[3].plot(np.abs(rx_arr_av.T))
    axs[3].set_title('Stacked signals -- Magnitude')
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].imshow(np.abs(rx_fft))
    axs[1].imshow(np.angle(rx_fft))
    plt.show()