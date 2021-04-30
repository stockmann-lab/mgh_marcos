#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))

from examples import spin_echo_train

if __name__ == "__main__":

    # Change these variables to get a good spin echo
    larmor_freq=15.454 # [MHz] Scanner larmor frequency, will need to change for drift
    rf_scaling=0.48 # [Arbitrary] RF power, higher for more power
    echo_count=4 # Echoes per train
    rx_period=25/3 # [us] Receive period
    echo_duration=5000 # [us]
    readout_duration=2500 # [us]

    # Run the experiment, hardcoded from marcos_client/examples
    rxd = spin_echo_train(
        larmor_freq=larmor_freq,
        rf_scaling=rf_scaling,
        echo_count=echo_count,
        rx_period=rx_period,
        echo_duration=echo_duration,
        readout_duration=readout_duration
        )

    # Split echos for FFT
    rx_arr = np.reshape(rxd, (echo_count, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(rx_arr, axis=0), axes=(0,))


    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.angle(rx_arr))
    axs[2].set_title('Stacked signals -- Phase')
    axs[3].plot(np.abs(rx_fft))
    axs[3].set_title('Stacked signals -- FFT')
    plt.show()