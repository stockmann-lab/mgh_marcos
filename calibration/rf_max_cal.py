#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))
sys.path.append(os.path.abspath("../config"))
import scanner_config as sc # pylint: disable=import-error

from examples import spin_echo_train # pylint: disable=import-error

if __name__ == "__main__":

    # Change these variables to get a good spin echo
    larmor_freq=15.453751 # [MHz] Scanner larmor frequency, will need to change for drift
    rf_pi2_duration = 100

    rx_period=25/3 # [us] Receive period 
    echo_duration=5000 # [us]
    readout_duration=2500 # [us]
    tr_delay=2e6 # [us] Time between readouts

    points = 11 # Points per calibration stage
    iterations = 6
    focusing = 2

    plot = True

    # Run the experiment, hardcoded from marcos_client/examples
    rf_min, rf_max = 0, 1
    rf_max_val = 0
    for it in range(iterations):
        rf_amp_vals = np.linspace(rf_min, rf_max, num=points, endpoint=True)
        rxd_list = []
        for i in range(points):
            rxd_list.append(spin_echo_train(
                larmor_freq=larmor_freq,
                rf_scaling=rf_amp_vals[i],
                echo_count=2,
                rx_period=rx_period,
                echo_duration=echo_duration,
                readout_duration=readout_duration,
                rf_pi2_duration=rf_pi2_duration
                ))
            time.sleep(2)
        
        rx_arr = np.reshape(rxd_list, (points, 2, -1))[:, 1, :]
        peak_max_arr = np.max(np.abs(rx_arr), axis=1, keepdims=False)
        rf_max_val = rf_amp_vals[np.argmax(peak_max_arr)]

        if plot:
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(f'Iteration {it + 1}/{iterations}')
            axs[0].plot(np.abs(rx_arr).T)
            axs[0].set_title('Stacked signals -- Magnitude')
            axs[1].plot(rf_amp_vals, peak_max_arr)
            axs[1].set_title(f'Maximum signals -- max at {rf_max_val}')
            plt.ion()
            plt.show()
            plt.draw()
            plt.pause(0.001)
        print(f'Iteration {it + 1}/{iterations} --- Max: {np.max(peak_max_arr)} at {rf_max_val:.4f}')
        rf_min = max(0, rf_max_val - focusing**(-1 * it - 2))
        rf_max = min(1, rf_max_val + focusing**(-1 * it - 2))
        if plot: plt.ioff()

    print(f'{rf_pi2_duration}us pulse, 90 degree maxed at {rf_max_val:.4f}')
    print(f'Estimated RF max: {0.25 / (rf_pi2_duration * rf_max_val) * 1e6:.2f} Hz')