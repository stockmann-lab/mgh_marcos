#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))

from examples import spin_echo_train
import time

if __name__ == "__main__":

    # Change these variables to get a good spin echo
    larmor_freq=15.454 # [MHz] Scanner larmor frequency, will need to change for drift
    rx_period=25/3 # [us] Receive period 
    echo_duration=5000 # [us]
    readout_duration=2500 # [us]
    tr_delay=2e6 # [us] Time between readouts

    points = 9 # Points per calibration stage
    iterations = 5
    focusing = 2


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
                readout_duration=readout_duration
                ))
            time.sleep(2)
        
        rx_arr = np.reshape(rxd_list, (points, -1))
        peak_max_arr = np.max(np.abs(rx_arr), axis=1, keepdims=False)
        rf_max_val = rf_amp_vals[np.argmax(peak_max_arr)]

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(f'Iteration {it + 1}/{iterations}')
        axs[0].plot(np.abs(rx_arr).T)
        axs[0].set_title('Stacked signals -- Magnitude')
        axs[1].plot(rf_amp_vals, peak_max_arr)
        axs[1].set_title(f'Maximum signals -- max at {rf_max_val}')
        plt.show()
        rf_min = max(0, rf_max_val - focusing**(-1 * it - 2))
        rf_max = min(1, rf_max_val + focusing**(-1 * it - 2))

    print(rf_max_val)