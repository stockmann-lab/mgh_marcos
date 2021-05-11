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

def larmor_cal(
    larmor_start=sc.LARMOR_FREQ, iterations=20, step_size=0.9,
    rf_pi2_duration=50, rf_max=sc.RF_MAX, trs=4, tr_spacing=1e6,
    echo_duration=9000, echo_count=4, rx_period=25/3, readout_duration=2500,
    plot=True):
    
    """
    Calibrate gradient maximum using a phantom of known width

    Args:
        larmor_start (float): [MHz] Starting larmor frequency for search
        iterations (int): [arb.] Number of times to iterate through search
        step_size (float): [arb.] Fraction of estimated offset to step for gradient descent
        rf_pi2_duration (float): [us] RF pi/2 pulse duration
        rf_max (float): [Hz] System RF max
        trs (int): [arb.] Number of times to repeat for averaging
        tr_spacing (float): [us] Time between averages and iterations
        echo_duration (float): [us] Time between echo peaks
        echo_count (int): [arb.] Number of echos per train
        rx_period (float): [us] Readout dwell time
        readout_duration (float): [us] Readout window around echo peak\
        plot (bool): Default True, plot final data

    Returns:
        float: Estimated larmor frequency in MHz
    """

    rf_scaling = .25 / (rf_max * 1e-6 * rf_pi2_duration)

    larmor_freq = larmor_start
    for i in range(iterations):
        print(f'Iteration {i + 1}/{iterations}: {larmor_freq:.5f} MHz')
 
        # Run the experiment, hardcoded from marcos_client/examples
        rxd = spin_echo_train(
            larmor_freq=larmor_freq,
            rf_scaling=rf_scaling,
            echo_count=echo_count,
            rx_period=rx_period,
            echo_duration=echo_duration,
            readout_duration=readout_duration,
            rf_pi2_duration=rf_pi2_duration,
            tr_pause_duration=tr_spacing,
            trs=trs
            )

        rxd = np.average(np.reshape(rxd, (trs, -1)), axis=0)

        # Split echos for FFT
        rx_arr = np.reshape(rxd, (echo_count, -1))
        avgs = np.zeros(echo_count)
        stds = np.zeros(echo_count)
        for echo_n in range(echo_count):
            dphis = np.ediff1d(np.angle(rx_arr[echo_n, :]))
            stds[echo_n] = np.std(dphis)
            ordered_dphis = dphis[np.argsort(np.abs(dphis))]
            large_change_ind = np.argmax(np.abs(np.ediff1d(ordered_dphis)))

            dphi_vals = ordered_dphis[:large_change_ind-1]
            avgs[echo_n] = np.mean(dphi_vals)

        dphi = np.mean(avgs[1:-1])
        dw = dphi / (rx_period * np.pi)
        std = np.mean(stds)
        print(f'  Estimated frequency offset: {dw:.6f} MHz')
        print(f'  Spread (std): {std:.6f}')

        larmor_freq += dw * step_size

        time.sleep(tr_spacing * 1e-6)

    print(f'Calibrated Larmor frequency: {larmor_freq:.5f} MHz')

    if plot:
        # Run once more to plot
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

        fft_bw = 1/(rx_period)
        fft_x = np.linspace(larmor_freq - fft_bw/2, larmor_freq + fft_bw/2, num=rx_fft.shape[0])

        fig, axs = plt.subplots(5, 1, constrained_layout=True)
        if std < 1:
            fig.suptitle(f'Larmor: {larmor_freq:.4f} MHz')
        else:
            fig.suptitle(f"Didn't converge -- Try eyeballing from bottom graph")
        axs[0].plot(np.real(rxd))
        axs[0].set_title('Concatenated signal -- Real')
        axs[1].plot(np.abs(rxd))
        axs[1].set_title('Concatenated signal -- Magnitude')
        axs[2].plot(np.angle(rx_arr))
        axs[2].set_title('Stacked signals -- Phase')
        axs[3].plot(fft_x, np.abs(rx_fft))
        axs[3].set_title('Stacked signals -- FFT')
        axs[4].plot(fft_x, np.mean(np.abs(rx_fft), axis=1, keepdims=False))
        axs[4].set_title('Averaged signals -- FFT')
        plt.show()

    return larmor_freq

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    larmor_cal()