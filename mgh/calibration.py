#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

import sys

import mgh.config as cfg # pylint: disable=import-error
from mgh.scripts import run_pulseq # pylint: disable=import-error

def larmor_cal(larmor_start=cfg.LARMOR_FREQ, iterations=10,
 step_size=0.6, plot=False, shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z,):

    larmor_freq = larmor_start
    seq_file = cfg.MGH_PATH + 'cal_seq_files/se.seq'
    echo_count = 1

    for i in range(iterations):
        print(f'Iteration {i + 1}/{iterations}: {larmor_freq:.5f} MHz')
 
        # Run the experiment, hardcoded from marcos_client/examples
        rxd, rx_t = run_pulseq(seq_file, rf_center=larmor_freq,
                tx_t=1, grad_t=10, tx_warmup=100,
                shim_x=shim_x, shim_y=shim_y, shim_z=shim_z,
                grad_cal=False, save_np=False, save_mat=False, save_msgs=False)

        rxd = np.average(np.reshape(rxd, (1, -1)), axis=0)

        # Split echos for FFT
        rx_arr = np.reshape(rxd, (echo_count, -1))
        avgs = np.zeros(echo_count)
        stds = np.zeros(echo_count)
        for echo_n in range(echo_count):
            dphis = np.ediff1d(np.angle(rx_arr[echo_n, :]))
            stds[echo_n] = np.std(dphis)
            ordered_dphis = dphis[np.argsort(np.abs(dphis))]
            large_change_ind = np.argmax(np.abs(np.ediff1d(np.abs(ordered_dphis))))
            dphi_vals = ordered_dphis[:large_change_ind-1]
            avgs[echo_n] = np.mean(dphi_vals)

        dphi = np.mean(avgs)
        dw = dphi / (rx_t * np.pi)
        std = np.mean(stds)
        print(f'  Estimated frequency offset: {dw:.6f} MHz')
        print(f'  Spread (std): {std:.6f}')

        larmor_freq += dw * step_size

        time.sleep(1)

    # Run once more to check final frequency
    rxd, rx_t = run_pulseq(seq_file, rf_center=larmor_freq,
            tx_t=1, grad_t=1, tx_warmup=100,
            shim_x = cfg.SHIM_X, shim_y = cfg.SHIM_Y, shim_z = cfg.SHIM_Z,
            grad_cal=False, save_np=False, save_mat=False, save_msgs=False)

    # Split echos for FFT
    rx_arr = np.reshape(rxd, (echo_count, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(rx_arr, axis=0), axes=(0,))

    fft_bw = 1/(rx_t)
    fft_x = np.linspace(larmor_freq - fft_bw/2, larmor_freq + fft_bw/2, num=rx_fft.shape[0])

    print(f'Calibrated Larmor frequency: {larmor_freq:.6f} MHz')
    if std >= 1:
        
        print(f"Didn't converge, try {fft_x[np.argmax(rx_fft[:, 0])]:.6f}")

    if plot:
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

def rf_max_cal(larmor_freq=cfg.LARMOR_FREQ, tr_spacing=2e6, echo_duration=5000, 
                 readout_duration=500, rx_period=25/3,
                 rf_pi2_duration=50, rf_max=cfg.RF_MAX,
                 plot=True, points=11, iterations=6, focus_factor=1.5):
    """
    Calibrate gradient maximum using a phantom of known width

    Args:
        larmor_freq (float): [MHz] Scanner larmor frequency
        tr_spacing (float): [us] Time between repetitions
        echo_duration (float): [us] Time between echo peaks
        readout_duration (float): [us] Readout window around echo peak
        rx_period (float): [us] Readout dwell time
        rf_pi2_duration (float): [us] RF pi/2 pulse duration
        rf_max (float): [Hz] System RF max
        plot (bool): Default True, plot final data
        points (int): Points to plot per iteration
        iterations (int): Iterations to focus in
        focus_factor (float): About to zoom in by each iteration -- must be greater than 1

    Returns:
        float: Estimated RF max in Hz
    """
    assert(focus_factor > 1)
    from marcos_client.examples import spin_echo_train

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
            time.sleep(tr_spacing)
        
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
        rf_min = max(0, rf_max_val - focus_factor**(-1 * it - 2))
        rf_max = min(1, rf_max_val + focus_factor**(-1 * it - 2))
        if plot: plt.ioff()

    print(f'{rf_pi2_duration}us pulse, 90 degree maxed at {rf_max_val:.4f}')
    print(f'Estimated RF max: {0.25 / (rf_pi2_duration * rf_max_val) * 1e6:.2f} Hz')
    return 0.25 / (rf_pi2_duration * rf_max_val) * 1e6

def grad_max_cal(phantom_width=10, larmor_freq=cfg.LARMOR_FREQ, calibration_power=0.8,
                 trs=2, tr_spacing=2e6, echo_duration=5000, 
                 readout_duration=500, rx_period=25/3, gradient_overshoot=100,
                 rf_pi2_duration=50, rf_max=cfg.RF_MAX,
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

    from marcos_client.examples import turbo_spin_echo

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
    if len(sys.argv) == 2:
        larmor_cal(iterations=int(sys.argv[1]), plot=True)
    else:
        larmor_cal(iterations=8, plot=True)
    