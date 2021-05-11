#!/usr/bin/env python3
#
# Run a pulseq file
# Code by Lincoln Craven-Brightman
#

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))
sys.path.append(os.path.abspath("../config"))
import scanner_config as sc # pylint: disable=import-error

import experiment as ex # pylint: disable=import-error
from flocra_pulseq_interpreter import PSInterpreter # pylint: disable=import-error

def run_pulseq(seq_file, rf_center=sc.LARMOR_FREQ, rf_max=sc.RF_MAX,
                gx_max=sc.GX_MAX, gy_max=sc.GY_MAX, gz_max=sc.GZ_MAX,
                tx_t=1, grad_t=10,
                shim_x = sc.SHIM_X, shim_y = sc.SHIM_Y, shim_z = sc.SHIM_Z,
                grad_cal=False, save_np=False, save_mat=False, save_msgs=False):

    # Convert .seq file to machine dict
    psi = PSInterpreter(rf_center=rf_center*1e6,
                        tx_warmup=100,
                        rf_amp_max=rf_max,
                        tx_t=tx_t,
                        grad_t=grad_t,
                        gx_max=gx_max,
                        gy_max=gy_max,
                        gz_max=gz_max,
                        log_file = '../logs/ps_interpreter')
    instructions, param_dict = psi.interpret(seq_file)

    # Shim
    instructions = shim(instructions, (shim_x, shim_y, shim_z))

    # Initialize experiment class
    expt = ex.Experiment(lo_freq=rf_center,
                         rx_t=param_dict['rx_t'],
                         init_gpa=True,
                         gpa_fhdo_offset_time=grad_t/3) 

    # Optionbally run gradient linearization calibration
    if grad_cal:
        expt.gradb.calibrate(channels=[0,1,2], max_current=1, num_calibration_points=30, averages=5, poly_degree=5)

    # Load instructions
    expt.add_flodict(instructions)

    # Run experiment
    rxd, msgs = expt.run()

    # Optionally save messages
    if save_msgs:
        print(msgs) # TODO include message saving

    # Announce completion
    nSamples = param_dict['readout_number']
    print(f'Finished -- read {nSamples} samples')

    # Optionally save rx output array as .npy file
    if save_np:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = f"../data/{current_time}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename,rxd['rx0'])

    # Optionally save rx output array as .mat file
    if save_mat:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = f"../data/{current_time}.mat"
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename,{'flocra_data': rxd['rx0']})

    # Return rx output array and rx period
    return rxd['rx0'], param_dict['rx_t']

def shim(instructions, shim):
    grads = ['grad_vx', 'grad_vy', 'grad_vz']
    for ch in range(3):
        updates = instructions[grads[ch]][1]
        updates[:-1] = updates[:-1] + shim[ch]
        assert(np.all(np.abs(updates) <= 1)), (f'Shim {shim[ch]} was too large for {grads[ch]}: '
                + f'{updates[np.argmax(np.abs(updates))]}')
        instructions[grads[ch]] = (instructions[grads[ch]][0], updates)
    return instructions

def plot_signal_1d(rxd, trs, rx_period, larmor_freq=sc.LARMOR_FREQ):
    # Split echos for FFT
    rx_arr = np.reshape(rxd, (trs, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx_arr, axes=(0,)), axis=0), axes=(0,))

    fft_bw = 1/(rx_period)
    fft_x = np.linspace(larmor_freq - fft_bw/2, larmor_freq + fft_bw/2, num=rx_fft.shape[0])

    _, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.angle(rx_arr))
    axs[2].set_title('Stacked signals -- Phase')
    axs[3].plot(fft_x, np.abs(rx_fft))
    axs[3].set_title('Stacked signals -- FFT')
    plt.show()

def plot_signal_2d(rxd, trs, rx_period, larmor_freq=sc.LARMOR_FREQ, averages=1, raw=False):
    rx_arr = np.reshape(rxd, (trs, -1))
    rx_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rx_arr)))

    if raw:
        rx_fft = rx_arr

    _, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rx_arr.T))
    axs[2].set_title('Stacked signals -- Magnitude')
    _, im_axs = plt.subplots(1, 2, constrained_layout=True)
    im_axs[0].imshow(np.abs(rx_fft))
    im_axs[1].imshow(np.angle(rx_fft))
    plt.show()

def plot_sinogram_2d(rxd, trs, rx_period, larmor_freq=sc.LARMOR_FREQ, averages=1):
    rx_arr = np.reshape(rxd, (trs, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx_arr, axes=(0,)), axis=0), axes=(0,))

    _, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rx_arr.T))
    axs[2].set_title('Stacked signals -- Magnitude')
    _, im_axs = plt.subplots(1, 2, constrained_layout=True)
    im_axs[0].imshow(np.abs(rx_fft))
    im_axs[1].imshow(np.angle(rx_fft))
    plt.show()

if __name__ == "__main__":
    seq_file = '../ISMRM21_demo/se_radial_3D_may_7_2021.seq'
    rxd, rx_t = run_pulseq(seq_file, save_np=False)
    plot_signal_1d(rxd, 64, rx_t)