#!/usr/bin/env python3
#
# Run a pulseq file
# Code by Lincoln Craven-Brightman
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import sys
import os

import mgh.config as cfg # pylint: disable=import-error
import marcos_client.experiment as ex # pylint: disable=import-error
from flocra_pulseq.interpreter import PSInterpreter # pylint: disable=import-error

def run_pulseq(seq_file, rf_center=cfg.LARMOR_FREQ, rf_max=cfg.RF_MAX,
                gx_max=cfg.GX_MAX, gy_max=cfg.GY_MAX, gz_max=cfg.GZ_MAX,
                tx_t=1, grad_t=10, tx_warmup=100,
                shim_x = cfg.SHIM_X, shim_y = cfg.SHIM_Y, shim_z = cfg.SHIM_Z,
                grad_cal=False, save_np=False, save_mat=False, save_msgs=False,
                expt=None, plot_instructions=False):

    # Convert .seq file to machine dict
    psi = PSInterpreter(rf_center=rf_center*1e6,
                        tx_warmup=tx_warmup,
                        rf_amp_max=rf_max,
                        tx_t=tx_t,
                        grad_t=grad_t,
                        gx_max=gx_max,
                        gy_max=gy_max,
                        gz_max=gz_max,
                        log_file = cfg.LOG_PATH + 'ps-interpreter')
    instructions, param_dict = psi.interpret(seq_file)

    # Shim
    instructions = shim(instructions, (shim_x, shim_y, shim_z))

    # Initialize experiment class
    if expt is None:
        expt = ex.Experiment(lo_freq=rf_center,
                         rx_t=param_dict['rx_t'],
                         init_gpa=True,
                         gpa_fhdo_offset_time=grad_t/3,
                         halt_and_reset=True)

    # Optionbally run gradient linearization calibration
    if grad_cal:
        expt.gradb.calibrate(channels=[0,1,2], max_current=1, num_calibration_points=30, averages=5, poly_degree=5)

    # Add flat delay to avoid housekeeping at the start
    flat_delay = 10
    for buf in instructions.keys():
        instructions[buf] = (instructions[buf][0] + flat_delay, instructions[buf][1])

    
    if plot_instructions:
        _, axs = plt.subplots(len(instructions), 1, constrained_layout=True)
        for i, key in enumerate(instructions.keys()):
            axs[i].step(instructions[key][0], instructions[key][1], where='post')
            axs[i].plot(instructions[key][0], instructions[key][1], 'rx')
            axs[i].set_title(key)
        plt.show()
        
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
        filename = cfg.DATA_PATH + f"/{current_time}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename,rxd['rx0'])

    # Optionally save rx output array as .mat file
    if save_mat:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = cfg.DATA_PATH + f"/{current_time}.mat"
        if os.path.exists(filename):
            os.remove(filename)
        sio.savemat(filename,{'flocra_data': rxd['rx0']})

    expt.__del__()

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

def plot_signal_1d(rxd, trs, rx_period, larmor_freq=cfg.LARMOR_FREQ):
    # Split echos for FFT
    rx_arr = np.reshape(rxd, (trs, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx_arr, axes=(0,)), axis=0), axes=(0,))
    x = np.linspace(0, rx_arr.shape[0] * rx_t * 1e-6, num=rx_arr.shape[0], endpoint=False)

    fft_bw = 1/(rx_period)
    fft_x = np.linspace(larmor_freq - fft_bw/2, larmor_freq + fft_bw/2, num=rx_fft.shape[0])

    _, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(x, np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(x, np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.angle(rx_arr))
    axs[2].set_title('Stacked signals -- Phase')
    # if peak_width:
    #     import scipy.signal as sig
    #     peaks, _ = sig.find_peaks(np.abs(rx_fft[:,0]), width=5)
    #     peak_results = sig.peak_widths(np.abs(rx_fft[:,0]), peaks, rel_height=0.5)
    #     fwhm = peak_results[0][0]
    #     axs[3].plot(np.abs(rx_fft))
    #     axs[3].hlines(*peak_results[1:], 'r')
    #     axs[3].set_title(f'Stacked signals -- FFT. Peak width: {fwhm * fft_bw / rx_fft.shape[0]:.3e} Hz')
    #     axs[3].margins(x=-0.45, y=0.05)
    #     print(fwhm)
    # else:
    #     axs[3].plot(fft_x, np.abs(rx_fft))
    #     axs[3].set_title('Stacked signals -- FFT')
    plt.show()

def plot_signal_2d(rxd, trs, raw=False):
    rx_arr = np.reshape(rxd, (trs, -1))
    rx_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rx_arr)))

    if raw:
        rx_fft = rx_arr

    _, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rx_arr))
    axs[2].set_title('Stacked signals -- Magnitude')
    fig, im_axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle('2D Image')
    im_axs[0].imshow(np.abs(rx_fft), cmap=plt.cm.bone)
    im_axs[0].set_title('Magnitude')
    im_axs[1].imshow(np.angle(rx_fft))
    im_axs[1].set_title('Phase')
    plt.show()

def plot_sinogram_2d(rxd, trs):
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
    im_axs[0].imshow(np.abs(rx_fft), aspect='auto')
    im_axs[1].imshow(np.angle(rx_fft), aspect='auto')
    plt.show()

if __name__ == "__main__":
    # Maybe clean up
    trs = 128
    if len(sys.argv) == 3:
        seq_file = '../ISMRM21_demo/' + sys.argv[1]
        dimension = int(sys.argv[2])
    elif len(sys.argv) == 4:
        seq_file = '../ISMRM21_demo/' + sys.argv[1]
        dimension = int(sys.argv[2])
        trs = int(sys.argv[3])
    else:
        seq_file = '../ISMRM21_demo/radial_test_LCB_3.seq'
        dimension = 2
    print(f'Running {seq_file}')
    rxd, rx_t = run_pulseq(seq_file, save_np=True, save_mat=True)
    if dimension == 1:
        plot_signal_1d(rxd, 1, rx_t)
    elif dimension == 2:
        plot_signal_2d(rxd, trs, rx_t)