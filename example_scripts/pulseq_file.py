#!/usr/bin/env python3
#
# loopback test using ocra-pulseq
#

import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

import sys
import os
sys.path.append(os.path.abspath("../marcos_client"))

import experiment as ex
from flocra_pulseq_interpreter import PSInterpreter
st = pdb.set_trace

# TODO input calibrated values
def run_pulseq(seq_file, rf_center=15.45, rf_max=10417, 
                grad_max=15e6, tx_t=1, grad_t=1, save_np=False):

    # Convert .seq file to machine dict
    psi = PSInterpreter(rf_center=rf_center*1e6,
                        rf_amp_max=rf_max,
                        tx_t=tx_t,
                        grad_t=grad_t,
                        grad_max=grad_max)
    instructions, param_dict = psi.interpret("../ISMRM21_demo/se.seq")
    
    print(instructions)
    quit()

    # Convert
    expt = ex.Experiment(lo_freq=rf_center,
                         rx_t=param_dict['rx_t'],
                         init_gpa=True,
                         gpa_fhdo_offset_time=grad_t/3) 

    # TODO include calibration
    #expt.gradb.calibrate(channels=[0,1,2], max_current=1, num_calibration_points=30, averages=5, poly_degree=5)

    # TODO include shims

    expt.add_flodict(instructions)

    rxd, msgs = expt.run()
    nSamples = param_dict['readout_number']
    print(f'Read {nSamples} samples')

    if save_np:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = f"pulseq_data/{current_time}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename,rxd['rx0'])

    return rxd

if __name__ == "__main__":
    rf_center = 15.45 # MHz
    tx_t = 1 # us
    grad_t = 10 # us between [num_grad_channels] channel updates

    gamma = 42570000 # Hz/T

    grad_max = 15e6 # [Hz/m] Calibrated value to normalize gradient amplitude -- use gradient_max_cal.py
    rf_max = 10000 # [Hz] Calibrated value to normalize RF amplitude -- use rf_max_cal.py

    seq_file = '../ISMRM21_demo/se.seq'

    rxd = run_pulseq(seq_file, rf_center=rf_center, rf_max=rf_max, 
                    grad_max=grad_max, tx_t=tx_t, grad_t=grad_t, save_np=False)
    
    plt.close()
    plt.ioff()

    plt.subplot(2,1,1)
    plt.plot(np.abs(rxd['rx0']))
    plt.subplot(2,1,2)
    plt.plot(np.abs(rxd['rx0']))
    plt.show()

  

 
