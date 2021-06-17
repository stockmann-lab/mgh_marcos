# mgh-flocra

## Git installation
This repo uses submodules ([details](https://git-scm.com/book/en/v2/Git-Tools-Submodules))

Run `git submodule init` and `git submodule update --remote --merge` to clone submodules and checkout the right branch for the first time.

To update submodules, run `git submodule update --remote --merge`.

## System setup
In `marcos_client/`, duplicate `local_config.py.example` and uncomment the lines that match your system. Rename the file `local_config.py`.

In `mgh/`, duplicate `config.py.example` and enter system maximum values you feel safe testing. Rename the file `config.py`.

Install packages in development mode (edits to the files change the installed code) by running `pip install -e .` both in the main repo folder and in `flocra-pulseq`.

In `marcos_extras`, run `setup_marcos.sh` from the command line with your Red Pitaya IP and version (for example: `./marcos_setup.sh 192.168.1.163 rp-122`). FLOCRA currently only works with RP-122. For more info on setting up marcos, you can look at this more in-depth [wiki](https://github.com/vnegnev/marcos_extras/wiki/setting_marcos_up).

To calibrate system values, there are calibration functions in `mgh.calibration`. To run a pulseq file, use `mgh.run_pulseq/`. From there, you can use `flocra_pulseq.interpreter` and modules in the `mgh` and `marcos_client` packages. 
