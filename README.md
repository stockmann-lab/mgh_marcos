# mgh-flocra

## Git installation
This repo uses submodules ([details](https://git-scm.com/book/en/v2/Git-Tools-Submodules))

Run `git submodule init` and `git submodule update --remote --merge` to clone submodules and checkout the right branch for the first time.

To update submodules, run `git submodule update --remote --merge`.

## System setup
In `marcos_client/`, duplicate `local_config.py.example` and uncomment the lines that match your system. Rename the file `local_config.py`.

In `config/`, duplicate `scanner_config.py.example` and enter system maximum values you feel safe testing. Rename the file `scanner_config.py`.

To calibrate system values, there are calibration scripts in `calibration/`, along with a README explaining them. Edit `scanner_config.py` to reflect the output values. 