# mgh-flocra

## Git installation
This repo uses submodules ([details](https://git-scm.com/book/en/v2/Git-Tools-Submodules))

Run `git submodule init` and `git pull --recurse-submodules` to clone submodules and checkout the right branch for the first time.

To update submodules, run `git submodule update --remote --merge`.

## System setup

### Run on installation, or changing IP:
In `marcos_client/`, duplicate `local_config.py.example`. Uncomment and edit your IP Address, your gradient board, and your system -- for FLOCRA, you'll need an RP-122. Rename the file `local_config.py`.

In `mgh/`, duplicate `config.py.example` and enter system maximum values you feel safe testing. Rename the file `config.py`.

Install packages in development mode (edits to the files change the installed code) by running `pip install -e .` both in the main repo folder and in `flocra-pulseq`.

### Run on each power cycle:
In `marcos_extras`, run `marcos_setup.sh` from the command line with your Red Pitaya IP and version (for example: `./marcos_setup.sh [IP] [version]` -> `./marcos_setup.sh 192.168.1.163 rp-122`). FLOCRA currently only works with RP-122. For more info on setting up marcos, you can look at this more in-depth [wiki](https://github.com/vnegnev/marcos_extras/wiki/setting_marcos_up).

The blue light should turn on -- if it's off, you need to run the setup again.

Once the setup is done, start the server by ssh-ing in and running `~/marcos_server`.

### Calibration -- Run periodically to correct for magnet drift:
To calibrate system values, there are calibration functions in `mgh.calibration`. To run a pulseq file, use `mgh.run_pulseq/`. From there, you can use `flocra_pulseq.interpreter` and modules in the `mgh` and `marcos_client` packages. 
