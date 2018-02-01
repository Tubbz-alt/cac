Collection of scripts to characterize and test epix10ka camera for LCLS experiment mfxx32516.


Files
--------

* dump_config.py dumps epix10ka configuration and pixel matrix to stdout and csv.
* dump_data_mpi.py dumps epix10ka and other devices to hdf5 file for offline analysis.
* get_run_from_runid.py gets the run number
* live_env.py displays alarms for dew point and other environmental properties.
* settings.sh source this file before running any of the scripts here on psana servers.
* submit_conf_dump.sh a shell script executed by the Automatic Run Processing.
* submit_dump_config.py gets executed by the bash script above and posts JSON results to Data Manager website.

Credits
---------

Many of these scripts are based on LCLS documentation and Jason Koglin's help.
