#!/bin/bash

# Dumps Epix10ka configuration (pixel map in png and csv, and diff from previous run)
# LCLS automatic run processing submit scripts
# see https://confluence.slac.stanford.edu/display/PSDM/Automatic+Run+Processing
#

# Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
# License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
# Date: 20180105


unset PYTHONPATH
unset LD_LIBRARY_PATH
unset DISPLAY XAUTHORITY

TGT_PATH=`readlink -e /reg/d/psdm/mfx/mfxx32516/results/abunimeh/2/configs`
SCP_PATH=`readlink -e /reg/d/psdm/mfx/mfxx32516/results/abunimeh/2/scripts`
BLOGDIR=`readlink -e /reg/neh/home/abunimeh/blogs`
# OUTLOG=`readlink -e /reg/neh/home/abunimeh/logme`

# echo "arg1 $1" > $OUTLOG
# echo "arg2 $2" >> $OUTLOG


if [[ $3 ]]; then
	QU=$3
else
	QU="psanaq"
fi

# source /reg/neh/home/abunimeh/settings.sh
source /reg/g/psdm/etc/psconda.sh
source activate ana-1.3.40
# conda info --envs >> $OUTLOG
# which python >> $OUTLOG

RUN=`python $SCP_PATH/get_run_from_runid.py $1 $2`

# echo "EXP $1" >> $OUTLOG
# echo "RUN_ID $2" >> $OUTLOG
# echo "RUN $RUN" >> $OUTLOG
# echo "URL $BATCH_UPDATE_URL" >> $OUTLOG

cd $TGT_PATH
# echo `pwd` >> $OUTLOG
# echo bsub -q "$QU" -o $BLOGDIR/%J.log python "$SCP_PATH/submit_dump_config.py" -r $RUN -i 0 -o "$TGT_PATH/r${RUN}_0.conf" >> $OUTLOG
bsub -q "$QU" -o $BLOGDIR/%J.log python "$SCP_PATH/submit_dump_config.py" -r $RUN -i 0 -o "$TGT_PATH/r${RUN}_0.conf" -p -d
bsub -q "$QU" -o $BLOGDIR/%J.log python "$SCP_PATH/submit_dump_config.py" -r $RUN -i 1 -o "$TGT_PATH/r${RUN}_1.conf" -p -d
bsub -q "$QU" -o $BLOGDIR/%J.log python "$SCP_PATH/submit_dump_config.py" -r $RUN -i 2 -o "$TGT_PATH/r${RUN}_2.conf" -p -d
