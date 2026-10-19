#!/bin/bash
source ../BaseStack/bin/setup_run.sh
PYTHONPATH=`pwd`/src:`pwd`/scripts:${PYTHONPATH}
export PYTHONPATH
source mla/bin/activate
