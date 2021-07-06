#!/bin/bash

cd $HOME/GPSS

# load module to env
module load python/3.8.5

# start the real simulation
python3 test_YZ.py

