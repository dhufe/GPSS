#!/bin/bash 

rm -rf build 
rm -rf *.png *.pdf *.mat

python3 setup.py clean
python3 setup.py build_ext --inplace
