#!/bin/bash
cd ~/hrh_v2_stage2_20261008
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/hrh_v2_stage1_20261008/venv/bin/python src/analyze.py s22 --jobs 2 > logs/analyze_s22_interim.log 2>&1 && touch logs/analyze_s22_interim.done || touch logs/analyze_s22_interim.failed
