#!/bin/bash
# """
# MIT License

# Copyright (c) 2024-2025 Yongning Lei

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# """
# baseP=/bcbl/home/public/Gari/VOTCLOC/main_exp
# code_dir=/export/home/tlei/tlei/soft/launchcontainers/src/launchcontainers/py_pipeline/04b_prf
# LOG_DIR=$baseP/BIDS/derivatives/prfresult/prfresult_logs
# HOME_DIR=$baseP/singularity_home
# version='0.1.1'
# if [ ! -d $LOG_DIR ]; then
# 	mkdir -p $LOG_DIR
# fi
# if [ ! -d $HOME_DIR ]; then
# 	mkdir -p $HOME_DIR
# fi
# current_time=$(date +"%Y-%m-%d_%H-%M-%S")

module load apptainer

cmd="unset PYTHONPATH; singularity run \
	-B /bcbl:/bcbl
	-B /export:/export
	-H $baseP/singularity_home \
    -B $baseP/BIDS/derivatives:/flywheel/v0/data/derivatives \
    -B $baseP/BIDS:/flywheel/v0/BIDS  \
	-B $json_path:/flywheel/v0/config.json \
	--cleanenv ${sif_path} \
	--verbose "

echo "This is the command running :$cmd"
echo "start running ####################"
eval $cmd

module unload apptainer
