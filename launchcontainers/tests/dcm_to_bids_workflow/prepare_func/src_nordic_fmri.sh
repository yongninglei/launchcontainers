#!/usr/bin/env bash
module load afni
module load fsl
module load matlab/R2021B
# args: 1=TB_PATH, 2=SRC_DIR, 3=OUTPUT_DIR, 4=SUB, 5=SES,
#       6=NORDIC_END, 7=DONORDIC, 8=DOTSNR, 9=FORCE 10 script_dir
unset $SUB
unset $SES
TB_PATH="$1"
SRC_DIR="$2"
OUTPUT_DIR="$3"
SUB="$4"
SES="$5"
NORDIC_END="$6"
DONORDIC="$7"
DOTSNR="$8"
FORCE="$9"
unset $script_dir
script_dir="${10}"

echo $SUB $SES
echo $script_dir
MATLAB_CMD="addpath('${TB_PATH}'); addpath('${script_dir}');\
nordic_fmri('${TB_PATH}','${SRC_DIR}','${OUTPUT_DIR}',\
'${SUB}','${SES}',${NORDIC_END},${DONORDIC},${DOTSNR},${FORCE}); \
exit;"

cmd="matlab -nodisplay -nosplash -r \"${MATLAB_CMD}\""

eval $cmd
