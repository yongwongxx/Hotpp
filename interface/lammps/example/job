#!/bin/sh
#BSUB -J reann 
#BSUB -e err
#BSUB -o out
#BSUB -q 62v100ib 
#BSUB -gpu num=1
#
module load gpu
/fs08/home/js_wangjj/src/lammps/lammps-29Sep2021/build/lmp_miao -in in.lammps >out
