#!/bin/bash

### for XSede comet cluster (slurm) ###
### submit sbatch ---ignore-pbs *.sh
#SBATCH --job-name=phantomD


source /home/tsriniva/anaconda3/bin/activate
source activate mfn

echo $1 $2 $3

# python train_mfn_phantom.py --mode PhantomDG --modality_drop $1 --g_loss_weight $2 --hparam_iter $3 
python train_mfn_phantom.py --mode PhantomD --modality_drop $1 --hparam_iter $2 
