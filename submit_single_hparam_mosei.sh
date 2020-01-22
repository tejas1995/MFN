#!/bin/bash

### for XSede comet cluster (slurm) ###
### submit sbatch ---ignore-pbs *.sh
#SBATCH --job-name=phantomD


source /home/tsriniva/anaconda3/bin/activate
source activate mfn

#python train_mfn_phantom_mosei.py --mode PhantomD --modality_drop $1 --hparam_iter $2
python train_mfn_phantom_mosei.py --mode PhantomDG --modality_drop $1 --g_loss_weight $2 --hparam_iter $3 
#python train_mfn_phantom_mosei.py --mode PhantomBlind --hparam_iter $1
#python train_mfn_phantom_mosei.py --mode PhantomICL --hparam_iter $1
