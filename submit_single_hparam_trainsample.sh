#!/bin/bash

### for XSede comet cluster (slurm) ###
### submit sbatch ---ignore-pbs *.sh
#SBATCH --job-name=phantomD


source /home/tsriniva/anaconda3/bin/activate
source activate mfn


# python train_mfn_phantom.py --mode PhantomDG --modality_drop $1 --g_loss_weight $2 --hparam_iter $3 
#python train_mfn_phantom.py --mode PhantomD --modality_drop $1 --hparam_iter $2

#python train_mfn_phantom_modalitygen.py --mode PhantomDG_GenModality --modality_drop $1 --g_loss_weight $2 --missing_modality $3 --phantom_modality $4 --permanent_modality $5 --hparam_iter $6

python test_mfn_phantom_trainsample.py --mode PhantomD --modality_drop $1 --hparam_iter $2
