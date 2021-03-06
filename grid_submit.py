import standard_grid
import sys

#dataset = "mosei"
dataset = "mosei_emotions"
#dataset = "pom"

filename= "train_mfn_phantom_{}.py".format(dataset)

#mode = 'PhantomD'
#mode = 'PhantomDG'
#mode = 'PhantomBlind'
#mode = 'PhantomICL'
#mode = 'PhantomIntermD'
#mode = 'PhantomIntermInputD'
#mode = 'MT2'
#mode = 'MT3'
mode = 'MT3+'

modality_drop = 1.0
g_loss_weight = 0.01


if dataset == "mosei":
	log_dir = "log/gridsearch_mosei_200epochs/{}".format(mode)
else:
	log_dir = "log/gridsearch_{}/{}".format(dataset, mode)

if mode in ['PhantomD', 'PhantomDG', 'PhantomIntermD', 'PhantomIntermInputD', 'MT2']:
	log_dir += "_D" + str(modality_drop)
if mode in ['PhantomDG', 'MT2', 'MT3', 'MT3+']:
	log_dir += "_G" + str(g_loss_weight)
log_dir += "/"

if __name__=='__main__':

	grid = standard_grid.Grid(filename, log_dir)

	grid.register('mode', [mode])
	if mode in ['PhantomD', 'PhantomDG', 'PhantomIntermD', 'PhantomIntermInputD', 'MT2']:
		grid.register('modality_drop', [modality_drop])
	if mode in ['PhantomDG', 'MT2', 'MT3', 'MT3+']:
		grid.register('g_loss_weight', [g_loss_weight])
	#grid.register('hparam_iter', list(range(100)))
	grid.register('hparam_iter', [96]) #MT3+
	#grid.register('hparam_iter', [59]) #MT2

	grid.generate_grid()

	grid.generate_shell_instances(prefix="source ~/anaconda3/bin/activate ; conda activate mfn ; python ",postfix="")


	total_at_a_time=12
	grid.create_runner(num_runners=total_at_a_time,runners_prefix=["sbatch -p gpu_low -c 1 --gres=gpu:1 --mem=10G -W"]*total_at_a_time)
