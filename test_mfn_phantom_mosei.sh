#for i in {0..99}
for i in 92
do
	# echo "Getting results for PhantomBlind ${i}"
	# python -W ignore test_mfn_phantom_mosei.py --mode PhantomBlind --hparam_iter $i > log/gridsearch_mosei_200epochs/phantomBlind/result_logs/${i}
	# echo "Getting results for PhantomD 0.0 ${i}"
	# python -W ignore test_mfn_phantom_mosei.py --mode PhantomD --modality_drop 0.0 --hparam_iter ${i} > log/gridsearch_mosei_200epochs/MT4_D0.0/result_logs/${i}
	# echo "Getting results for PhantomDG 1.0 ${i}"
	# python -W ignore test_mfn_phantom_mosei.py --mode PhantomDG --modality_drop 1.0 --g_loss_weight 0.01 --hparam_iter ${i} > log/gridsearch_mosei_200epochs/phantomDG_D1.0_G0.01/result_logs/${i}
	# echo "Getting results for PhantomICL ${i}"
	# python -W ignore test_mfn_phantom_mosei.py --mode PhantomICL --hparam_iter ${i} > log/gridsearch_mosei_200epochs/phantomICL/result_logs/${i}
	# echo "Getting results for MT2 ${i}"
	# python -W ignore test_mfn_phantom_mosei.py --mode MT2 --modality_drop 1.0 --g_loss_weight 0.01 --hparam_iter ${i} > log/gridsearch_mosei_200epochs/MT2_D1.0_G0.01/result_logs/${i}
	echo "Getting results for MT3 ${i}"
	python -W ignore test_mfn_phantom_mosei.py --mode MT3 --g_loss_weight 0.01 --hparam_iter ${i} > log/gridsearch_mosei_200epochs/MT3_G0.01/result_logs/${i}
	#echo "Getting results for MT3+ ${i}"
	#python -W ignore test_mfn_phantom_mosei.py --mode MT3+ --g_loss_weight 0.01 --hparam_iter ${i} > log/gridsearch_mosei_200epochs/MT3+_G0.01/result_logs/${i}

done
