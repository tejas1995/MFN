for i in {0..99}
do
	# echo "Getting results for PhantomBlind ${i}"
	# python -W ignore test_mfn_phantom_mosei_emotions.py --mode PhantomBlind --hparam_iter $i > log/gridsearch_mosei_emotions/phantomBlind/result_logs/${i}
	#echo "Getting results for PhantomD ${i}"
	#python -W ignore test_mfn_phantom_mosei_emotions.py --mode PhantomD --modality_drop 0.5 --hparam_iter ${i} > log/gridsearch_mosei_emotions/phantomD_D0.5/result_logs/${i}
	#echo "Getting results for PhantomDG ${i}"
	#python -W ignore test_mfn_phantom_mosei_emotions.py --mode PhantomDG --modality_drop 0.5 --g_loss_weight 0.01 --hparam_iter ${i} > log/gridsearch_mosei_emotions/phantomDG_D0.5_G0.01/result_logs/${i}
	echo "Getting results for PhantomICL ${i}"
	python -W ignore test_mfn_phantom_mosei_emotions.py --mode PhantomICL --hparam_iter ${i} > log/gridsearch_mosei_emotions/PhantomICL/result_logs/${i}

done
