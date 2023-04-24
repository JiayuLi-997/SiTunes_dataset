#! /bin/bash
cd ../src
DATADIR="../dataset/"

# setting 2
for seed in {101...110}
do
python main.py --model_name SVM --save_anno setting2_CONTEXT_all_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --regularization 2.0 --kernel linear --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name SVM --save_anno setting2_CONTEXT_sub_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --regularization 1e-4 --kernel rbf --gamma 'scale' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name SVM --save_anno setting2_CONTEXT_obj_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --regularization 2.0 --kernel rbf --gamma 'auto' --class_num 3 --metrics accuracy,macro_f1,micro_f1
done

# setting 3
for seed in {101...110}
do
python main.py --model_name SVM --save_anno setting3_CONTEXT_all_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --regularization 2.0 --kernel rbf --gamma 'scale' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name SVM --save_anno setting3_CONTEXT_sub_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --regularization 1e-4 --kernel rbf --gamma 'scale' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name SVM --save_anno setting3_CONTEXT_obj_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --regularization 1e-4 --kernel rbf --gamma 'scale' --class_num 3 --metrics accuracy,macro_f1,micro_f1
done


