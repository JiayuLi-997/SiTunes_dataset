#! /bin/bash
cd ../src
DATADIR="../dataset/"

# setting 2
for seed in {101...110}
do
python main.py --model_name RF --save_anno setting2_CONTEXT_all_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --max_depth 3 --n_estimators 300 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name RF --save_anno setting2_CONTEXT_sub_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --max_depth 5 --n_estimators 500 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name RF --save_anno setting2_CONTEXT_obj_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --max_depth 5 --n_estimators 80 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
done

# setting 3
for seed in {101...110}
do
python main.py --model_name RF --save_anno setting3_CONTEXT_all_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --max_depth 7 --n_estimators 40 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name RF --save_anno setting3_CONTEXT_sub_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --max_depth 5 --n_estimators 50 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name RF --save_anno setting3_CONTEXT_obj_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --max_depth 3 --n_estimators 200 --min_samples_leaf 1 --min_samples_split 2 --class_num 3 --metrics accuracy,macro_f1,micro_f1
done

