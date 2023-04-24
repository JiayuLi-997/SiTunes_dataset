#! /bin/bash
cd ../src
DATADIR="../dataset/"

# setting 2
for seed in {101...110}
do
python main.py --model_name MLP --save_anno setting2_CONTEXT_all_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --learning_rate 5e-3 --batch_size 256 --alpha 1e-6 --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name MLP --save_anno setting2_CONTEXT_sub_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --learning_rate 2e-3 --batch_size 64 --alpha 1e-4 --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name MLP --save_anno setting2_CONTEXT_obj_${seed} --dataname setting2-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --learning_rate 1e-2 --batch_size 32 --alpha 1e-4 --solver 'sgd' --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
done

# setting 3
for seed in {101...110}
do
python main.py --model_name MLP --save_anno setting3_CONTEXT_all_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_all --random_seed ${seed} --learning_rate 1e-3 --batch_size 512 --alpha 1e-6 --solver 'lbfgs' --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name MLP --save_anno setting3_CONTEXT_sub_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_sub --random_seed ${seed} --learning_rate 5e-4 --batch_size 32 --alpha 1e-6 --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
python main.py --model_name MLP --save_anno setting3_CONTEXT_obj_${seed} --dataname setting3-${seed} --context_column_group CONTEXT_obj --random_seed ${seed} --learning_rate 1e-2 --batch_size 256 --alpha 1e-4 --solver 'sgd' --activation 'relu' --class_num 3 --metrics accuracy,macro_f1,micro_f1
done
