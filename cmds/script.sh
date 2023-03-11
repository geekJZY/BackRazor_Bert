########## Baseline ##########
num_gpus=4
step=100
seed=5
train_epochs=5
lr=5e-5

OUTPUT_DIR=../checkpoints_bert/rte_E${train_epochs}_lr${lr}
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12376 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps


##########  BackRazor ##########
num_gpus=4
step=100
seed=5
train_epochs=5
lr=5e-5

OUTPUT_DIR=../checkpoints_bert/rte_backRazor_E${train_epochs}_lr${lr}

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12276 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps \
--backRazor True



########## mem test ##########
num_gpus=1
step=100
seed=5
lr=5e-5
train_epochs=5

OUTPUT_DIR=../checkpoints_bert/rte_E${train_epochs}_lr${lr}
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12376 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps
