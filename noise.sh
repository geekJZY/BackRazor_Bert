OUTPUT_DIR=mnli_bert_noise
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_glue_noise_label.py      --save_total_limit 1      --model_name_or_path bert-base-cased      --task_name mnli      --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.1 > bert.out &

OUTPUT_DIR=mnli_bert_noise
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_glue_noise_label.py      --save_total_limit 1      --model_name_or_path bert-base-cased      --task_name mnli      --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.5 > bert_0.5.out &

OUTPUT_DIR=mnli_bert_noise_0.8
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12346 non-GPT-2/examples/pytorch/text-classification/run_glue_noise_label.py      --save_total_limit 1      --model_name_or_path bert-base-cased      --task_name mnli      --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.8 > bert_0.8.out &




OUTPUT_DIR=xnli_mbert_en
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12342 non-GPT-2/examples/pytorch/text-classification/run_xnli_noise_label.py --language en     --save_total_limit 1      --model_name_or_path bert-base-multilingual-cased    --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.1 > mbert_0.1.out &

OUTPUT_DIR=xnli_mbert_en_2
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_xnli_noise_label.py --language en     --save_total_limit 1      --model_name_or_path bert-base-multilingual-cased    --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.2 > mbert_0.2.out &

OUTPUT_DIR=xnli_mbert_en_2
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_xnli_noise_label.py --language en     --save_total_limit 1      --model_name_or_path bert-base-multilingual-cased    --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.4 > mbert_0.4.out &

OUTPUT_DIR=xnli_mbert_en_2
lr=5e-5
num_gpus=4
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_xnli_noise_label.py --language en     --save_total_limit 1      --model_name_or_path bert-base-multilingual-cased    --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.8 > mbert_0.8.out &





OUTPUT_DIR=mnli_bert_noise
lr=5e-5
num_gpus=1
step=1000
WANDB_DISABLED=true NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=12341 non-GPT-2/examples/pytorch/text-classification/run_glue_noise_label.py      --save_total_limit 1      --model_name_or_path bert-base-cased      --task_name mnli      --output_dir ${OUTPUT_DIR}      --do_train      --do_eval      --num_train_epochs 3      --save_steps ${step}      --seed 3     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps ${step}      --load_best_model_at_end True      --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps --noise_level 0.1