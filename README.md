# BackRazor For Bert

## Environment Setting

1. Install the packages required by [backRazor](https://github.com/VITA-Group/BackRazor_Neurips22)
2. Install the transformers and datasets
```bash
python setup.py install
pip3 install datasets==2.10.1
```

## Runing cmds

Baseline
```bash
num_gpus=4
step=100
seed=5
train_epochs=10
lr=2e-5


OUTPUT_DIR=../checkpoints_bert/rte_E${train_epochs}_lr${lr}
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12376 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps

```

BackRazor
```bash
num_gpus=4
step=100
seed=5
train_epochs=10
lr=2e-5

OUTPUT_DIR=../checkpoints_bert/rte_backRazor_E${train_epochs}_lr${lr}
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12276 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps \
--backRazor True
```

Memory testing
```bash
num_gpus=1
step=100
seed=5
lr=2e-5
train_epochs=10

OUTPUT_DIR=../checkpoints_bert/rte_E${train_epochs}_lr${lr}
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
--master_port=12376 examples/pytorch/text-classification/run_glue.py --save_total_limit 1      \
--model_name_or_path bert-base-uncased --task_name rte --output_dir ${OUTPUT_DIR} --do_train     \
--do_eval  --num_train_epochs ${train_epochs} --save_steps ${step}  --seed ${seed}  --per_device_train_batch_size 8     \
--max_seq_length 512 --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps ${step} \
--load_best_model_at_end True --metric_for_best_model eval_accuracy  --learning_rate ${lr} --evaluation_strategy steps
```

## Acknowledge
The partial code of this implement comes from [transformers](https://github.com/huggingface/transformers)

## Cite
```
@inproceedings{
jiang2022back,
title={Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropogation},
author={Jiang, Ziyu and Chen, Xuxi and Huang, Xueqin and Du, Xianzhi and Zhou, Denny and Wang, Zhangyang},
booktitle={Advances in Neural Information Processing Systems 36},
year={2022}
}
```