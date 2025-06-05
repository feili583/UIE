## Dependency:

* [apex](https://github.com/NVIDIA/apex)
* [torch-struct](https://github.com/harvardnlp/pytorch-struct)

## Preparation
Put your dataset under the data folder.

Secondly, pretrained LM (i.e., [bert-chinese](https://huggingface.co/hfl/chinese-bert-wwm), [bert](https://huggingface.co/google-bert/bert-base-uncased/tree/main))

## Running our approaches

Train:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
--output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_train_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
--test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
--dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/train --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json
```

Test:
```bash
python train.py --output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_dev_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
--test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/test --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json
