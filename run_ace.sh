# python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12355 train.py \
# --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_only_relation/pot_train_1031.txt --predict_file data/EE/ACE_only_relation/pot_dev_1031.txt \
# --test_file data/EE/ACE_only_relation/pot_test_1031.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --learning_rate 3e-5 --num_train_epochs 40 --overwrite_output_dir --save_steps 1000 --fp16 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_relation_1031 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python train.py --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_train_1224.txt --predict_file data/EE/ACE/pot_dev_1224.txt \
# --test_file data/EE/ACE/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 48 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_1224 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_genia --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/conll2003/train.txt --predict_file data/conll2003/dev.txt \
# --test_file data/conll2003/test.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset CONLL --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll2003/train --valid_pattern_path ./data/conll2003/valid_pattern.json

# python train.py --output_dir outputs_genia --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/conll2003/dev.txt --predict_file data/conll2003/dev.txt \
# --test_file data/conll2003/test.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset CONLL --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll2003/test --valid_pattern_path ./data/conll2003/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_conll --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/genia/train.data --predict_file data/genia/dev.data \
# --test_file data/genia/test.data --max_seq_length 64 --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 3 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 4 \
# --dataset GENIA --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/genia/train --valid_pattern_path ./data/genia/valid_pattern.json

# python train.py --output_dir outputs_conll --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/genia/dev.data --predict_file data/genia/dev.data \
# --test_file data/genia/test.data --max_seq_length 64 --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 3 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset GENIA --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/genia/test --valid_pattern_path ./data/genia/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_ace_1 --model_type roberta --config_name ./roberta-large \
# --model_name_or_path ./roberta-large --train_file data/EE/ACE/pot_train_0520.txt --predict_file data/EE/ACE/pot_dev_0520.txt \
# --test_file data/EE/ACE/pot_test_0520.txt --max_seq_length 64 --per_gpu_train_batch_size 18 --per_gpu_eval_batch_size 48 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/pot_train_0520 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python train.py --output_dir outputs_ace_1 --model_type roberta --config_name ./roberta-large \
# --model_name_or_path ./roberta-large --train_file data/EE/ACE/pot_test_0520.txt --predict_file data/EE/ACE/pot_test_0520.txt \
# --test_file data/EE/ACE/pot_test_0520.txt --max_seq_length 64 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/pot_test_0520 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_train_1224.txt --predict_file data/EE/ACE/pot_dev_1224.txt \
# --test_file data/EE/ACE/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --fp16 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_1224 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python train.py --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_dev_1224.txt --predict_file data/EE/ACE/pot_dev_1224.txt \
# --test_file data/EE/ACE/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_1224 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12355 train.py \
# --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_test_1025.txt --predict_file data/EE/ACE/pot_dev_1025.txt \
# --test_file data/EE/ACE/pot_test_1025.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --learning_rate 3e-5 --num_train_epochs 40 --overwrite_output_dir --save_steps 1000 --fp16 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_1025 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python train.py --output_dir outputs_ace --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_test_1025.txt --predict_file data/EE/ACE/pot_dev_1025.txt \
# --test_file data/EE/ACE/pot_test_1025.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_1025 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

# python train.py --output_dir outputs_ace_add --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_add/pot_train.txt --predict_file data/EE/ACE_add/pot_dev.txt \
# --test_file data/EE/ACE_add/pot_test.txt --max_seq_length 64 --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 48 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add.txt --valid_pattern_path ./data/EE/ACE_add/valid_pattern.json
#120
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12355 \
# train.py --output_dir outputs_ace_add --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_add/pot_train_1105.txt --predict_file data/EE/ACE_add/pot_dev_1025.txt \
# --test_file data/EE/ACE_add/pot_test_1025.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add.txt --fp16 --valid_pattern_path ./data/EE/ACE_add/valid_pattern.json

# torchrun --master_port=25641 train.py --output_dir outputs_ace_event --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_event/pot_train_1027.txt --predict_file data/EE/ACE_event/pot_dev_1027.txt \
# --test_file data/EE/ACE_event/pot_test_1027.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 --fp16 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine --local_rank 0 \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_event.txt --valid_pattern_path ./data/EE/ACE_event/valid_pattern.json

# torchrun --master_port=25642 train.py --output_dir outputs_ace_relation --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_relation/pot_train_1027.txt --predict_file data/EE/ACE_relation/pot_dev_1027.txt \
# --test_file data/EE/ACE_relation/pot_test_1027.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 --fp16 \
# --dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine --local_rank 0 \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_relation.txt --valid_pattern_path ./data/EE/ACE_relation/valid_pattern.json


# torchrun --master_port=25643 train.py --output_dir outputs_ace_add_relation --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_add_relation/pot_train_1027.txt --predict_file data/EE/ACE_add_relation/pot_dev_1027.txt \
# --test_file data/EE/ACE_add_relation/pot_test_1027.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add_relation.txt --fp16 --local_rank 0 --valid_pattern_path ./data/EE/ACE_add_relation/valid_pattern.json

# torchrun --master_port=25644 train.py --output_dir outputs_ace_add_event --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_add_event/pot_train_1027.txt --predict_file data/EE/ACE_add_event/pot_dev_1027.txt \
# --test_file data/EE/ACE_add_event/pot_test_1027.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add_event.txt --fp16 --local_rank 0 --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
--output_dir outputs_relation --model_type bert --config_name ./outputs_relation \
--model_name_or_path ./outputs_relation --train_file data/EE/ACE_relation/pot_train_1224.txt --predict_file data/EE/ACE_relation/pot_dev_1224.txt \
--test_file data/EE/ACE_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 12 \
--dataset ACE_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_relation/train --valid_pattern_path ./data/EE/ACE_relation/valid_pattern.json

python train.py --output_dir outputs_relation --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_relation/pot_dev_1224.txt --predict_file data/EE/ACE_relation/pot_dev_1224.txt \
--test_file data/EE/ACE_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_relation/test --valid_pattern_path ./data/EE/ACE_relation/valid_pattern.json

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
--output_dir outputs_event --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_event/pot_train_1224.txt --predict_file data/EE/ACE_event/pot_dev_1224.txt \
--test_file data/EE/ACE_event/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 12 \
--dataset ACE_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_event/train --valid_pattern_path ./data/EE/ACE_event/valid_pattern.json

python train.py --output_dir outputs_event --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_event/pot_dev_1224.txt --predict_file data/EE/ACE_event/pot_dev_1224.txt \
--test_file data/EE/ACE_event/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_event/test --valid_pattern_path ./data/EE/ACE_event/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_entity/pot_dev_1224.txt --predict_file data/EE/ACE_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_entity/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 4 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
# --dataset ACE_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_entity/train --valid_pattern_path ./data/EE/ACE_entity/valid_pattern.json

# python train.py --output_dir outputs_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_entity/pot_dev_1224.txt --predict_file data/EE/ACE_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_entity/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_entity/test --valid_pattern_path ./data/EE/ACE_entity/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_add_relation --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_relation/pot_dev_1224.txt --predict_file data/EE/ACE_add_relation/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset ACE_ADD_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_relation/train --valid_pattern_path ./data/EE/ACE_add_relation/valid_pattern.json

# python train.py --output_dir outputs_add_relation --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_relation/pot_dev_1224.txt --predict_file data/EE/ACE_add_relation/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE_ADD_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_relation/test --valid_pattern_path ./data/EE/ACE_add_relation/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_add_event --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_dev_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset ACE_ADD_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_event/train --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

# python train.py --output_dir outputs_add_event --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_dev_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE_ADD_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_event/test --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
# --output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_dev_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/train --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json

# python train.py --output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_dev_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/test --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json
