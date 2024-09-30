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

python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
--output_dir outputs_ace_2 --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_train_1224.txt --predict_file data/EE/ACE/pot_dev_1224.txt \
--test_file data/EE/ACE/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 3 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 \
--dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/EE/pot_train_1224 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

python train.py --output_dir outputs_ace_2 --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE/pot_dev_1224.txt --predict_file data/EE/ACE/pot_dev_1224.txt \
--test_file data/EE/ACE/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/EE/pot_test_1224 --valid_pattern_path ./data/EE/ACE/valid_pattern.json

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
