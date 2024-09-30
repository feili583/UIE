# python train.py --output_dir outputs_ace_add --model_type bert --config_name bert-base-cased \
# --model_name_or_path bert-base-cased --train_file data/EE/ACE_add/pot_train.txt --predict_file data/EE/ACE_add/pot_dev.txt \
# --test_file data/EE/ACE_add/pot_test.txt --max_seq_length 64 --per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 \
# --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add.txt --valid_pattern_path ./data/EE/ACE_add/valid_pattern.json

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="10.126.62.79" --master_port=12345 train.py \
--output_dir outputs_ace_add --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_train_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
--test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 3 --gradient_accumulation_steps 16 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 \
--dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add_train_event_1224 --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

python train.py --output_dir outputs_ace_add --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_dev_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
--test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 64 --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 3 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset JSON_ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee_add_test_event_1224 --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json
