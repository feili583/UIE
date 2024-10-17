# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_entity/pot_train_1224.txt --predict_file data/EE/ACE_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset ACE_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_entity/train --valid_pattern_path ./data/EE/ACE_entity/valid_pattern.json

# python train.py --output_dir outputs_entity --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/EE/ACE_entity/pot_dev_1224.txt --predict_file data/EE/ACE_entity/pot_dev_1224.txt \
# --test_file data/EE/ACE_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ACE_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_entity/test --valid_pattern_path ./data/EE/ACE_entity/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_14lap --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/14lap/pot/train.txt --predict_file data/absa/14lap/pot/dev.txt \
# --test_file data/absa/14lap/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/14lap/train --valid_pattern_path data/absa/14lap/pot/valid_pattern.json

# python train.py --output_dir outputs_14lap --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/14lap/pot/dev.txt --predict_file data/absa/14lap/pot/dev.txt \
# --test_file data/absa/14lap/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/14lap/test --valid_pattern_path data/absa/14lap/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_14res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/14res/pot/train.txt --predict_file data/absa/14res/pot/dev.txt \
# --test_file data/absa/14res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/14res/train --valid_pattern_path data/absa/14res/pot/valid_pattern.json

# python train.py --output_dir outputs_14res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/14res/pot/dev.txt --predict_file data/absa/14res/pot/dev.txt \
# --test_file data/absa/14res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/14res/test --valid_pattern_path data/absa/14res/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_15res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/15res/pot/train.txt --predict_file data/absa/15res/pot/dev.txt \
# --test_file data/absa/15res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/15res/train --valid_pattern_path data/absa/15res/pot/valid_pattern.json

# python train.py --output_dir outputs_15res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/15res/pot/dev.txt --predict_file data/absa/15res/pot/dev.txt \
# --test_file data/absa/15res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/15res/test --valid_pattern_path data/absa/15res/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_16res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/16res/pot/train.txt --predict_file data/absa/16res/pot/dev.txt \
# --test_file data/absa/16res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/16res/train --valid_pattern_path data/absa/16res/pot/valid_pattern.json

# python train.py --output_dir outputs_16res --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/absa/16res/pot/dev.txt --predict_file data/absa/16res/pot/dev.txt \
# --test_file data/absa/16res/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset 14lap --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/16res/test --valid_pattern_path data/absa/16res/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_ace04 --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/ace2004/pot/train.txt --predict_file data/ace2004/pot/dev.txt \
# --test_file data/ace2004/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset ace2004 --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ace2004/train --valid_pattern_path data/ace2004/pot/valid_pattern.json

# python train.py --output_dir outputs_ace04 --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/ace2004/pot/dev.txt --predict_file data/ace2004/pot/dev.txt \
# --test_file data/ace2004/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset ace2004 --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/ace2004/test --valid_pattern_path data/ace2004/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_cadec --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/cadec/pot/train.txt --predict_file data/cadec/pot/dev.txt \
# --test_file data/cadec/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset cadec --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/cadec/train --valid_pattern_path data/cadec/pot/valid_pattern.json

# python train.py --output_dir outputs_cadec --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/cadec/pot/dev.txt --predict_file data/cadec/pot/dev.txt \
# --test_file data/cadec/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset cadec --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/cadec/test --valid_pattern_path data/cadec/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_conll --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/conll2003/train.txt --predict_file data/conll2003/dev.txt \
# --test_file data/conll2003/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset CONLL --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll2003/train --valid_pattern_path data/conll2003/valid_pattern.json

# python train.py --output_dir outputs_conll --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/conll2003/dev.txt --predict_file data/conll2003/dev.txt \
# --test_file data/conll2003/test.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset CONLL --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll2003/test --valid_pattern_path data/conll2003/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_casie --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/event/casie/pot/train.txt --predict_file data/event/casie/pot/dev.txt \
# --test_file data/event/casie/pot/test.txt --max_seq_length 64 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset casie --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/casie/train --valid_pattern_path data/event/casie/pot/valid_pattern.json

# python train.py --output_dir outputs_casie --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/event/casie/pot/dev.txt --predict_file data/event/casie/pot/dev.txt \
# --test_file data/event/casie/pot/test.txt --max_seq_length 64 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset casie --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/casie/test --valid_pattern_path data/event/casie/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12346 train.py \
# --output_dir outputs_genia --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/genia/train.data --predict_file data/genia/dev.data \
# --test_file data/genia/test.data --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
# --dataset GENIA --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/genia/train --valid_pattern_path data/genia/valid_pattern.json

# python train.py --output_dir outputs_genia --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/genia/dev.data --predict_file data/genia/dev.data \
# --test_file data/genia/test.data --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset GENIA --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/genia/test --valid_pattern_path data/genia/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12347 train.py \
# --output_dir outputs_conll04 --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/conll04/pot/train.txt --predict_file data/rel/conll04/pot/dev.txt \
# --test_file data/rel/conll04/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset conll04 --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll04/train --valid_pattern_path data/rel/conll04/pot/valid_pattern.json

# python train.py --output_dir outputs_conll04 --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/conll04/pot/dev.txt --predict_file data/rel/conll04/pot/dev.txt \
# --test_file data/rel/conll04/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset conll04 --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/conll04/test --valid_pattern_path data/rel/conll04/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12348 train.py \
# --output_dir outputs_nyt --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/nyt/pot/train.txt --predict_file data/rel/nyt/pot/dev.txt \
# --test_file data/rel/nyt/pot/test.txt --max_seq_length 80 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset nyt --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/nyt/train --valid_pattern_path data/rel/nyt/pot/valid_pattern.json

# python train.py --output_dir outputs_nyt --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/nyt/pot/dev.txt --predict_file data/rel/nyt/pot/dev.txt \
# --test_file data/rel/nyt/pot/test.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset nyt --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/nyt/test --valid_pattern_path data/rel/nyt/pot/valid_pattern.json

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
# --output_dir outputs_scierc --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/scierc/pot/train.txt --predict_file data/rel/scierc/pot/dev.txt \
# --test_file data/rel/scierc/pot/test.txt --max_seq_length 64 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 \
# --do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 6 \
# --dataset scierc --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/scierc/train --valid_pattern_path data/rel/scierc/pot/valid_pattern.json

# python train.py --output_dir outputs_scierc --model_type bert --config_name ./bert-base-cased \
# --model_name_or_path ./bert-base-cased --train_file data/rel/scierc/pot/dev.txt --predict_file data/rel/scierc/pot/dev.txt \
# --test_file data/rel/scierc/pot/test.txt --max_seq_length 64 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
# --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
# --dataset scierc --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
# --latent_size 1 --seed 12345 --save_predict_path predict_file/scierc/test --valid_pattern_path data/rel/scierc/pot/valid_pattern.json


python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12349 train.py \
--output_dir outputs_add_relation --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_relation/pot_train_1224.txt --predict_file data/EE/ACE_add_relation/pot_dev_1224.txt \
--test_file data/EE/ACE_add_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
--dataset ACE_ADD_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_relation/train --valid_pattern_path ./data/EE/ACE_add_relation/valid_pattern.json

python train.py --output_dir outputs_add_relation --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_relation/pot_dev_1224.txt --predict_file data/EE/ACE_add_relation/pot_dev_1224.txt \
--test_file data/EE/ACE_add_relation/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_ADD_RELATION --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_relation/test --valid_pattern_path ./data/EE/ACE_add_relation/valid_pattern.json

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
--output_dir outputs_add_event --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_train_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
--test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
--dataset ACE_ADD_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_event/train --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

python train.py --output_dir outputs_add_event --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_event/pot_dev_1224.txt --predict_file data/EE/ACE_add_event/pot_dev_1224.txt \
--test_file data/EE/ACE_add_event/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_ADD_EVENT --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_event/test --valid_pattern_path ./data/EE/ACE_add_event/valid_pattern.json

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
--output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_train_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
--test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_train --learning_rate 3e-5 --num_train_epochs 50 --overwrite_output_dir --save_steps 1000 --gradient_accumulation_steps 3 \
--dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/train --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json

python train.py --output_dir outputs_add_entity --model_type bert --config_name ./bert-base-cased \
--model_name_or_path ./bert-base-cased --train_file data/EE/ACE_add_entity/pot_dev_1224.txt --predict_file data/EE/ACE_add_entity/pot_dev_1224.txt \
--test_file data/EE/ACE_add_entity/pot_test_1224.txt --max_seq_length 80 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 \
--do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE_ADD_ENTITY --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/ACE_add_entity/test --valid_pattern_path ./data/EE/ACE_add_entity/valid_pattern.json
