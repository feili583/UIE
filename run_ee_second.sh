python train.py --output_dir outputs_ee_second --model_type bert --config_name bert-base-cased \
--model_name_or_path bert-base-cased --train_file data/EE/pot_new_second/train.txt \
--predict_file data/EE/pot_new_second/dev.txt --test_file data/EE/pot_new_second/test.txt \
--max_seq_length 64 --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 48 --do_train \
--do_predict \--learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 \
--dataset ACE --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine \
--latent_size 1 --seed 12345 --save_predict_path predict_file/EE/ee.txt