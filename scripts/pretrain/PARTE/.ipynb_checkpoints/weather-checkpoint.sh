# export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u main.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path weather.csv \
        --model_id PARTE \
        --model PARTE \
        --data weather \
        --features M \
        --seq_len 336 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --d_model 16 \
        --d_ff 128 \
        --n_heads 4 \
        --kernel_size 200 \
        --seg_len 24 \
        --p_tmask 0.2 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        --use_tda \
        --seed 5000
done