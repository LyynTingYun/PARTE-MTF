# export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u main.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path ETTm1.csv \
        --model_id PARTE \
        --model PARTE \
        --data ETTm1 \
        --features M \
        --seq_len 336 \
        --pred_len $pred_len \
        --freq m \
        --e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 32 \
        --d_ff 32 \
        --n_heads 16 \
        --kernel_size 200 \
        --seg_len 24 \
        --p_tmask 0.2 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 64 \
        --use_tda \
        --seed 3000
done