export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=512

model=PAttn
methods_h='PAttn'
tag_file=main.py

gpu_loc=0
percent=100

pre_lens_h='96 192 336 720'
filename=Electricity.txt

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
lr=0.00005
bs=16

python $tag_file \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id 'Electricity_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --method $method \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done