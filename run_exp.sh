#!/bin/bash

i=7
class_cond_p_x_sample=1
uncond=0
depth=2
width=2
past_weight=1.0
warmup_iters=1000
norm="batch"
p_x_y_weight=0.0
net_type="convnet"

name=ebr_norm_${norm}_width_${width}_wi_${warmup_iters}_past_${past_weight}_uncond_${uncond}_ccpxs_${class_cond_p_x_sample}_pxy_${p_x_y_weight}

CUDA_VISIBLE_DEVICES=${i} nohup python train_wrn_ebm_cl.py --net_type ${net_type} --lr .0001 --dataset cifar100 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight ${p_x_y_weight} --sigma .1 --width ${width} --depth ${depth} --save_dir out/${name} --plot_uncond --plot_cond --warmup_iters ${warmup_iters} --num_splits 5 --n_epochs 100 --p_y_given_x_past_weight ${past_weight} --class_cond_p_x_sample > out/${name}.cout 2> out/${name}.cerr &
