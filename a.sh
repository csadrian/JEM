

i=4

width=10


class_cond_p_x_sample=1
uncond=0

for width in 2
do
for past_weight in 0.0 1.0
do
for warmup_iters in 1000
do
for norm in None
do
for p_x_y_weight in 0.0
do

name=ebr_norm_${norm}_width_${width}_wi_${warmup_iters}_past_${past_weight}_uncond_${uncond}_ccpxs_${class_cond_p_x_sample}_pxy_${p_x_y_weight}

CUDA_VISIBLE_DEVICES=${i} nohup python train_wrn_ebm_cl.py --lr .0001 --dataset cifar100 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight ${p_x_y_weight} --sigma .1 --width ${width} --depth 28 --save_dir out/${name} --plot_uncond --plot_cond --warmup_iters ${warmup_iters} --num_splits 5 --n_epochs 100 --p_y_given_x_past_weight ${past_weight} --class_cond_p_x_sample > ${name}.cout 2> ${name}.cerr &

((i=i+1))
done
done
done
done
done
