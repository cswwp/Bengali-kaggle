#!/bin/bash

# Rob First experiment with Parker's setup
python train.py \
    --model senet50 \
    --outdir 0305/seresnext50_resize128_gridmask/ \
    --gpu_ids 0,1 \
    --width 128 \
    --height 128  \
    --feather_data_path BengaliData/feather_resize128/ \
    --mixup 0 \
    --image_mode gray \
    --patience 3 \
    --LR_SCHEDULER REDUCED \
    --optimizer RADAM \
    --lr 1e-3 \
    --lr_ratio 0.9 \
    --batch_size 448 \
    --gridmask True \
    --gridmask_num_grid '(3,7)'

