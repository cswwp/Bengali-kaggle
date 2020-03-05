# Rob First experiment with Parker's setup
python train.py \
    --model senet50 \
    --outdir 0304/seresnext50_resize224_rob1/ \
    --gpu_ids 0,1 \
    --width 224 \
    --height 224  \
    --feather_data_path BengaliData/feather_resize224/ \
    --mixup 1 \
    --image_mode gray \
    --patience 2 \
    --LR_SCHEDULER REDUCED \
    --optimizer RADAM \
    --lr 1e-3 \
    --lr_ratio 0.9 \
    --batch_size 128