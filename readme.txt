best result
CV 0.9913 LB 0.9810


CMD RUN:
--patience 4 --LR_SCHEDULER REDUCED --optimizer RADAM --lr 1e-3 --lr_ratio 0.65 --batch_size 128

python train.py --model senet50 --outdir YOUR_OUT_DIR
                --gpu_ids 2,3
                --width 128 --height 128
                --feather_data_path BengaliData/feather_resize128/
                --mixup 1
                --image_mode gray
                --patience 3
                --LR_SCHEDULER REDUCED
                --optimizer RADAM
                --image_mode gray
                --lr 1e-3
                --lr_ratio 0.9
                --batch_size 512



model: which model to use
outdir: model and log save dir
gpu_ids: which gpu will use, gpu index start from 0
width: input image width
height: input image height
feather_data_path: data location for train and val, generate by offline with parquet2feather in data.py
mixup: use cutmix or not
image_mode: input image mode rgb or gray
patience: ReduceLROnPlateau patience
LR_SCHEDULER : which one schedular will use
optimizer: which optimizer will use
lr : learning rate
lr_ratio: ReduceLROnPlateau factor
batch_size: batch_size












