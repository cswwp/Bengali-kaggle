# Bengali-kaggle

best result

liner5 CV 0.9905 LB 0.9810  (train with cutmix randomshiftrotate and input size is 128x128x1)
Liner1 CV 0.9913 LB 0.9810  (train with cutmix randomshiftrotate and input size is 128x128x1)



CMD RUN:
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




liner5 head + cutmix + rotate inputsize 128x128x1 global_max_recall CV 0.9905 LB 0.9810
liner1 head + cutmix + rotate inputsize 128x128x1 global_max_recall CV 0.9913 LB 0.9810


So the liner5's gap between CV an LB is small, so it should be better model







