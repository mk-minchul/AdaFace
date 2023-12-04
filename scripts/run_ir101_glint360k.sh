python main.py \
    --data_root /media/turingvideo/data2/datasets/glint360k \
    --train_data_path glint360k \
    --val_data_path glint360k \
    --prefix ir101_glint360k_adaface \
    --use_wandb \
    --gpus 4 \
    --use_16bit \
    --arch ir_101 \
    --batch_size  128\
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2
