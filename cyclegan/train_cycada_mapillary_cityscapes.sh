CUDA_VISIBLE_DEVICES=0 python train.py --name cycada_mapillary2cityscapes \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 1 \
    --dataset_mode unaligned --dataroot /mnt/data2/dataset/cycada_data/mapillary_cityscapes/ \
    --which_direction BtoA
#    --num_workers=0

