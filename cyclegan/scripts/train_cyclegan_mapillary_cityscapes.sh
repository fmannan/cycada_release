set -ex
python train.py --dataroot /mnt/data2/dataset/cycada_data/mapillary_cityscapes/ --name mapillary_cityscapes_cyclegan --model cycle_gan --pool_size 50 --no_dropout
