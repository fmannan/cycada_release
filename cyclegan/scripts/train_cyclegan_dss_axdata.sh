set -ex
python train.py --dataroot /mnt/data2/dataset/cycada_data/dss_axdata/ --name dss_axdata --model cycle_gan --pool_size 50 --no_dropout
