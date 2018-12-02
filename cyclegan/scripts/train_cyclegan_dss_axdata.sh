set -ex
python train.py --dataroot /data6/cycada_data/dss_axdata --name dss_axdata2 --model cycle_gan --pool_size 50 --no_dropout --display_port 6006 --checkpoints_dir=/data6/cycada_data/ckpts/

