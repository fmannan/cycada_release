set -ex
python train.py --display_env=dss_axdata_512 --loadSize=542 --fineSize=512 --dataroot /data6/cycada_data/dss_axdata --name dss_axdata_512 --model cycle_gan --pool_size 50 --no_dropout --display_port 6006 --checkpoints_dir=/data6/cycada_data/ckpts/

