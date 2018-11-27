set -ex
python run_model.py --outdir=/mnt/data2/dataset/output_dss_axdata/ --resize_or_crop None --dataroot /mnt/data2/dataset/cycada_data/dss_axdata/ --name dss_axdata --model cycle_gan --phase test --no_dropout
