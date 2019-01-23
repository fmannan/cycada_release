set -ex
python run_model.py --checkpoints_dir=/data6/cycada_data/ckpts/ --outdir=/data6/cycdata_data/output_dss_axdata_512/ --resize_or_crop None --dataroot /data6/cycada_data/dss_axdata/ --name dss_axdata_512 --model cycle_gan --phase test --no_dropout
