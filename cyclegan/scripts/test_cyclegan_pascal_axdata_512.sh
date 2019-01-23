set -ex
python run_model.py --display_env=main --checkpoints_dir=/data6/cycada_data/ckpts/ --outdir=/data6/cycada_data/output_pascal_axdata_512/ --resize_or_crop None --dataroot /data6/cycada_data/voc_axdata/ --name pascal_axdata_512 --model cycle_gan --phase test --no_dropout
