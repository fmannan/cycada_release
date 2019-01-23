set -ex
python run_model.py --display_env=eval_mapillary_axdata512 --resize_or_crop None --checkpoints_dir=/data6/cycada_data/ckpts/ --outdir=/data6/cycdata_data/output_mapillary_axdata_512/  --dataroot /data6/cycada_data/mapillary_axdata/ --name mapillary_axdata_512 --model cycle_gan --phase test --no_dropout
# --loadSize=542 --fineSize=512 
# --resize_or_crop None
