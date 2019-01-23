set -ex
python run_model.py --checkpoints_dir=/data6/cycada_data/ckpts/ --outdir=/data6/cycada_data/output_coco_axdata_512/ --resize_or_crop None --dataroot /data6/cycada_data/coco_axdata/ --name coco_axdata_512 --model cycle_gan --phase test --no_dropout
