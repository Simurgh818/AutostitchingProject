bazel run isl:launch -- --alsologtostderr --base_directory home/sinadabiri/venvs/tensorflow \
  --mode EVAL_EVAL --metric INFER_FULL --stitch_crop_size 375 \
   --restore_directory /home/sinadabiri/venvs/tensorflow/in-silico-labeling-master/isl/checkpoints \
     --read_pngs --dataset_eval_directory /home/sinadabiri/venvs/tensorflow/in-silico-labeling-master/isl/condition_b_sample --infer_channel_whitelist DAPI_CONFOCAL,MAP2_CONFOCAL,NFH_CONFOCAL
