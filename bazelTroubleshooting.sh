
# Running the pre-trained model.

export BASE_DIRECTORY=/home/sinadabiri/Testing-In-Silico-Labeling/in-silico-labeling/isl

bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375\
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_b_sample \
  --infer_channel_whitelist DAPI_CONFOCAL,MAP2_CONFOCAL,NFH_CONFOCAL \
  	> Testing_output_preTrainedModel_8-7-18_375crop_16images.txt \
  	2> testing_error_preTrainedModel_8-7-18_375crop_16images.txt


# ==================================================================================
# Training the pre-trained model on a new dataset.

export BASE_DIRECTORY=/home/sinadabiri/Testing-In-Silico-Labeling/in-silico-labeling/isl


bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375 \
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B3 \
  --infer_channel_whitelist DAPI_CONFOCAL,CELLMASK_CONFOCAL \
  --noinfer_simplify_error_panels \
    > testing_output_preTrainedModel_newDataSet_8-7-18_375crop_B3images.txt \
    2> testing_error_preTrainedModel_newDataSet_8-7-18_375crop_B3images.txt

    
# ==================================================================================
# Training

export BASE_DIRECTORY=/home/sinadabiri/Testing-In-Silico-Labeling/in-silico-labeling/isl

bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode TRAIN \
  --metric LOSS \
  --master "" \
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_train_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B2  \
    > testing_output_Training_8-7-18_375crop_16image.txt \
    2> testing_error_Training_8-7-18_375crop_16image.txt

tensorboard --logdir $BASE_DIRECTORY

# ==================================================================================
# To generate predictions:

export BASE_DIRECTORY=/home/sinadabiri/Testing-In-Silico-Labeling/in-silico-labeling/isl


bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375 \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B3 \
  --infer_channel_whitelist DAPI_CONFOCAL,CELLMASK_CONFOCAL \
  --noinfer_simplify_error_panels \
    > testing_output_Predictions_8-7-18_375crop_16image.txt \
    2> testing_error_Predictions_8-7-18_375crop_16image.txt

