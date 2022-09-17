MODEL_DIR='gs://melodi-bucket0/models' \
T5X_DIR='./' \
TFDS_DATA_DIR='gs://melodi-bucket0/tfds_data' \
python3 t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
