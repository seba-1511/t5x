MODEL_DIR='./models' \
T5X_DIR='./' \
TFDS_DATA_DIR='/media/seba-1511/OCZ/data/' \
python3 t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
