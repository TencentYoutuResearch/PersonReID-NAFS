GPUS=
export CUDA_VISIBLE_DEVICES=$GPUS

IMAGE_DIR=
BASE_ROOT=
ANNO_DIR=$BASE_ROOT/data/processed_data

CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
PRETRAINED_PATH=$BASE_ROOT/pretrained/resnet50-19c8e357.pth
FOCAL_TYPE=none

lr=0.0011
num_epoches=60
batch_size=64
lr_decay_ratio=0.9
epoches_decay=20_30_40

python ${BASE_ROOT}/test.py \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_start 10 \
    --checkpoint_dir $CKPT_DIR \
    --feature_size 768 \
    --focal_type $FOCAL_TYPE \
    --part2 3 \
    --part3 2 \
    --reranking



