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

num_classes=11003

python $BASE_ROOT/train.py \
    --CMPC \
    --CMPM \
    --CONT \
    --pretrained \
    --model_path $PRETRAINED_PATH \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay} \
    --num_classes ${num_classes} \
    --focal_type $FOCAL_TYPE \
    --feature_size 768 \
    --lambda_cont 0.1 \
    --part2 3 \
    --part3 2 \
    --reranking



