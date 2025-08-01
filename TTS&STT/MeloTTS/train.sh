CONFIG=$1
GPUS=$2
PRETRAIN_G=$3
PRETRAIN_D=$4
PRETRAIN_DUR=$5

MODEL_NAME=$(basename "$(dirname "$CONFIG")")
PORT=10902

# 인자 조건부 처리
PRETRAIN_ARGS=""
[ -n "$PRETRAIN_G" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_G $PRETRAIN_G"
[ -n "$PRETRAIN_D" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_D $PRETRAIN_D"
[ -n "$PRETRAIN_DUR" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_dur $PRETRAIN_DUR"

while :
do
  torchrun --nproc_per_node=$GPUS \
           --master_port=$PORT \
           train.py --config "$CONFIG" \
                    --model "$MODEL_NAME" \
                    $PRETRAIN_ARGS

  for PID in $(ps -aux | grep "$CONFIG" | grep python | awk '{print $2}')
  do
    echo $PID
    kill -9 $PID
  done
  sleep 30
done