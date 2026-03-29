TOTAL_BATCH_SIZE=64
MICRO_TRAIN_BATCH_SIZE=1
MICRO_EVAL_BATCH_SIZE=1
NUM_GPUS=1
SAVE_STEPS=500
EVAL_STEPS=500
GRADIENT_CHECKPOINTING=True
BF16=True
LEARNING_RATE=2e-6
MODEL_MAX_LENGTH=4096
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / MICRO_TRAIN_BATCH_SIZE / NUM_GPUS))

MODEL_NAME_OR_PATH="check/SPC-Critic-0"

DATA_DIR=../data
TRAIN_DATA_PATH=../data/train/converted/combined_train_data.jsonl.gz

OUTPUT_DIR=saved_models/spc_critic_medical
LOGS_PATH=${OUTPUT_DIR}/logs
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH

MASTER_ADDR=${CHIEF_IP:-"127.0.0.1"}
MASTER_PORT=6000
TMP_DIR=${OUTPUT_DIR}/tmp
mkdir -p $TMP_DIR

NODE_IP_LIST=${NODE_IP_LIST:-"127.0.0.1"}
echo $NODE_IP_LIST > ${TMP_DIR}/env.txt

sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts


deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} src/offline_rl.py \
    --output_dir ${OUTPUT_DIR} \
    --do_train True \
    --data_paths ${TRAIN_DATA_PATH} \
    --model_type qwen \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --remove_unused_columns False \
    --report_to tensorboard \
    --overwrite_output_dir True \
    --per_device_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --evaluation_strategy no \
    --warmup_steps 100 \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --bf16 ${BF16} \
    --lm_kl_coeff 0.1 \
    --lm_sft_coeff 0.15 \
    --deepspeed config/ds_config.json
