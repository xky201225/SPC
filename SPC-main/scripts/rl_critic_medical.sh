TOTAL_BATCH_SIZE=64
MICRO_TRAIN_BATCH_SIZE=1
MICRO_EVAL_BATCH_SIZE=1
NUM_GPUS=1
SAVE_STEPS=100
EVAL_STEPS=100
GRADIENT_CHECKPOINTING=True
BF16=True
LEARNING_RATE=2e-6
MODEL_MAX_LENGTH=2048
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / MICRO_TRAIN_BATCH_SIZE / NUM_GPUS))

MODEL_NAME_OR_PATH="check/SPC-Critic-2"

DATA_DIR="../data/train"
MEDQA_DATA_PATH="${DATA_DIR}/MedQA_spc_critic_train.part01.json"
PUBMEDQA_DATA_PATH="${DATA_DIR}/PubMedQA_spc_critic_train.part01.json"

OUTPUT_DIR=saved_models/SPC-Critic-3-Medical
LOGS_PATH=${OUTPUT_DIR}/logs
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH

echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Base Model: ${MODEL_NAME_OR_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Model Max Length: ${MODEL_MAX_LENGTH}"
echo "Training Data (Medical Only):"
echo "  - ${MEDQA_DATA_PATH}"
echo "  - ${PUBMEDQA_DATA_PATH}"
echo "=========================================="

for f in "${MEDQA_DATA_PATH}" "${PUBMEDQA_DATA_PATH}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        echo "Current directory: $(pwd)"
        echo "Looking for files in: $(dirname "$f")"
        ls -la "$(dirname "$f")" 2>/dev/null || echo "Directory not found"
        exit 1
    fi
done

deepspeed --num_gpus=1 src/offline_rl.py \
    --output_dir ${OUTPUT_DIR} \
    --do_train True \
    --data_paths ${MEDQA_DATA_PATH} ${PUBMEDQA_DATA_PATH} \
    --model_type qwen \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --remove_unused_columns False \
    --report_to tensorboard \
    --overwrite_output_dir True \
    --per_device_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs 3 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --eval_strategy no \
    --warmup_steps 10 \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --bf16 ${BF16} \
    --lm_kl_coeff 0.1 \
    --lm_sft_coeff 0.15 \
    --deepspeed config/ds_config.json

echo "=========================================="
echo "Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
