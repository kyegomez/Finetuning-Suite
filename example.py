from finetuning_suite import FineTuner

model_id="google/flan-t5-xxl"

dataset_name="samsum"

finetune = FineTuner(
    model_id=model_id,
    dataset_name="samsum",
    max_length=150,
    lora_r=16,
    lora_alpha=32,
    quantize=True
)


finetune.train