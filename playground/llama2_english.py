from datasets import load_dataset
from transformers import AutoTokenizer

from finetuning_suite.finetuner import FineTuner

tokenizer = AutoTokenizer.from_pretrained("Phind/Phind-CodeLlama-34B-v1")

def data_preprocessing(dataset="Abirate/english_quotes"):
    data = load_dataset(dataset)
    data = data.map(
        lambda samples: tokenizer(samples["quote"]), batched=True
    )


def trainer(model):
    import transformers

    # needed for gpt-neo-x tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data_preprocessing["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


FineTuner(
    model_id="Phind/Phind-CodeLlama-34B-v1",
    preprocessor=data_preprocessing,
    trainer_config=trainer
)

