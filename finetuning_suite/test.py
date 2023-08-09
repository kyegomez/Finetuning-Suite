import logging
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

class PEFTFineTuner:
    def __init__(self, model_id: str, dataset_name: str = "samsum", device: str = None, max_length: int = 20, quantize: bool = False, quantization_config: dict = None):
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.max_length = max_length
        self.dataset_name = dataset_name

        # Load dataset and tokenizer
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        bnb_config = None
        if quantize:
            if not quantization_config:
                quantization_config = {
                    'load_in_4bit': True,
                    'bnb_4bit_use_double_quant': True,
                    'bnb_4bit_quant_type': "nf4",
                    'bnb_4bit_compute_dtype': torch.bfloat16
                }
            bnb_config = BitsAndBytesConfig(**quantization_config)

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, quantization_config=bnb_config)
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load the model: {e}")
            raise

    def preprocess_data(self, max_source_length, max_target_length):
        def preprocess_function(sample, padding="max_length"):
            inputs = ["summarize: " + item for item in sample["dialogue"]]
            model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
            labels = self.tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = self.dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
        return tokenized_dataset

    def train(self, output_dir, num_train_epochs):
        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.model = prepare_model_for_int8_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        # Data Collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=-100, pad_to_multiple_of=8)

        # Training Args
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            report_to="tensorboard"
        )
        tokenized_dataset = self.preprocess_data(512, 150)  # Here, you might want to determine max_source_length and max_target_length dynamically.
        trainer = Seq2SeqTrainer(model=self.model, args=training_args, data_collator=data_collator, train_dataset=tokenized_dataset["train"])
        trainer.train()

    def generate(self, prompt_text, max_length=None):
        max_length = max_length if max_length else self.max_length
        inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage example:
# finetuner = PEFTFineTuner(model_id="google
