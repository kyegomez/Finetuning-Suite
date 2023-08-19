import logging

import torch
from datasets import load_dataset
from peft import TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
)

from finetuning_suite.base import DefaultPreprocessor
from finetuning_suite.trainer.base import DefaultTrainerConfig

class FineTuner:
    def __init__(self, 
            model_id: str, 
            device: str = None,
            dataset_name=None, 
            lora_r=16,
            lora_alpha=32,
            lora_target_modules=["q", "v"],
            lora_bias="none",
            preprocessor=None,
            lora_task_type=TaskType.SEQ_2_SEQ_LM,
            max_length=1000, 
            quantize: bool = False, 
            quantization_config: dict = None,
            trainer_config=None):
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor if preprocessor else DefaultPreprocessor(self.tokenizer)
        self.trainer_config = trainer_config if trainer_config else DefaultTrainerConfig

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias
        self.lora_task_type = lora_task_type


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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config)
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load the model or the tokenizer: {e}")
            raise

    def __call__(self, prompt_text: str, max_length: int = None):
        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise

    def preprocess_data(self):
        tokenized_dataset = self.dataset.map(self.preprocessor.preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
        return tokenized_dataset
    
    def train(self, output_dir, num_train_epochs):
        self.model, data_collator, training_args = self.trainer_config.configure(self.model, self.tokenizer, output_dir, num_train_epochs)
        
        tokenized_dataset = self.preprocessor_datas()
        trainer = Seq2SeqTrainer(model=self.model, args=training_args, data_collator=data_collator, train_dataset=tokenized_dataset["train"])
        trainer.train()

    def generate(self, prompt_text: str, max_length: int = None):
        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise







