from abc import ABC, abstractmethod
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)


class TrainerConfiguration(ABC):
    @abstractmethod
    def configure(self, model, tokenizer, output_dir, num_train_epochs, *args, **kwargs):
        """
        Configures the model collator, and training arguments

        Returns:
            tuple: (configured model, data_collator, training_args)
        """
        pass


class DefaultTrainerConfig(TrainerConfiguration):

    def configure(self, model, tokenizer, output_dir, num_train_epochs, *args, **kwargs):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,  label_pad_token_id=-100, pad_to_multiple_of=8 )

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

        return model, data_collator, training_args
    