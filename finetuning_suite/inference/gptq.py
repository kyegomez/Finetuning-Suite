import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

class GPTQInference:
    def __init__(
        self,
        model_id,
        quantization_config_bits: int = 4,
        quantization_config_dataset: str = None,
        max_length: int = 500
    ):
        self.model_id = model_id
        self.quantization_config_bits = quantization_config_bits
        self.quantization_config_dataset = quantization_config_dataset
        self.max_length = max_length


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.quantization_config = GPTQConfig(
            bits=self.quantization_config_bits,
            dataset=quantization_config_dataset,
            tokenizer=self.tokenizer
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config
        )

    def generate(
            self, 
            prompt: str,
            max_length: int = None
        ):
        max_length = max_length if max_length else self.max_length

        try:
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True
                )

            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
        
        except Exception as error:
            print(f"Error: {error} in inference mode, please change the inference logic or try again")
            raise