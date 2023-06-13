# SOTA FINETUNING GUIDE

Fine-tuning Multi-Modality Models for High Performance
This guide provides a step-by-step process for fine-tuning any multi-modality model to achieve state-of-the-art (SOTA) performance using the Finetuning-Suite. The guide covers three methods: GPTQ, GGML, and PEFT or QLORA.

Prerequisites
Cloud computing environment (e.g., Google Colab, Runpod, Lambda Labs)
AutoGPTQ for creating GPTQ models
Convert.py script and quantize binary for GGML models (from llama.cpp or the GGML repo)
Quickstart
Load a model in 4-bit by passing the argument load_in_4bit=True when calling the from_pretrained method and providing a device map (pass "auto" to get a device map that will be automatically inferred).
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
Copy code
Configure the BitsAndBytesConfig for the desired quantization settings.
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
Copy code
GPTQ, GGML, and PEFT or QLORA Methods
Follow the instructions provided in the AutoGPTQ repository for each method:

GPTQ
GGML
PEFT or QLORA
Fine-tuning and Evaluation
Prepare your dataset and preprocess it using the Finetuning-Suite.

Fine-tune your model using the desired method (GPTQ, GGML, or PEFT/QLORA).

Evaluate the fine-tuned model using the provided evaluation metrics and visualization tools.

For a more detailed guide on fine-tuning and evaluation, refer to the Finetuning-Suite README.md.

Additional Resources
AutoGPTQ Repository
GGML Repository
PEFT or QLORA Repository
