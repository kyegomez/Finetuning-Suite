#conduct
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id="Kosmos-X"


#stacked
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

text = "What is your theory of everything?"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model_4bit.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
















# model_id = "your model"

# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(model)

# text = "Hello my name is"
# device = "cuda:0"

# inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))




# #v2
# import torch
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_cd_bf16 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

# outputs = model_cd_bf16.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))




# ####v3
# #The 4bit integration comes with 2 different quantization types: FP4 and NF4. The NF4 dtype stands for Normal Float 4 and is introduced in the [QLoRA paper](https://arxiv.org/abs/2305.14314)
# from transformers import BitsAndBytesConfig

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
# )

# model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)



# #nested quantization for more memory efficient inference and training
# from transformers import BitsAndBytesConfig

# double_quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
# )

# model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
# outputs = model_double_quant.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# #everything
# import torch
# from transformers import BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# outputs = model_4bit.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))






