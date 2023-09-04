# Documentation for the GPTQInference Class

## Introduction

`GPTQInference` is a class designed to leverage the capabilities of pre-trained GPT-like models from the HuggingFace transformers library, while also incorporating quantization for efficient inference. Quantization reduces the model size and inference time by using a smaller number of bits to represent the weights, which can be especially beneficial for deployment on resource-constrained devices.

## Class Definition

```python
class GPTQInference:
    def __init__(
        self,
        model_id: str,
        quantization_config_bits: int = 4,
        quantization_config_dataset: str = None,
        max_length: int = 500
    ):
```

### Parameters:

- `model_id (str)`: Identifier for the pre-trained model to be loaded. This typically corresponds to model names or paths in the HuggingFace Model Hub.
  
- `quantization_config_bits (int, default=4)`: Number of bits used for quantization. By default, it uses 4 bits.
  
- `quantization_config_dataset (str, default=None)`: Dataset identifier for the quantization process. If provided, the dataset is used to fine-tune the quantization parameters.
  
- `max_length (int, default=500)`: The maximum length of the generated sequences.

## Functionality and Usage

### Initialization

Upon instantiation, the class:

1. Loads the tokenizer corresponding to the provided model identifier.
2. Initializes a quantization configuration with the given parameters and the loaded tokenizer.
3. Loads the model for causal language modeling based on the provided model identifier and attaches the quantization configuration to it.

### Generation

The `generate` method provides a way to produce text based on a given prompt:

```python
    def generate(self, prompt: str):
```

#### Parameters:

- `prompt (str)`: The input string based on which the model will generate a continuation or completion.

#### Returns:

- `str`: The generated text continuation.

### How It Works:

1. The prompt is tokenized using the loaded tokenizer and converted into a tensor.
2. The tensor is then passed to the model's `generate` method to produce a sequence of token IDs.
3. The generated token IDs are decoded to produce the final text.

Note: In the case of any exceptions during the generation, an error message is printed and the exception is raised.

## Usage Examples

### Example 1: Basic Usage

```python
from zeta import GPTQInference

model_id = "gpt2-medium"
inference_engine = GPTQInference(model_id)
output_text = inference_engine.generate("Once upon a time")
print(output_text)
```

### Example 2: Using Custom Quantization Bits

```python
from zeta import GPTQInference

model_id = "gpt2-medium"
inference_engine = GPTQInference(model_id, quantization_config_bits=2)
output_text = inference_engine.generate("The future of AI is")
print(output_text)
```

### Example 3: Specifying a Dataset for Quantization

```python
from zeta import GPTQInference

model_id = "gpt2-medium"
inference_engine = GPTQInference(model_id, quantization_config_dataset="my_dataset")
output_text = inference_engine.generate("The beauty of nature is")
print(output_text)
```

## Mathematical Formulation

Quantization is a process that involves mapping a continuous or large set of values to a finite range. For a weight \( w \) in the neural network, the quantized weight \( w_q \) can be represented as:

\[ w_q = Q(w, B) \]

Where:
- \( Q \) is the quantization function.
- \( B \) represents the number of bits used for quantization, which in our case is given by `quantization_config_bits`.

This process ensures that the model size is reduced and the inference becomes faster, albeit at the potential cost of some loss in precision.

## Additional Tips

- While quantization can speed up model inference and reduce model size, it may also result in a slight degradation of model performance. It's always a good practice to evaluate the quantized model's performance on a validation set.
  
- If you encounter unexpected errors during inference, ensure that the `model_id` provided corresponds to a valid pre-trained model in the HuggingFace Model Hub.

## References and Resources

- [HuggingFace Transformers Library](https://huggingface.co/transformers/)
  
