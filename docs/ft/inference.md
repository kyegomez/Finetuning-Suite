# Module Name: Inference

The `Inference` class is a part of a custom module that facilitates text generation using a pre-trained causal language model from the Hugging Face `transformers` library. The class provides functionalities for loading a pre-trained model, tokenizing input text, and generating text based on a given prompt. Additionally, it supports quantization of model weights to reduce the model size and accelerate inference.

## Class Definition

```python
class Inference:
    def __init__(
            self, 
            model_id: str, 
            device: str = None, 
            max_length: int = 20, 
            quantize: bool = False, 
            quantization_config: dict = None
        ):
```

### Parameters:

- `model_id` (str): The identifier of the pre-trained model to be loaded. This can be the path to a local directory containing the model files or a model id from the Hugging Face model hub.
- `device` (str, optional): The device on which the model will be loaded and inference will be performed. Default is `None`, which means that it will use CUDA if available, otherwise CPU.
- `max_length` (int, optional): The maximum length of the generated text. Default is 20.
- `quantize` (bool, optional): A flag indicating whether to quantize the model weights. Default is `False`.
- `quantization_config` (dict, optional): A dictionary containing the configuration for quantization. Default is `None`.

## Methods

### `__call__(self, prompt_text: str, max_length: int = None) -> str`

Generates text based on the provided `prompt_text`.

#### Parameters:

- `prompt_text` (str): The text prompt based on which the text will be generated.
- `max_length` (int, optional): The maximum length of the generated text. If not provided, the `max_length` specified during initialization will be used.

#### Returns:

- `str`: The generated text.

### `run(self, prompt_text: str, max_length: int = None) -> str`

This method is an alternative to the `__call__` method and performs the same operation.

#### Parameters:

- `prompt_text` (str): The text prompt based on which the text will be generated.
- `max_length` (int, optional): The maximum length of the generated text. If not provided, the `max_length` specified during initialization will be used.

#### Returns:

- `str`: The generated text.

## Usage Examples:

### Example 1: Basic Usage

```python
from zeta import Inference

model_id = "gpt2-small"
inference = Inference(model_id=model_id)

prompt_text = "Once upon a time"
generated_text = inference(prompt_text)
print(generated_text)
```

### Example 2: Specifying Maximum Length

```python
from zeta import Inference

model_id = "gpt2-small"
inference = Inference(model_id=model_id, max_length=50)

prompt_text = "In a land far, far away"
generated_text = inference.run(prompt_text, max_length=30)
print(generated_text)
```

### Example 3: Using Quantization

```python
from zeta import Inference

model_id = "gpt2-small"
quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': torch.bfloat16
}
inference = Inference(model_id=model_id, quantize=True, quantization_config=quantization_config)

prompt_text = "Once upon a time"
generated_text = inference(prompt_text)
print(generated_text)
```

## Mathematical Formulation:

The `Inference` class uses a pre-trained causal language model for text generation. The probability of each word in the vocabulary is computed using the softmax function:

\[ P(w_i | w_1, ..., w_{i-1}) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} \]

Where:
- \( w_i \) is the ith word in the sequence.
- \( z_i \) is the logit for the ith word in the vocabulary.
- \( V \) is the size of the vocabulary.

The text is generated word by word, where each word is sampled from the probability distribution over the vocabulary computed by the model.

## Limitations:

1. Memory Consumption: Generating text with large models requires a significant amount of GPU memory. It is recommended to use a GPU with at least 16 GB of memory for generating text with large models.

2. Computation Time: Generating text with large models requires a significant amount of computation time. It is recommended to use a powerful GPU to accelerate the inference process.

3. Quantization Accuracy: Quantizing the model weights reduces the model size and accelerates inference, but may also result in a slight decrease in model accuracy. It is recommended to evaluate the quantized model on a validation set to ensure that the accuracy is acceptable for the specific application.

## Conclusion:

The `Inference` class facilitates text generation using pre-trained models from the Hugging Face `transformers` library. This class includes functionalities for loading a pre-trained model, tokenizing input text, and generating text based on a given prompt. It also supports quantization of model weights to reduce the model size and accelerate inference.