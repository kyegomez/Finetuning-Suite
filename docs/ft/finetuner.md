# FineTuner Class

## Overview and Introduction
The FineTuner class is a part of the finetuning suite, designed for fine-tuning pre-trained language models for various natural language processing tasks. This class serves as a wrapper for several functionalities, including data preprocessing, model training, and text generation using a pre-trained model. The main components include the HuggingFace's transformers library, the peft library for task types, and the datasets library for loading datasets. 

The FineTuner class is designed to streamline the process of fine-tuning by handling various configurations and components such as LoRA (Low-Rank Adaptation), quantization, and inference. This is especially important in the context of large-scale models and datasets where fine-tuning needs to be efficient and adaptable.

## Class Definition

```python
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
                trainer_config=None,
                inference_handler=None
            ):
    ...

```

### Parameters:

- `model_id` (str): The identifier of the pre-trained model to be fine-tuned.
- `device` (str, optional): The device to run the model on, either 'cpu' or 'cuda'. Default is 'cuda' if available, otherwise 'cpu'.
- `dataset_name` (str, optional): The name of the dataset to be used for training. Default is `None`.
- `lora_r` (int, optional): The rank of LoRA layers. Default is 16.
- `lora_alpha` (int, optional): The over-parameterization ratio of LoRA layers. Default is 32.
- `lora_target_modules` (list, optional): The target modules for LoRA. Default is ["q", "v"].
- `lora_bias` (str, optional): The bias of LoRA. Default is "none".
- `preprocessor` (Preprocessor, optional): The preprocessor for tokenizing the dataset. Default is `DefaultPreprocessor`.
- `lora_task_type` (TaskType, optional): The task type for LoRA. Default is `TaskType.SEQ_2_SEQ_LM`.
- `max_length` (int, optional): The maximum length of the generated text. Default is 1000.
- `quantize` (bool, optional): Whether to quantize the model weights. Default is `False`.
- `quantization_config` (dict, optional): The configuration for quantization. Default is `None`.
- `trainer_config` (TrainerConfig, optional): The configuration for the trainer. Default is `DefaultTrainerConfig`.
- `inference_handler` (InferenceHandler, optional): The handler for inference. Default is `DefaultInferenceHandler`.

### Methods:

#### `__call__(self, prompt_text: str, max_length: int = None) -> str`

Generates text based on the provided `prompt_text`.

Parameters:
- `prompt_text` (str): The text prompt to base the generation on.
- `max_length` (int, optional): The maximum length of the generated text. Default is `self.max_length`.

Returns:
- `str`: The generated text.

#### `preprocess_data(self)`

Preprocesses the dataset by tokenizing the text and removing unnecessary columns.

Returns:
- `Dataset`: The tokenized dataset.

#### `train(self, output_dir, num_train_epochs)`

Trains the model on the preprocessed dataset.

Parameters:
- `output_dir` (str): The directory to save the trained model.
- `num_train_epochs` (int): The number of epochs to train the model.

#### `generate(self, prompt_text: str, max_length: int = None) -> str`

Generates text based on the provided `prompt_text` using the `inference_handler`.

Parameters:
- `prompt_text` (str): The text prompt to base the generation on.
- `max_length` (int, optional): The maximum length of the generated text. Default is `self.max_length`.

Returns:
- `str`: The generated text.

## Functionality and Usage

### Preprocessing Data
Before training the model, the data needs to be preprocessed. The `preprocess_data` method tokenizes the dataset using the specified preprocessor. It removes the unnecessary columns and returns the tokenized dataset.

### Training the Model
The `train` method trains the model using the specified trainer configuration, preprocessed dataset, and training arguments. It configures the model, tokenizer, and training arguments using the `trainer_config` object, and then initializes the `Seq2SeqTrainer` with the configured model, training arguments, data collator, and training dataset. It then trains the model using the `train` method of the `Seq2SeqTrainer`.

### Generating Text
The `generate` method generates text based on the provided prompt text using the specified inference handler. It calls the `generate` method of the `inference_handler` object with the specified prompt text, model, tokenizer, device, and maximum length.


## Additional Information and Tips

### Quantization
Quantization is the process of converting the weights and biases of the model from floating point to integer values. This is useful for reducing the memory requirements and speeding up the model inference. The `quantize` parameter specifies whether to quantize the model or not, and the `quantization_config` parameter specifies the configuration for quantization. The default quantization configuration is 4-bit quantization with double quantization and "nf4" quantization type.

### Low-Rank Adaptation (LoRA)
LoRA is a method for fine-tuning pre-trained models with a small number of additional parameters. The `lora_r` parameter specifies the rank for LoRA, the `lora_alpha` parameter specifies the scaling factor for LoRA, the `lora_target_modules` parameter specifies the target modules for LoRA, and the `lora_bias` parameter specifies the bias for LoRA.



### Usage Examples:

#### Example 1:

```python
from finetuning_suite import FineTuner
import torch

# Initialize the FineTuner
finetuner = FineTuner(
    model_id="gpt2",
    device="cuda",
    dataset_name="dialogue",
    lora_r=16,
    lora_alpha=32,
    max_length=1000,
    quantize=True
)

# Preprocess the data
tokenized_dataset = finetuner.preprocess_data()

# Train the model
output_dir = "./trained_model"
num_train_epochs = 3
finetuner.train(output_dir, num_train_epochs)

# Generate text
prompt_text = "Once upon a time"
generated_text = finetuner.generate(prompt_text)
print(generated_text)
```

#### Example 2:

```python
from finetuning_suite import FineTuner
import torch

# Initialize the FineTuner with custom quantization configuration
quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': torch.bfloat16
}

finetuner = FineTuner(
    model_id="gpt2",
    device="cuda",
    dataset_name="dialogue",
    max_length=500,
    quantize=True,
    quantization_config=quantization_config
)

# Generate text
prompt_text = "Once upon a time"
generated_text = finetuner.generate(prompt_text)
print(generated_text)
```

#### Example 3:

```python
from finetuning_suite import FineTuner

# Initialize the FineTuner with custom trainer configuration
trainer_config = DefaultTrainerConfig(
    output_dir="./trained_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

finetuner = FineTuner(
    model_id="gpt2",
    device="cuda",
    dataset_name="dialogue",
    max_length=1000,
    trainer_config=trainer_config
)

# Train the model
finetuner.train()

# Generate text
prompt_text = "Once upon a time"
generated_text = finetuner.generate(prompt_text)
print(generated_text)
```

### Mathematical Formulation:

The `FineTuner` class includes several components, each with its own mathematical formulation:

1. LoRA (Low-Rank Adaptation): LoRA is a technique to adapt pre-trained models for new tasks with limited data. Given a weight matrix \(W \in \mathbb{R}^{m \times n}\) of a pre-trained model, LoRA decomposes \(W\) into low-rank and diagonal components:

\[ W = UDV^T + \Delta \]

Where:
- \( U \in \mathbb{R}^{m \times r} \) and \( V \in \mathbb{R}^{n \times r} \) are low-rank matrices.
- \( D \in \mathbb{R}^{r \times r} \) is a diagonal matrix.
- \( \Delta \in \mathbb{R}^{m \times n} \)

 is a residual matrix.

2. Quantization: The `FineTuner` class supports quantizing the model weights to reduce the model size and accelerate inference. The quantization process involves mapping the full-precision weights to a smaller set of quantized values. For example, in 4-bit quantization, the weights are mapped to one of 16 possible values.

3. Text Generation: The `FineTuner` class generates text using a causal language model. Given a prompt text, the model predicts the next word in the sequence until the maximum length is reached or a stop token is generated. The probability of each word in the vocabulary is computed using the softmax function:

\[ P(w_i | w_1, ..., w_{i-1}) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} \]

Where:
- \( w_i \) is the ith word in the sequence.
- \( z_i \) is the logit for the ith word in the vocabulary.
- \( V \) is the size of the vocabulary.

### Implementation Details:

The `FineTuner` class is implemented using the Hugging Face `transformers` library. The `transformers` library provides pre-trained models, tokenizers, and training utilities for natural language processing tasks.

The `FineTuner` class includes the following components:

1. Preprocessing: The `preprocess_data` method tokenizes the dataset using the tokenizer associated with the specified `model_id`. The `DefaultPreprocessor` class is used for tokenization by default, but a custom preprocessor can be specified using the `preprocessor` parameter.

2. Training: The `train` method fine-tunes the model on the preprocessed dataset. The `DefaultTrainer` class from the `transformers` library is used for training by default, but a custom trainer can be specified using the `trainer_config` parameter.

3. Inference: The `generate` method generates text based on a provided `prompt_text`. The `DefaultInferenceHandler` class is used for inference by default, but a custom inference handler can be specified using the `inference_handler` parameter.

4. Quantization: The `FineTuner` class supports quantizing the model weights to reduce the model size and accelerate inference. The `quantize` parameter determines whether to quantize the model weights, and the `quantization_config` parameter specifies the configuration for quantization.

### Limitations:

The `FineTuner` class has some limitations:

1. Memory Consumption: Fine-tuning large models requires a significant amount of GPU memory. It is recommended to use a GPU with at least 16 GB of memory for fine-tuning large models.

2. Computation Time: Fine-tuning large models requires a significant amount of computation time. It is recommended to use a powerful GPU to accelerate the training process.

3. Quantization Accuracy: Quantizing the model weights reduces the model size and accelerates inference, but may also result in a slight decrease in model accuracy. It is recommended to evaluate the quantized model on a validation set to ensure that the accuracy is acceptable for the specific application.


------

## Custom Finetuning, Inference, and Data preprocessing logic

#### `TrainerConfiguration` Abstract Class

The `TrainerConfiguration` abstract class is designed to offer flexibility to users, enabling them to define custom training configurations and strategies to be used with the `FineTuner` class.

- **configure(model, tokenizer, output_dir, num_train_epochs)**: This method is responsible for configuring the model, data collator, and training arguments.

#### Usage:

1. **Extend TrainerConfiguration**:
   
   Create your custom trainer configuration by extending `TrainerConfiguration`:

   ```python
   class MyTrainerConfig(TrainerConfiguration):
       def configure(self, model, tokenizer, output_dir, num_train_epochs):
           ...
           return model, data_collator, training_args
   ```

2. **Pass to FineTuner**:
   
   Once you've defined your configuration, pass it to the `FineTuner`:

   ```python
   my_config = MyTrainerConfig()
   finetuner = FineTuner(model_id='your_model_id', trainer_config=my_config)
   ```

This setup makes the `FineTuner` extremely flexible, accommodating various training strategies and configurations as required by the user.

-----

## Documentation

### Overview

Our system is architected to be modular, making it versatile and customizable at various stages of the fine-tuning process. Three primary components encapsulate the critical steps: 
1. **Preprocessing**: Transform raw input data into a format suitable for training.
2. **Training Configuration**: Set the training parameters, model adjustments, and collators.
3. **Inference**: Generate outputs based on user-provided input.

The entire system relies on abstract base classes, allowing developers to create custom implementations for each of the aforementioned steps.

### 1. Preprocessing 

#### Abstract Base Class: `Preprocessor`
- **Initial Parameters**:
  - `tokenizer`: The tokenizer associated with the model.
  
- **Methods**:
  - `preprocess_function(sample, padding="max_length")`: Transforms the input data sample to a format suitable for model input.

#### Default Implementation: `DefaultPreprocessor`
Converts dialogues into summaries, tokenizes them, and manages padding/truncation.

#### Customization
To create a custom preprocessor, inherit from the `Preprocessor` class and implement the `preprocess_function` method.

### 2. Training Configuration

#### Abstract Base Class: `TrainerConfiguration`
- **Methods**:
  - `configure(model, tokenizer, output_dir, num_train_epochs, *args, **kwargs)`: Configures the model, data collator, and training arguments.

#### Default Implementation: `DefaultTrainerConfig`
Uses LoRA configurations and sets up a `Seq2Seq` collator and training arguments.

#### Customization
Inherit from `TrainerConfiguration` and implement the `configure` method to customize the training setup.

### 3. Inference 

#### Abstract Base Class: `InferenceHandler`
- **Methods**:
  - `generate(prompt_text, model, tokenizer, device, max_length)`: Processes the prompt text and uses the model to generate an output.

#### Default Implementation: `DefaultInferenceHandler`
Encodes the prompt, generates sequences with the model, and decodes the output.

#### Customization
Developers can inherit from `InferenceHandler` and implement the `generate` method to customize the inference logic.

---

### Examples

1. **Custom Preprocessor**:
```python
class MyPreprocessor(Preprocessor):
    def preprocess_function(self, sample, padding="max_length"):
        # Custom preprocessing logic here
        ...
        return processed_sample
```

2. **Custom Trainer Configuration**:
```python
class MyTrainerConfig(TrainerConfiguration):
    def configure(self, model, tokenizer, output_dir, num_train_epochs, *args, **kwargs):
        # Custom training configuration logic here
        ...
        return custom_model, custom_data_collator, custom_training_args
```

3. **Custom Inference Handler**:
```python
class MyInferenceHandler(InferenceHandler):
    def generate(self, prompt_text, model, tokenizer, device, max_length):
        # Custom inference logic here
        ...
        return custom_output
```

### Conclusion
This documentation provides a roadmap for creating custom implementations for preprocessing, training, and inference logic. The modular architecture ensures flexibility and promotes adherence to the open/closed principle, making the system easily extensible without modifying existing code. Ensure your custom classes inherit from the appropriate base class and implement the required methods for seamless integration.






-------



# Custom Preprocesing aDocumentation

### `Preprocessor` Abstract Class Documentation

#### Overview:
The `Preprocessor` abstract class serves as a blueprint for custom data preprocessing strategies to be used with the `FineTuner` class. The primary goal is to provide a polymorphic structure that enables users to create their custom preprocessing functions while adhering to the established interface.

#### Structure:
- The class contains a single abstract method, `preprocess_function`, that subclasses must implement.
- An optional tokenizer can be passed during initialization and used within the preprocessing method if needed.

#### Rules for extending:
1. **Mandatory Implementation**: Any class extending the `Preprocessor` must provide a concrete implementation of the `preprocess_function`.
2. **Method Signature**: The `preprocess_function` must have the same signature across all implementations: `(sample, padding="max_length")`.
3. **Return Type**: The `preprocess_function` should return a dictionary compatible with the transformer's model input. Typically, this includes tokenized input sequences and associated labels.
4. **Use Tokenizer Judiciously**: While the tokenizer is provided and can be used within the preprocess function, it's essential to remember that different tokenizers may have distinct properties and methods. Ensure compatibility.
5. **Ensure Padding Compatibility**: Since padding is a parameter, make sure to handle different padding strategies like `max_length`, `longest`, etc.

#### Example:

```python
from abc import ABC, abstractmethod

class Preprocessor(ABC):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_function(self, sample, padding="max_length"):
        pass
```

#### Usage:
To use the `Preprocessor` class, follow the steps below:

1. **Create a Custom Preprocessor**:
    Extend the `Preprocessor` abstract class and implement the `preprocess_function` according to your requirements.

   ```python
   class CustomPreprocessor(Preprocessor):
       def preprocess_function(self, sample, padding="max_length"):
           # Your custom preprocessing logic here
           pass
   ```

2. **Pass to FineTuner**:
   Instantiate your custom preprocessor and pass it to the `FineTuner` during initialization.

   ```python
   custom_preprocessor = CustomPreprocessor(tokenizer=YourTokenizer)
   finetuner = FineTuner(model_id='your_model_id', preprocessor=custom_preprocessor)
   ```

By adhering to the outlined structure and rules, you ensure that custom preprocessing functions are easily integrated into the existing pipeline and remain compatible with the overall training and generation processes.

--- 

This documentation provides a concise overview, rules, and guidelines for effectively using and extending the `Preprocessor` abstract class.

---

# Custom Training Logic
### 1. The `TrainerConfiguration` Abstract Class

Let's remove the specifics of the Lora config and collator, and instead provide one abstract method that allows for configuring the model and trainer.

```python
from abc import ABC, abstractmethod

class TrainerConfiguration(ABC):

    @abstractmethod
    def configure(self, model, tokenizer, output_dir, num_train_epochs):
        """Configures the model, collator, and training arguments.
        
        Returns:
            tuple: (configured_model, data_collator, training_args)
        """
        pass
```

### 2. Default Implementation

A simple default implementation can retain the previous configurations:

```python
class DefaultTrainerConfig(TrainerConfiguration):
    
    def configure(self, model, tokenizer, output_dir, num_train_epochs):
        # LoraConfig (just as an example)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)

        # DataCollator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

        # Training arguments
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
```

### 3. Integration with `FineTuner`

Incorporate the `TrainerConfiguration` into `FineTuner`:

```python
class FineTuner:
    
    def __init__(self, model_id: str, device: str = None, dataset_name=None, trainer_config=None, ...):
        ...
        self.trainer_config = trainer_config if trainer_config else DefaultTrainerConfig()

    ...

    def train(self, output_dir, num_train_epochs):
        self.model, data_collator, training_args = self.trainer_config.configure(self.model, self.tokenizer, output_dir, num_train_epochs)
        
        tokenized_dataset = self.preprocess_data(512, 150)
        trainer = Seq2SeqTrainer(model=self.model, args=training_args, data_collator=data_collator, train_dataset=tokenized_dataset["train"])
        trainer.train()
```

---

### Conclusion:

The `FineTuner` class in the `finetuning_suite` module of the `zeta` library facilitates the fine-tuning of pre-trained models for causal language modeling tasks using the Hugging Face `transformers` library. This class includes functionalities for data preprocessing, model training, and text generation. It also supports quantizing the model weights to reduce the model size and accelerate inference.