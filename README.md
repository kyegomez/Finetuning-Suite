[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


![Finetuning suite logo](ft-logo.png)

Finetune any model with unparalled performance, speed, and reliability using Qlora, BNB, Lora, Peft in less than 30 seconds, just press GO.


# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Zeta for you.


## 📦 Installation 📦

```bash
$ pip3 install finetuning-suite
```

## 🚀 Quick Start 🚀

```python
from finetuning_suite import FineTuner

# Initialize the fine tuner
model_id="google/flan-t5-xxl"
dataset_name = "samsung"

tuner = FineTuner(
    model_id=model_id,
    dataset_name=dataset_name,
    max_length=150,
    lora_r=16,
    lora_alpha=32,
    quantize=True
)

# Generate content
prompt_text = "Summarize this idea for me."
print(tuner(prompt_text))
```

## 🎉 Features 🎉

- **World-Class Quantization**: Get the most out of your models with top-tier performance and preserved accuracy! 🏋️‍♂️
  
- **Automated PEFT**: Simplify your workflow! Let our toolkit handle the optimizations. 🛠️

- **LoRA Configuration**: Dive into the potential of flexible LoRA configurations, a game-changer for performance! 🌌

- **Seamless Integration**: Designed to work seamlessly with popular models like LLAMA, Falcon, and more! 🤖







## 🛣️ Roadmap 🛣️

Here's a sneak peek into our ambitious roadmap! We're always evolving, and your feedback and contributions can shape our journey! ✨

- [ ] **More Example Scripts**:
  - [ ] Using GPT models
  - [ ] Transfer learning examples
  - [ ] Real-world application samples

- [ ] **Polymorphic Preprocessing Function**:
  - [ ] Design a function to handle diverse datasets
  - [ ] Integrate with known dataset structures from popular sources
  - [ ] Custom dataset blueprint for user-defined structures

- [ ] **Extended Model Support**:
  - [ ] Integration with Lama, Falcon, etc.
  - [ ] Support for non-English models

- [ ] **Comprehensive Documentation**:
  - [ ] Detailed usage guide
  - [ ] Best practices for fine-tuning
  - [ ] Benchmarks for quantization and LoRA features
  
- [ ] **Interactive Web Interface**:
  - [ ] GUI for easy fine-tuning
  - [ ] Visualization tools for model insights

- [ ] **Advanced Features**:
  - [ ] Integration with other quantization techniques
  - [ ] Support for more task types beyond text generation
  - [ ] Model debugging and introspection tools
  - [ ] Integrate TRLX from Carper

... And so much more coming up!

## 💌 Feedback & Contributions 💌

We're excited about the journey ahead and would love to have you with us! For feedback, suggestions, or contributions, feel free to open an issue or a pull request. Let's shape the future of fine-tuning together! 🌱

## 📜 License 📜

MIT


# Share the Love! 💙

Spread the message of the Finetuning-Suite, this is an foundational tool to help everyone quantize and finetune state of the art models.

Sharing the project helps us reach more people who could benefit from it, and it motivates us to continue developing and improving the suite.

Click the buttons below to share Finetuning-Suite on your favorite social media platforms:

- [Share on Twitter](https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FFinetuning-Suite&text=Check%20out%20Finetuning-Suite!%20A%20great%20resource%20for%20machine%20learning%20finetuning.%20%23AI%20%23MachineLearning%20%23GitHub)

- [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FFinetuning-Suite)

- [Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FFinetuning-Suite&title=Finetuning-Suite&summary=Check%20out%20this%20fantastic%20resource%20for%20machine%20learning%20finetuning!)

- [Share on Reddit](https://reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FFinetuning-Suite&title=Check%20out%20Finetuning-Suite!)

Also, we'd love to see how you're using Finetuning-Suite! Share your projects and experiences with us by tagging us on Twitter [@finetuning-suite](https://twitter.com/kyegomezb).

Lastly, don't forget to ⭐️ the repository if you find it useful. Your support means a lot to us! Thank you! 💙





### Documentation:

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