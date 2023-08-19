[Join Agora](https://discord.gg/qUtxnK2NMf) the community of engineers changing the world with their creations.

![Finetuning suite logo](ft-logo.png)

Finetune any model with unparalled performance, speed, and reliability using Qlora, BNB, Lora, Peft in less than 30 seconds, just press GO.
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



# Documentation
Certainly! To provide a custom preprocessing function, a user would create a new class that extends the `Preprocessor` abstract class. They then need to override the `preprocess_function` method to implement their own preprocessing logic. 

Here's an example:

### 1. Extend the Preprocessor abstract class

Let's assume a user wants to apply a preprocessing function that only takes the first sentence of each dialogue as the input:

```python
class FirstSentencePreprocessor(Preprocessor):

    def preprocess_function(self, sample, padding="max_length"):
        # Just take the first sentence of each dialogue
        inputs = [item.split('.')[0] for item in sample["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        labels = self.tokenizer(text_target=sample["sumamry"], max_length=max_target_length, padding=padding, truncation=True)
        
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
```

### 2. Pass the custom preprocessor to the FineTuner

After defining the custom preprocessor, the user should create an instance of this class and pass it to the `FineTuner` class during initialization.

```python
# Create a custom preprocessor instance
custom_preprocessor = FirstSentencePreprocessor(tokenizer=AutoTokenizer.from_pretrained('your_model_id'))

# Initialize FineTuner with the custom preprocessor
finetuner = FineTuner(model_id='your_model_id', dataset_name='your_dataset', preprocessor=custom_preprocessor)

# Continue with training or other operations...
finetuner.train(output_dir='your_output_directory', num_train_epochs=5)
```

By following these steps, users can flexibly introduce their own preprocessing logic into the pipeline without altering the core classes, adhering to a modular and extensible design.