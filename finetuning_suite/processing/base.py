from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_function(self, sample, padding="max_length"):
        pass


# Step 2: Default Preprocessor
class DefaultPreprocessor(Preprocessor):

    def preprocess_function(self, sample, padding="max_length", max_source_length=None, max_target_length=None):
        inputs = ["summarize" + item for item in sample["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        labels = self.tokenizer(text_target=sample["sumamry"], max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs