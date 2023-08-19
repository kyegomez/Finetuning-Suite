from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_function(self, sample, padding="max_length"):
        pass



