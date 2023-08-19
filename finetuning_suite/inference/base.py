from abc import ABC, abstractmethod

class inference(ABC):
    def __init__(self, model, tokenizer, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def inference(self):
        pass 
