from abc import ABC, abstractmethod

class InferenceHandler(ABC):
    @abstractmethod
    def run(
        self, 
        prompt_text=None, 
        model=None, 
        tokenizer=None, 
        device=None, 
        max_length = None
    ):
        pass


class DefaultInferenceHandler(InferenceHandler):
    def run(
            self, 
            prompt_text, 
            model, 
            tokenizer, 
            device, 
            max_length
        ):
        inputs = tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        outputs = model.run(inputs, max_length=max_length, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    


