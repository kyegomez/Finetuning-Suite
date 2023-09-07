from fts import Inference

model = Inference(
    model_id="georgesung/llama2_7b_chat_uncensored",
    quantized=True
)

model.run("What is your name")