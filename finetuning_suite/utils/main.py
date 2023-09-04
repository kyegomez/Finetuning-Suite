

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
    print(
        f"Trainable params: {trainable_params} || all params {all_param} || trainable: {100 * trainable_params / all_param}"
    )