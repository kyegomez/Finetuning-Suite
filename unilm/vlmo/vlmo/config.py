from sacred import Experiment

ex = Experiment("VLMo")


def _loss_names(d):
    ret = {
        "itm": 0, # image-text matching loss
        "itc": 0, # image-text contrastive loss
        "mlm": 0, # masked language modeling loss
        "textmlm": 0, # text-only masked language modeling
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0, # retrieval task ft
    }
    ret.update(d)
    return ret


@ex.config
def config():
    _loss_names({"itm": 1, "itc": 1, "mlm": 1})

    # Image setting

    # Text Setting

    # Transformer Setting

    # Optimizer Setting

    # Downstream Setting

    # PL Trainer Setting

    # below params varies with the environment


# ----------------------- language pretraining config -----------------------


@ex.named_config
def task_textmlm_base():
    _loss_names({"textmlm": 1})


@ex.named_config
def task_textmlm_base_plus():
    _loss_names({"textmlm": 1})


# ----------------------- vision-language pretraining config -----------------------


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm_itc_base():
    _loss_names({"itm": 1, "mlm": 1, "itc": 1})


@ex.named_config
def task_mlm_itm_itc_base_plus():
    _loss_names({"itm": 1, "mlm": 1, "itc": 1})


@ex.named_config
def task_mlm_itm_itc_large():
    _loss_names({"itm": 1, "mlm": 1, "itc": 1})


# ----------------------- NLVR2 fine-tuning configs -----------------------


@ex.named_config
def task_finetune_nlvr2_base():
    _loss_names({"nlvr2": 1})


@ex.named_config
def task_finetune_nlvr2_base_plus():
    _loss_names({"nlvr2": 1})


@ex.named_config
def task_finetune_nlvr2_base_image384():
    _loss_names({"nlvr2": 1})


@ex.named_config
def task_finetune_nlvr2_base_plus_image384():
    _loss_names({"nlvr2": 1})


@ex.named_config
def task_finetune_nlvr2_large():
    _loss_names({"nlvr2": 1})


@ex.named_config
def task_finetune_nlvr2_large_image384():
    _loss_names({"nlvr2": 1})


# ----------------------- VQAv2 Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_vqa_base_image480():
    _loss_names({"vqa": 1})


@ex.named_config
def task_finetune_vqa_base_plus_image480():
    _loss_names({"vqa": 1})


@ex.named_config
def task_finetune_vqa_large_image480():
    _loss_names({"vqa": 1})


# ----------------------- F30K IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_f30k_base():
    _loss_names({"irtr": 1.0})


@ex.named_config
def task_finetune_irtr_f30k_base_image384():
    _loss_names({"irtr": 1.0})


@ex.named_config
def task_finetune_irtr_f30k_base_plus_image384():
    _loss_names({"irtr": 1.0})


@ex.named_config
def task_finetune_irtr_f30k_large_image384():
    _loss_names({"irtr": 1.0})


# ----------------------- COCO IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_coco_base_image384():
    _loss_names({"irtr": 1.0})


@ex.named_config
def task_finetune_irtr_coco_base_plus_image384():
    _loss_names({"irtr": 1.0})


@ex.named_config
def task_finetune_irtr_coco_large_image384():
    _loss_names({"irtr": 1.0})


# ----------------------- Other configs -----------------------


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def step1_5k():
    pass


@ex.named_config
def step3k():
    pass


@ex.named_config
def step200k():
    pass


@ex.named_config
def step500k():
    pass