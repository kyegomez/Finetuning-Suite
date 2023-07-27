from __future__ import absolute_import, division, print_function

import logging


from markuplmft.models.markuplm import MarkupLMConfig, MarkupLMTokenizer, MarkupLMForQuestionAnswering


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    mp = "../../../../../results/markuplm-base"
    op = "./moli"
    config = MarkupLMConfig.from_pretrained(mp)
    logger.info("=====Config for model=====")
    logger.info(str(config))
    max_depth = config.max_depth
    tokenizer = MarkupLMTokenizer.from_pretrained(mp)
    model = MarkupLMForQuestionAnswering.from_pretrained(mp, config=config)

    tokenizer.save_pretrained(op)