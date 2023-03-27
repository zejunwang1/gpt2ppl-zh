# coding=utf-8

import numpy as np
from gpt2ppl import GPT2PPL

def main():
    sent1 = "机器学习与人工智能"
    sent2 = "机器学习与人工只能"

    model = GPT2PPL()
    """
    model = GPT2PPL(
        model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
        tokenizer_mode="bert",
        device="cuda",
        stride=512
    )
    """
    
    ppl1 = model.get_ppl(sent1)
    ppl2 = model.get_ppl(sent2)

    print("Sentence: {}, Perplexity Score: {}".format(sent1, ppl1))
    print("Sentence: {}, Perplexity Score: {}".format(sent2, ppl2))

if __name__ == "__main__":
    main()
