# coding=utf-8
# email: wangzejunscut@126.com
# refer: https://huggingface.co/docs/transformers/perplexity

import re
import torch
from transformers import BertTokenizerFast, GPT2TokenizerFast, GPT2LMHeadModel

def is_Chinese(c):
    cp = ord(c[0])
    if cp >= 0x4E00 and cp <= 0x9FA5:
        return True
    return False

def Chinese_chars_num(text):
    count = 0
    for c in text:
        if is_Chinese(c):
            count += 1
    return count

def split_sentence(text, min_length=8, max_length=510, return_loc=False):
    pos_list = []
    sentence_list = []
    lines = text.split("\n")
    pos = 0
    for line in lines:
        if not line.rstrip():
            pos += len(line)
            pos += 1
            continue
        line = re.sub("([。！？!?…])(.)", r"\1\n\2", line)
        line = line.split("\n")
        begin = 0
        end = len(line)
        while begin < end:
            sentence = line[begin]
            if len(sentence) < min_length:
                while len(sentence) < min_length and begin < end - 1:
                    begin += 1
                    sentence += line[begin]
                pos_list.append(pos)
                sentence_list.append(sentence)
                pos += len(sentence)
            elif len(sentence) > max_length:
                while len(sentence) > max_length:
                    pos_list.append(pos)
                    sentence_list.append(sentence[:max_length])
                    sentence = sentence[max_length:]
                    pos += max_length
                if len(sentence) < min_length:
                    sentence_list[len(sentence_list) - 1] += sentence
                else:
                    pos_list.append(pos)
                    sentence_list.append(sentence)
                    pos += len(sentence)
            else:
                pos_list.append(pos)
                sentence_list.append(sentence)
                pos += len(sentence)
            begin += 1
        pos += 1

    return (pos_list, sentence_list) if return_loc else sentence_list


class GPT2PPL:
    def __init__(
        self, 
        model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
        tokenizer_mode="bert",
        device="cuda",
        stride=512
    ):
        assert device in ["cuda", "cpu"]
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        
        if tokenizer_mode == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        
        self.device = device
        self.stride = stride
        self.max_positions = self.model.config.n_positions
    
    def get_ppl(self, text):
        encodings = self.tokenizer(text.strip(), add_special_tokens=False, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_positions, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        ppl = ppl.cpu().tolist()
        return ppl
        
    def get_ppl_per_sentence(self, text, ratio=0.4, min_length=8, max_length=510, return_loc=False):
        outputs = split_sentence(text, min_length, max_length, return_loc)
        sentence_list = outputs
        if return_loc:
            pos_list, sentence_list = outputs
        
        ppl_list = []
        for i in range(len(sentence_list)):
            sentence = sentence_list[i]
            n = Chinese_chars_num(sentence)
            if len(sentence) < min_length or n / len(sentence) < ratio:
                continue
            ppl = self.get_ppl(sentence)
            if return_loc:
                ppl_list.append((pos_list[i], sentence, ppl))
            else:
                ppl_list.append((sentence, ppl))

        return ppl_list

