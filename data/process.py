import numpy as np
import torch
from torch.utils.data import TensorDataset
import copy

class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000, min_freq=2, src_max_len=16, tgt_max_len=64):
        """Construct a text index converter.
        Args:
            text_path: txt file path.
            max_vocab: maximum number of words.
        """

        with open(text_path, 'r', encoding='utf8') as f:
            text = f.read()
        text = text.replace('\n', ' ').replace(':', ' ')
        vocab = set(text)
        # If the number of words is larger than limit, clip the words with minimum frequency.
        vocab_count = {}
        for word in vocab:
            vocab_count[word] = 0
        for word in text:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab-3]
        vocab = ['<pad>','<eos>','<unk>']+[x[0] for x in vocab_count_list]
        self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        self.src_max_len=src_max_len
        self.tgt_max_len=tgt_max_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return self.word_to_int_table['<unk>']

    def int_to_word(self, index):
        if index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return arr

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


def get_dataset(text_path, arr_to_idx, src_max_len, tgt_max_len):
    with open(text_path, 'r', encoding='utf8') as f:
        text = f.read()
    src_list=[]
    tgt_list=[]
    label_list=[]
    for t in text.strip().split('\n'):
        tlist=t.split(':')
        if len(tlist)!=2:
            print(t)
            continue
        src=arr_to_idx(tlist[0])[:src_max_len]
        tgt=arr_to_idx(tlist[1])[:tgt_max_len-1]
        label=copy.copy(tgt)
        tgt=[1]+tgt
        label.append(1)
        while len(src)<src_max_len:
            src.append(0)
        while len(tgt)<tgt_max_len:
            tgt.append(0)
            label.append(0)
        src_list.append(src)
        tgt_list.append(tgt)
        label_list.append(label)
    src_tensor=torch.tensor(src_list)
    tgt_tensor=torch.tensor(tgt_list)
    label_tensor=torch.tensor(label_list)
    print(src_tensor)
    return TensorDataset(src_tensor,tgt_tensor,label_tensor)