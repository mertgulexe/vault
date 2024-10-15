import torch
from typing import List

### TOKENIZER & VOCABULARY CONFIGS
"""
Notes: of course not everything about a real tokenizer has not been implemented
here. here are some of the missing parts:
1) batch size is not a hyperparameter right now. it is just determined by the
length of "sentences" list.
2) edge cases like empty input list
3) more special tokens, maybe?
4) Cannot convert indices to words.
"""
SOS_TOKEN = 0  # start of sentence
EOS_TOKEN = 1  # end of sentence
PAD_TOKEN = 2  # padding
UNK_TOKEN = 3  # unknown word
index2word = {
    SOS_TOKEN: "[SOS]",
    EOS_TOKEN: "[EOS]",
    PAD_TOKEN: "[PAD]",
    UNK_TOKEN: "[UNK]"
}
vocab = "Two roads diverged in a wood , and I - I took the one less traveled by , And that has made all the difference .".lower().split()
vocab_set = set(vocab)
VOCAB_SIZE = len(vocab_set) + len(index2word)
# indexing words
for word in sorted(vocab_set):
    index2word[len(index2word)] = word
word2index = {v:k for k, v in index2word.items()}

# creating our custom tokenizer
def custom_tokenizer(sentences: List[str], max_len: int) -> torch.Tensor:
    sentence_list = []
    for sentence in sentences:
        indices = [word2index.get(s, UNK_TOKEN) for s in sentence.lower().split()]
        indices = [SOS_TOKEN] + indices[:(max_len-2)] + [EOS_TOKEN]  # truncation. "-2" for special tokens
        indices.extend([PAD_TOKEN] * (max_len - len(indices)))  # padding
        sentence_list.append(torch.tensor(indices))
    return torch.vstack(tensors=sentence_list).to(torch.long)