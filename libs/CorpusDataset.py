from transformers import AutoTokenizer
import glob
from torch.utils.data import DataLoader, IterableDataset
import torch
class CorpusDataset(IterableDataset):
    def __init__(self, file_list, window_size, step_length, tokenizer):
        super(CorpusDataset).__init__()
        self.file_list = file_list
        self.window_size = window_size
        self.step_length = step_length
        self.tokenizer = tokenizer
    def __iter__(self):
        # For those who want to handle the boundary case:
        # [a, b, c, d, e, f] => text. Step_size =s 2, window_size = 4
        # First window: (input) [a, b, c, d] (output) [b, c, d, e] => idx = 0
        # second window: (input) [c, d, e, f] (output) [d, e, f, pad] => idx = 2
        # third window: (input) [e, f, pad, pad] (output) [f, pad, pad] => idx = 4
        # the number of pad = idx + window_size - len(text) for input.
        #                   = idx + 1 + window_size - len(text) for output.
        # I choose to ignore using <pad> as input in training.
        for corpus_f in self.file_list:
            with open(corpus_f, 'r') as f_handle:
                print(f"Loading the dataset {corpus_f} into memory...")
                current_corpus = f_handle.read()
                print("Converting the dataset to token ids...")
                tokenized_current_corpus_input_ids = self.tokenizer.encode(current_corpus,
                                                                      return_tensors="pt")
                tokenized_current_corpus_input_ids = torch.squeeze(tokenized_current_corpus_input_ids)
                print("Conversion Complete.", tokenized_current_corpus_input_ids.shape,
                      "Tokens in the corpus.")
            for idx in range(0, len(tokenized_current_corpus_input_ids) - self.window_size, self.step_length):
                # Note that, in there we drop the last part of the corpus if it cannot form a full-size window.
                # we do not use <pad> to pad the last part of the corpus.
                input_ids = tokenized_current_corpus_input_ids[idx : idx + self.window_size]
                output_ids = tokenized_current_corpus_input_ids[idx + 1 : idx + 1 + self.window_size]
                yield input_ids, output_ids
