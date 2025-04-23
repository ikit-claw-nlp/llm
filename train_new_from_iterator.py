import os
import glob
from transformers import AutoTokenizer

def corpus_iterator_func():
    corpus_files = glob.glob("./data/text/*.txt")
    for file in corpus_files:
        with open(file, 'r', encoding='utf8') as f_handle:
            corpus_text = f_handle.read()
        for idx in range(0, len(corpus_text), 1000):
            yield corpus_text[idx: idx + 1000]

if __name__ == "__main__":
    rakuten_tokenizer = AutoTokenizer.from_pretrained("Rakuten/RakutenAI-7B")
    corpus_iter = corpus_iterator_func()
    tokenizer = rakuten_tokenizer.train_new_from_iterator(
        corpus_iter,
        vocab_size = 10000, # This is not working because it's too small.
        show_progress = True
    )
    tokenizer.save_pretrained("rakuten-based-ikit-tokenizer")