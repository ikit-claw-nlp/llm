from tokenizers import SentencePieceBPETokenizer
import glob

if __name__ == "__main__":
    corpus_files = glob.glob("./data/text/*.txt")
    tokenizer = SentencePieceBPETokenizer()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOA]"]
    tokenizer.train(
        files = corpus_files,
        vocab_size = 5000,
        min_frequency = 5,
        special_tokens = special_tokens,
        show_progress = True
    )
    # Remember create the dir tiny_llm_tokenizer in advance!
    tokenizer.save("tiny_llm_tokenizer/tokenizer.json")