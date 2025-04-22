from tokenizers import SentencePieceBPETokenizer
import glob
"""
Reference:
- https://zenn.dev/syoyo/articles/8647ae42a3be63
- https://github.com/huggingface/tokenizers
- The [source code](https://github.com/huggingface/tokenizers/blob/f2ec3b239b0a7a9866b01ec5cbd4d44243a40a16/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py#L78) of SentencePieceTokenizer
- https://discuss.huggingface.co/t/training-sentencepiece-from-scratch/3477
"""
if __name__ == "__main__":
    corpus_files = glob.glob("./data/text/*.txt")
    tokenizer = SentencePieceBPETokenizer()
    # <s> has ID 0, <pad> has id 1, etc.
    # Note that, <unk> must be contained in the special token list. Otherwise, the 
    # tokenizer will raise an exception (Unknown token <unk> is not found in the vocabulary) later.
    # The unknown token must be <unk>. I think it's a bug in the tokenizer library.
    # cf. https://huggingface.co/docs/tokenizers/v0.20.3/en/quicktour?code=python#training-the-tokenizer, cited as follows.
    # The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth.
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tokenizer.train(
        files = corpus_files,
        vocab_size = 10000,
        min_frequency = 5,
        special_tokens = special_tokens,
        show_progress = True
    )
    # Remember create the dir tiny_llm_tokenizer in advance!
    # the "save_model" doesn't work here.
    tokenizer.save("tiny_llm_tokenizer/tokenizer.json")