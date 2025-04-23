import sentencepiece as spm

if __name__ == "__main__":
    spm.SentencePieceTrainer.train(
    # cf. https://github.com/google/sentencepiece#train-sentencepiece-model
    # it says the input must be one-sentence-per-line text.
    input = "data/full_text/sentence_tokenized_corpus.corpus", 
    model_prefix = "ikit_spm",
    vocab_size=50000,
    character_coverage=0.9995,
    num_threads=32,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
)