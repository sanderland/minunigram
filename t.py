from datasets import load_dataset
from py_unigram.pretokenize import pretokenize_corpus, SPACES_PRE_TOKENIZER_REGEX
from py_unigram.train import train_unigram

def load_data():
    ds = load_dataset("sanderland/monolingual-tokenizer-data",
            data_files=[f"eng_latn_300mb.txt"],
            split="train",
        )
    texts = list(ds["text"])
    pretokens = pretokenize_corpus(texts, SPACES_PRE_TOKENIZER_REGEX)
    return pretokens, texts

n = 16384
pretokens, texts = load_data()
model, stats = train_unigram(pretokens=pretokens, vocab_size=n, max_token_len=16, initial_vocab_factor=10, verbose=True)
assert len(model.tokens) == n

num_toks = []
for t in texts:
    num_toks.append(len(model.encode(t)))

print("Total tokens:", sum(num_toks))
print("Average tokens per text:", sum(num_toks) / len(num_toks))
print("Min tokens per text:", min(num_toks))
print("Max tokens per text:", max(num_toks))


