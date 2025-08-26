#%%

from py_unigram.pretokenize import pretokenize_corpus, SPACES_PRE_TOKENIZER_REGEX
from py_unigram.train import train_unigram
import math

def swift_data():
    with open("tests/data/swift_clean.txt") as f:
        texts = f.read().splitlines()
        pretokens = pretokenize_corpus(texts, SPACES_PRE_TOKENIZER_REGEX)
    return pretokens, texts

n = 1024
pretokens, texts = swift_data()
model1, stats1 = train_unigram(pretokens=pretokens, vocab_size=n, max_token_len=16, initial_vocab_factor=10, verbose=True)
model2, stats2 = train_unigram(pretokens=pretokens, vocab_size=n, max_token_len=16, initial_vocab_factor=11, verbose=True)

print(stats1)
print(stats2)

#%% num shared tokens
tokens1 = {t.text:t for t in model1.tokens_by_id.values()}
tokens2 = {t.text:t for t in model2.tokens_by_id.values()}

num_shared = len(tokens1.keys() & tokens2.keys())
print(f"Shared tokens: {num_shared}")
for t in tokens1.keys() & tokens2.keys():
    t1 = tokens1[t]
    t2 = tokens2[t]
    print(f"{t1.id:5} {repr(t):25}  {math.exp(t1.log_prob):<8.5%} {math.exp(t2.log_prob):<8.5%}")

m1only = tokens1.keys() - tokens2.keys()
m1t = sorted([tokens1[t] for t in m1only], key=lambda t: t.log_prob)
print(f"Only in m1: {len(m1only)}:")
for t in m1t:
    print(f"{t.id:5} {repr(t.text):25}  {math.exp(t.log_prob):<8.5%}")

# %%
