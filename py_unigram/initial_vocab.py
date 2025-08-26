import heapq
import logging
import math
from collections import Counter
from typing import List, Set, Tuple
from py_unigram.model import Token


def _kasai_lcp(s: str, sa: List[int]) -> List[int]:
    """Compute LCP array from suffix array using Kasai's algorithm.
    Returns an array lcp where lcp[i] = LCP(sa[i], sa[i-1]) with lcp[0] = 0.
    """
    n = len(s)
    rank = [0] * n
    for i, si in enumerate(sa):
        rank[si] = i
    lcp = [0] * n
    k = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            k = 0
            continue
        j = sa[r - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[r] = k
        if k:
            k -= 1
    return lcp

def _compute_substring_frequencies_simple(
    pretokens: dict[str, int], max_token_length: int
) -> Counter:
    """
    Computes substring frequencies by analyzing each pre-token in isolation (brute-force).
    This method is fast but cannot discover frequent substrings that span across
    the original pre-token boundaries.
    """
    substring_freq = Counter()
    for pretoken, freq in pretokens.items():
        L = len(pretoken)
        for i in range(L):
            for j in range(i + 1, min(L + 1, i + max_token_length + 1)):
                substring_freq[pretoken[i:j]] += freq
    return substring_freq


def _compute_substring_frequencies_spm_like(
    logger: logging.Logger, pretokens: dict[str, int], max_token_length: int
) -> Counter:
    """
    Computes substring frequencies using a suffix array over the entire corpus.
    This is a faithful recreation of SentencePiece's global analysis, allowing it
    to discover any frequent substring.
    """
    try:
        from pydivsufsort import divsufsort
    except ImportError:
        logger.error("The 'spm_like' algorithm requires the 'pydivsufsort' package.")
        logger.error("Please install it via: pip install pydivsufsort")
        raise

    substring_freq = Counter()
    # Use the NULL byte as a delimiter to prevent tokens from spanning sentences.
    DELIMITER = b"\x00"

    logger.info("Concatenating corpus for suffix array construction...")
    # Build as UTF-8 bytes so pydivsufsort can handle non-ascii input
    corpus = DELIMITER.join([text.encode("utf-8") for text, freq in pretokens.items() for _ in range(freq)])
    n = len(corpus)
    logger.info(f"Corpus size: {n:,} bytes. Building suffix and LCP arrays...")

    sa = divsufsort(corpus)
    lcp = _kasai_lcp(corpus, sa)

    logger.info("Extracting frequent substrings from LCP array...")
    stack = []
    # Use a stack-based algorithm to find all repeated substrings from the LCP array.
    for i in range(1, n):
        height = lcp[i]
        start_pos = i - 1
        while stack and stack[-1]['height'] > height:
            top = stack.pop()
            freq = i - top['start_pos']
            length = top['height']
            if freq > 1 and length > 1:
                substring_bytes = corpus[sa[top['start_pos']] : sa[top['start_pos']] + length]
                if DELIMITER in substring_bytes:
                    start_pos = top['start_pos']
                    continue
                try:
                    substring_text = substring_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    start_pos = top['start_pos']
                    continue
                if 1 < len(substring_text) <= max_token_length:
                    substring_freq[substring_text] = freq
            start_pos = top['start_pos']

        if not stack or stack[-1]['height'] < height:
            stack.append({'height': height, 'start_pos': start_pos})

    # The suffix array method only finds repeated substrings.
    # We must manually add single characters to the frequency count.
    for text, freq in pretokens.items():
        for char in text:
            # Don't add the delimiter if it accidentally gets in.
            if char != "\u0000":
                substring_freq[char] += freq
                
    return substring_freq


def _finalize_tokens(
    selected_tokens: List[Tuple[float, str]], required_tokens: Set[str]
) -> List[Token]:
    """Converts a list of selected token strings into final Token objects."""
    valid_tokens = [(score, text) for score, text in selected_tokens if score > 0]
    if not valid_tokens:
        return []

    log_sum_scores = math.log(sum(score for score, _ in valid_tokens))
    return [
        Token(
            text=text,
            id=i,
            log_prob=math.log(score) - log_sum_scores,
            required=(text in required_tokens),
        )
        for i, (score, text) in enumerate(valid_tokens)
    ]


def build_initial_vocab(
    logger: logging.Logger,
    pretokens: dict[str, int],
    required_tokens: set[str],
    num_tokens: int,
    max_token_length: int = 16,
    *,
    algo: str = "spm_like",
) -> list[Token]:
    """
    Builds an initial vocabulary for Unigram model training.

    This function dispatches to one of two algorithms:
    - "spm_like" (default): A faithful recreation of SentencePiece's seed
      generation using a suffix array for global substring analysis. Requires
      the 'pydivsufsort' package. This is the recommended algorithm.
    - "simple": A faster, brute-force method that analyzes pre-tokens in
      isolation. Kept for backward compatibility.

    Args:
        logger: A logging.Logger object.
        pretokens: A dictionary mapping pre-tokenized strings to their frequencies.
        required_tokens: A set of token strings that must be included.
        num_tokens: The target number of tokens for the initial vocabulary.
        max_token_length: The maximum length of substrings to consider.
        algo: The algorithm to use, either "spm_like" or "simple".

    Returns:
        A list of Token objects representing the initial vocabulary.
    """
    algo_norm = (algo or "").lower().strip()
    logger.info(f"Building initial vocabulary using '{algo_norm}' algorithm.")

    if algo_norm == "spm_like":
        substring_freq = _compute_substring_frequencies_spm_like(
            logger, pretokens, max_token_length
        )
    elif algo_norm == "simple":
        substring_freq = _compute_substring_frequencies_simple(
            pretokens, max_token_length
        )
    else:
        raise ValueError(f"Unknown initial vocab algo: {algo}. Must be 'spm_like' or 'simple'.")

    # --- Common logic for scoring and selection ---

    # Ensure all single characters are required tokens.
    required_tokens.update({t for t in substring_freq if len(t) == 1})

    # Ensure all required tokens have a non-zero frequency so they can get a score.
    for token in required_tokens:
        substring_freq[token] = max(substring_freq.get(token, 0), 1)

    # Score candidates using SentencePiece's heuristic: frequency * length.
    all_tokens = [
        (freq * len(token), token) for token, freq in substring_freq.items()
    ]

    # Select the top N tokens, giving absolute priority to required tokens.
    selected_tokens = heapq.nlargest(
        num_tokens, all_tokens, key=lambda item: (item[1] in required_tokens, item[0])
    )

    # Convert to final Token objects with log probabilities.
    tokens = _finalize_tokens(selected_tokens, required_tokens)

    logger.info(f"🌱 Selected {len(tokens):,} initial tokens from {len(all_tokens):,} candidates.")
    logger.debug(f"   ├─ Source: {len(pretokens):,} unique and {sum(pretokens.values()):,} total pretokens")
    logger.debug(f"   ├─ Max token length: {max_token_length}")
    logger.debug(f"   └─ Total required tokens: {len(required_tokens):,}")

    return tokens