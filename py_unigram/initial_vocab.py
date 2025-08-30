import heapq
import logging
import math
from collections import Counter
from py_unigram.model import Token

"""
Initial vocabulary generation utilities.

This module provides two ways to compute substring frequencies used to seed a
Unigram tokenizer's initial vocabulary:

- simple: counts substrings within each pretoken independently (fast, local)
- spm_like: uses a global suffix-array + LCP approach (SentencePiece-like)

It also includes helpers to score/select an initial set of tokens and to
convert them into the `Token` objects used by the model.

Notes on the SPM-like algorithm:
- It concatenates all pretokens into a single UTF-8 byte corpus with a NULL
  delimiter so repeats can be discovered across boundaries while preventing
  substrings from crossing sentence boundaries.
- Repeated substrings are discovered from the LCP array by tracking intervals
  on a stack; when an interval closes, we know its frequency and length.
- Only substrings with frequency > 1 and length > 1 are emitted.
- Single characters are added afterwards to ensure all chars are present.
"""


def _kasai_lcp(corpus: bytes | str, suffix_array: list[int]) -> list[int]:
    """Compute the LCP (Longest Common Prefix) array from a suffix array.

    Kasai's algorithm runs in linear time. Given a corpus and its suffix array,
    returns an array `lcp_array` such that:
    lcp_array[i] = LCP(suffix at suffix_array[i], suffix at suffix_array[i-1])
    with lcp_array[0] = 0.
    """
    corpus_length = len(corpus)
    rank_by_position = [0] * corpus_length
    for position_index, suffix_position in enumerate(suffix_array):
        rank_by_position[suffix_position] = position_index
    lcp_array = [0] * corpus_length
    common_prefix_length = 0
    for position_index in range(corpus_length):
        suffix_rank = rank_by_position[position_index]
        if suffix_rank == 0:
            common_prefix_length = 0
            continue
        previous_suffix_pos = suffix_array[suffix_rank - 1]
        while (
            position_index + common_prefix_length < corpus_length
            and previous_suffix_pos + common_prefix_length < corpus_length
            and corpus[position_index + common_prefix_length]
            == corpus[previous_suffix_pos + common_prefix_length]
        ):
            common_prefix_length += 1
        lcp_array[suffix_rank] = common_prefix_length
        if common_prefix_length:
            common_prefix_length -= 1
    return lcp_array

def _compute_substring_frequencies_simple(
    pretokens: dict[str, int], max_token_length: int
) -> Counter:
    """
    Count substrings within each pretoken independently (no cross-token repeats).

    This is a straightforward O(N * L^2) method over each pretoken's length L.
    It is fast and simple, but cannot discover repeated substrings that appear
    across pretoken boundaries.
    """
    substring_freq = Counter()
    for pretoken, freq in pretokens.items():
        token_len = len(pretoken)
        # Enumerate all substrings up to max_token_length
        for i in range(token_len):
            # j is exclusive; clamp to max_token_length
            for j in range(i + 1, min(token_len + 1, i + max_token_length + 1)):
                substring = pretoken[i:j]
                substring_freq[substring] += freq
    return substring_freq


def _compute_substring_frequencies_spm_like(
    logger: logging.Logger,
    pretokens: dict[str, int],
    max_token_length: int,
    *,
    include_single_chars: bool = True,
    verbose: bool = False,
    log_limit: int = 50,
) -> Counter:
    """
    Compute substring frequencies using a global suffix array + LCP analysis.

    This mirrors SentencePiece's seed generation:
    1) Build a single UTF-8 byte corpus by joining all pretokens with a NULL
       delimiter (so repeats are global but cannot cross sentence boundaries).
    2) Build the suffix array (SA) and compute the LCP array via Kasai.
    3) Scan the LCP with a stack to identify repeated substrings (length>1,
       frequency>1). Each time an interval "closes" we know its frequency.

    Important detail: This procedure emits repeated substrings when their LCP
    interval closes. It captures maximal repeats; shorter prefixes that are
    always part of a longer repeated substring may not be emitted unless their
    own LCP interval also closes with freq>1. If you need all repeated prefixes
    as well, you can augment this to also emit prefix substrings for each
    closed interval.
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

    if verbose:
        # Show SA entries and their corresponding suffixes (truncated)
        show = min(n, max(0, log_limit))
        logger.info(f"SA preview (first {show} of {n}): index -> pos | suffix")
        for idx in range(show):
            pos = sa[idx]
            # Show up to 40 UTF-8 chars of the suffix for readability.
            suffix_bytes = corpus[pos:pos + 160]  # bytes slice, may cut mid-char
            try:
                suffix_text = suffix_bytes.decode("utf-8", errors="replace")
            except Exception:
                suffix_text = str(suffix_bytes)
            # Trim for display
            if len(suffix_text) > 40:
                suffix_text = suffix_text[:37] + "..."
            # Make NUL visible
            suffix_text = suffix_text.replace("\u0000", "\\0")
            logger.info(f"  SA[{idx:>4}] -> {pos:>6} | '{suffix_text}'")

        # Show LCP preview
        lcp_preview = ", ".join(str(x) for x in lcp[:show])
        logger.info(f"LCP preview (first {show}): [{lcp_preview}] ...")

    logger.info("Extracting frequent substrings from LCP array...")
    stack = []
    # Use a stack-based algorithm to find all repeated substrings from the LCP array.
    # Each stack item represents an LCP "interval" with a uniform minimum height.
    # When the current LCP height drops below the top's height, the interval
    # closes and we can compute its frequency and emit the corresponding substring.
    for i in range(1, n):
        height = lcp[i]
        start_pos = i - 1  # left boundary in SA-index space for current interval
        while stack and stack[-1]['height'] > height:
            top = stack.pop()
            # Number of suffixes covered by the interval is its frequency
            freq = i - top['start_pos']
            # The common prefix length for that interval determines substring length
            length = top['height']
            if freq > 1 and length > 1:
                # Representative occurrence starts at SA[start]; any is fine
                substring_bytes = corpus[
                    sa[top['start_pos']] : sa[top['start_pos']] + length
                ]
                if DELIMITER in substring_bytes:
                    if verbose:
                        logger.info(
                            f"skip (has NUL): i={i} start={top['start_pos']} len={length} freq={freq}"
                        )
                    start_pos = top['start_pos']
                    continue
                try:
                    substring_text = substring_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    if verbose:
                        logger.info(
                            f"skip (decode): i={i} start={top['start_pos']} len={length} freq={freq}"
                        )
                    start_pos = top['start_pos']
                    continue
                if 1 < len(substring_text) <= max_token_length:
                    # Record frequency for this repeated substring interval
                    substring_freq[substring_text] = freq
                    if verbose:
                        logger.info(
                            f"emit: i={i} start={top['start_pos']} len={length} freq={freq} -> '{substring_text}'"
                        )
            start_pos = top['start_pos']

        if not stack or stack[-1]['height'] < height:
            # Start a new interval at this height
            stack.append({'height': height, 'start_pos': start_pos})
            if verbose and height > 0:
                logger.info(
                    f"push: i={i} height={height} start={start_pos} (open interval)"
                )

    # Flush any remaining intervals on the stack.
    # This emits substrings for intervals that extend to the end of the SA scan,
    # matching implementations that enumerate all internal nodes (e.g., our C++ test).
    i = n
    height = 0
    start_pos = n - 1
    while stack and stack[-1]['height'] > height:
        top = stack.pop()
        freq = i - top['start_pos']
        length = top['height']
        if freq > 1 and length > 1:
            substring_bytes = corpus[
                sa[top['start_pos']] : sa[top['start_pos']] + length
            ]
            if DELIMITER in substring_bytes:
                start_pos = top['start_pos']
            else:
                try:
                    substring_text = substring_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    start_pos = top['start_pos']
                else:
                    if 1 < len(substring_text) <= max_token_length:
                        substring_freq[substring_text] = freq
                        if verbose:
                            logger.info(
                                f"emit(flush): i=end start={top['start_pos']} len={length} freq={freq} -> '{substring_text}'"
                            )
        start_pos = top['start_pos']

    # The suffix array method above only finds repeated substrings (freq>1).
    # Optionally add single characters so they are also present.
    if include_single_chars:
        for text, freq in pretokens.items():
            for char in text:
                # Don't add the delimiter if it accidentally gets in.
                if char != "\u0000":
                    substring_freq[char] += freq
                
    return substring_freq


def _finalize_tokens(
    selected_tokens: list[tuple[float, str]], required_tokens: set[str]
) -> list[Token]:
    """Convert scored token strings into `Token` objects with log-probabilities.

    The score for each token is typically frequency * length. We convert those
    positive scores into a normalized log-probability by dividing by the sum of
    scores and taking the log, i.e. log(score) - log(sum_scores).
    """
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
    Build an initial vocabulary for Unigram training by scoring substrings.

    This dispatches to one of two substring counters and then applies a common
    "frequency × length" scoring heuristic with priority given to
    `required_tokens` when selecting the top `num_tokens` candidates.

    Algorithms:
    - "spm_like" (default): Global suffix array analysis (SentencePiece-like).
      Requires `pydivsufsort`. Recommended for best quality.
    - "simple": Local brute-force over each pretoken.

    Args:
        logger: Where progress/debug messages are written.
        pretokens: Map of pretoken string -> frequency in corpus.
        required_tokens: Tokens that must be present in the seed vocabulary.
        num_tokens: Target size for the initial vocabulary.
        max_token_length: Ignore substrings longer than this.
        algo: One of {"spm_like", "simple"}.

    Returns:
        List of `Token` objects with text, id, log_prob, and required flag.
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