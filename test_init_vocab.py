#!/usr/bin/env python3
"""
Test script for init_vocab with smp_like algorithm.
"""

import logging
from py_unigram.initial_vocab import _compute_substring_frequencies_spm_like
from py_unigram.utils import create_logger

def main():
    # Hard-coded list of strings
    texts = [
        "Grammy Awards ceremony",
        "Grammy winning song",
        "Grand"
    ]
    pretokens = {text: 1 for text in texts}
    logger = create_logger("test_init_vocab", verbose=True)
    # Call the underlying substring frequency function
    try:
        substring_frequencies = _compute_substring_frequencies_spm_like(
            logger=logger,
            pretokens=pretokens,
            max_token_length=100,
            include_single_chars=False,
            verbose=True,
        )
        
        print(f"\n=== Raw Substring Frequencies (sorted by string value) ===")
        print(f"Found {len(substring_frequencies)} unique substrings:")
        
        # Sort substrings by their string value and print the results
        sorted_substrings = sorted(substring_frequencies.items(), key=lambda x: x[0])
        for i, (substring, frequency) in enumerate(sorted_substrings):
            print(f"  {i+1:3d}. '{substring}' (freq: {frequency})")
            
        # Print some statistics
        print(f"\nStatistics:")
        print(f"  - Total unique substrings: {len(substring_frequencies)}")
        print(f"  - Single character substrings: {sum(1 for s in substring_frequencies if len(s) == 1)}")
        print(f"  - Multi character substrings: {sum(1 for s in substring_frequencies if len(s) > 1)}")
        print(f"  - Average substring length: {sum(len(s) for s in substring_frequencies) / len(substring_frequencies):.2f}")
        print(f"  - Longest substring: '{max(substring_frequencies.keys(), key=len)}' ({max(len(s) for s in substring_frequencies)} chars)")
        print(f"  - Total frequency count: {sum(substring_frequencies.values())}")
        
    except Exception as e:
        print(f"Error calling _compute_substring_frequencies_spm_like: {e}")
        raise

if __name__ == "__main__":
    main()