from pypinyin import lazy_pinyin, Style
import jieba
import os
from importlib.resources import files
import re
import numpy as np
from typing import Optional, Union
def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list

def get_tokenizer(dataset_name, tokenizer: str = "custom"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks based on estimated character count,
    respecting sentence boundaries where possible.
    Max_chars is an estimate, actual byte length might vary.
    """
    chunks = []
    current_chunk = ""
    # More robust sentence splitting for English and Chinese
    sentences = re.split(r'(?<=[.?!;；。？！])\s*', text)
    # Filter out empty strings that can result from splitting
    sentences = [s for s in sentences if s]

    if not sentences:
        if text: # Handle case where text has no sentence-ending punctuation
            sentences = [text]
        else:
            return [] # No text, no chunks

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Estimate length (simple char count, pinyin will expand this later)
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " " # Add space between sentences
        else:
            # If adding the sentence exceeds max_chars
            if current_chunk: # Add the previous chunk if it exists
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " " # Start new chunk with current sentence
            else: # Sentence itself is longer than max_chars
                # Simple split for very long sentences (could be improved)
                parts = [sentence[i:i+max_chars] for i in range(0, len(sentence), max_chars)]
                chunks.extend(p.strip() + (" " if i < len(parts)-1 else "") for i, p in enumerate(parts))
                current_chunk = "" # Reset current chunk

    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())

    # Filter out any potential empty chunks again
    chunks = [c for c in chunks if c]
    return chunks

def list_str_to_idx(
    text: list[list[str]], # Expects list of lists of chars/pinyin
    vocab_char_map: dict[str, int],
    max_length: int,
    padding_value=0, # Use 0 for padding index (which maps to space or unknown)
):
    outs = []
    #unk_idx = vocab_char_map.get('<unk>', vocab_char_map.get(' ', 0)) # Use space if <unk> not present

    for t in text:
        # Map characters/pinyin, using unk_idx for unknown ones
        list_idx_tensors = [vocab_char_map.get(c, 0) for c in t]
        text_ids = np.asarray(list_idx_tensors, dtype=np.int32)

        # Add 1 to all indices (making padding 1, original indices shifted)
        text_ids = text_ids + 1 # Let's reconsider this, maybe padding with 0 is better if space is 0

        # Pad sequence
        pad_len = max_length - text_ids.shape[-1]
        if pad_len < 0:
            print(f"Warning: Truncating text sequence from {text_ids.shape[-1]} to {max_length}")
            text_ids = text_ids[:max_length]
            pad_len = 0

        # Pad with the designated padding_value (e.g., 0)
        text_ids = np.pad(text_ids, ((0, pad_len)), constant_values=padding_value)
        outs.append(text_ids)

    if not outs:
      return np.array([], dtype=np.int32).reshape(0, max_length)

    stacked_text_ids = np.stack(outs)
    return stacked_text_ids