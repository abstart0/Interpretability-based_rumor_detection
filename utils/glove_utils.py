"""
Load GloVe pre-trained word vectors and build embedding matrix.
"""

import os
import zipfile

import numpy as np
import torch
import torch.nn as nn


def load_glove(path: str) -> dict:
    """
    Load GloVe vectors from a text file into a dict {word: np.array}.

    Expected format per line: word f1 f2 ... fN
    """
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors[word] = vec
            except ValueError:
                continue
    return vectors


def build_embedding_matrix(vocab: dict, glove_vectors: dict, embed_dim: int) -> torch.Tensor:
    """
    Build an embedding weight matrix that matches the project vocabulary.

    Words in vocab but not in GloVe get random initialization (uniform [-0.25, 0.25]).
    <PAD> (id=0) gets zeros.

    Args:
        vocab:         token -> id mapping
        glove_vectors: GloVe {word: np.array}
        embed_dim:     GloVe vector dimension (must match the file)

    Returns:
        torch.Tensor of shape (vocab_size, embed_dim)
    """
    vocab_size = len(vocab)
    weight = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim)).astype(np.float32)
    weight[0] = 0.0  # <PAD>

    found = 0
    for word, idx in vocab.items():
        if word in glove_vectors:
            weight[idx] = glove_vectors[word]
            found += 1

    print(f"  GloVe coverage: {found}/{vocab_size} ({100 * found / vocab_size:.1f}%)")
    return torch.from_numpy(weight)


def load_glove_and_matrix(vocab: dict, glove_path: str, embed_dim: int) -> torch.Tensor:
    """
    Convenience: load GloVe file and build embedding matrix in one call.
    """
    print(f"Loading GloVe from: {glove_path}")
    vectors = load_glove(glove_path)
    print(f"  Loaded {len(vectors):,} GloVe vectors (dim={embed_dim})")
    return build_embedding_matrix(vocab, vectors, embed_dim)


def download_and_extract_glove(url: str, dest_dir: str, target_file: str) -> str:
    """
    Download GloVe zip and extract the target file.

    Args:
        url:         download URL for the zip
        dest_dir:    directory to save files
        target_file: name of the file to extract from the zip

    Returns:
        path to the extracted GloVe text file
    """
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "glove.zip")
    txt_path = os.path.join(dest_dir, target_file)

    if os.path.exists(txt_path):
        print(f"GloVe file already exists: {txt_path}")
        return txt_path

    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    # Extract only the target file
    print(f"Extracting {target_file} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(target_file, dest_dir)

    os.remove(zip_path)
    print(f"Extracted to: {txt_path}")
    return txt_path
