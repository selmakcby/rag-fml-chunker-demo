#!/usr/bin/env python3
"""
Tiny search util over your index (no LLM), handy to sanity check retrieval.
Returns top-k chunk titles/paths by cosine similarity.
"""

import os, json, argparse, numpy as np, requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def load_index(index_dir: str):
    vecs = np.load(os.path.join(index_dir, "vectors.npy"))
    meta = []
    with open(os.path.join(index_dir, "meta.jsonl"), "r") as f:
        for line in f:
            meta.append(json.loads(line))
    return vecs.astype(np.float32), meta

def embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    emb = r.json().get("embedding")
    return np.array(emb, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (bn @ an).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    vecs, meta = load_index(args.index)
    qv = embed(args.q)
    sims = cosine(qv, vecs)
    idx = np.argpartition(-sims, min(args.k, len(sims)-1))[:args.k]
    idx = idx[np.argsort(-sims[idx])]

    for rank, i in enumerate(idx, 1):
        m = meta[i]
        print(f"[{rank:02}] {sims[i]:.4f}  {m['type']:6}  {m['title']}  →  {m['path']}")

if __name__ == "__main__":
    main()
