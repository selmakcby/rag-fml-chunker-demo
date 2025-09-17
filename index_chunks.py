#!/usr/bin/env python3
"""
Build a lightweight vector index over chunked FML docs for Ollama-based RAG.

- Scans <chunks_root>/{project,floor,design,room,item}/*.json
- Text = summary (if present) else doc, plus breadcrumbs reconstructed via parent_id links
- Embeddings via Ollama: POST http://localhost:11434/api/embeddings
  model configurable with OLLAMA_EMBED_MODEL (default: "nomic-embed-text")
- Outputs:
  - <index_dir>/vectors.npy       (float32 [N, D])
  - <index_dir>/meta.jsonl        (one json per line: {"id","type","path","title","text"})
  - <index_dir>/config.json       (for retrieval config / sanity)

Usage:
  python index_chunks.py --chunks /path/to/fml_chunks --out /path/to/index
"""

import os, json, glob, argparse, time
from typing import Dict, Any, List, Tuple
import numpy as np
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def write_json(p: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def gather_files(root: str) -> List[str]:
    out = []
    for typ in ("project","floor","design","room","item"):
        out.extend(glob.glob(os.path.join(root, typ, "*.json")))
    return sorted(out)

def build_maps(root: str) -> Tuple[Dict[str, Dict[str,Any]], Dict[str, Dict[str,Any]]]:
    """Return: (by_id, by_path) so we can follow parent_id for breadcrumbs."""
    by_id, by_path = {}, {}
    for p in gather_files(root):
        d = read_json(p)
        by_path[p] = d
        if "id" in d:
            by_id[d["id"]] = d
    return by_id, by_path

def title_for(doc: Dict[str,Any]) -> str:
    t = doc.get("type","")
    attrs = doc.get("attrs",{}) or {}
    name = attrs.get("name") or attrs.get("label") or attrs.get("role") or ""
    if name:
        return f"{t}: {name}"
    return t

def breadcrumb(doc: Dict[str,Any], by_id: Dict[str,Dict[str,Any]]) -> str:
    chain = []
    cur = doc
    seen = set()
    # climb to project
    for _ in range(6):
        if not cur or not isinstance(cur, dict): break
        typ = cur.get("type","")
        attrs = cur.get("attrs",{}) or {}
        nm = attrs.get("name") or attrs.get("label") or attrs.get("role") or ""
        if nm: chain.append(f"{typ}:{nm}")
        else:  chain.append(typ or "node")
        pid = cur.get("parent_id")
        if not pid or pid in seen: break
        seen.add(pid)
        cur = by_id.get(pid)
    return " > ".join(reversed(chain))

def text_for(doc: Dict[str,Any], trail: str) -> str:
    base = doc.get("summary") or doc.get("doc") or ""
    attrs = doc.get("attrs",{}) or {}
    # add a few useful attrs for retrieval
    extras = []
    for key in ("role","name","label"):
        if attrs.get(key):
            extras.append(f"{key}: {attrs[key]}")
    # final payload
    payload = "\n".join(
        [f"[breadcrumb] {trail}", base] + ([f"[attrs] " + " | ".join(extras)] if extras else [])
    )
    return payload.strip()

def embed_texts(texts: List[str]) -> np.ndarray:
    """Call Ollama embeddings API. Returns float32 array [N, D]."""
    vecs = []
    s = requests.Session()
    for i, t in enumerate(texts):
        # Ollama expects: {"model": "...", "prompt": "text"}
        r = s.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": t})
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"no embedding returned for item {i}")
        vecs.append(np.array(emb, dtype=np.float32))
        if (i+1) % 100 == 0:
            print(f"  embedded {i+1}/{len(texts)}")
    return np.vstack(vecs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="root folder produced by rag_chunker (with project/floor/design/room/item)")
    ap.add_argument("--out", required=True, help="output index dir")
    ap.add_argument("--max_chars", type=int, default=1200, help="truncate long texts to this length before embedding")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    by_id, _ = build_maps(args.chunks)

    files = gather_files(args.chunks)
    if not files:
        raise SystemExit(f"No chunk files found under {args.chunks}")

    meta_lines, texts = [], []
    for p in files:
        d = read_json(p)
        br = breadcrumb(d, by_id)
        t = text_for(d, br)
        if args.max_chars and len(t) > args.max_chars:
            t = t[:args.max_chars]
        meta = {
            "id": d.get("id",""),
            "type": d.get("type",""),
            "path": os.path.relpath(p, args.chunks),
            "title": title_for(d),
            "breadcrumb": br,
        }
        meta_lines.append(meta)
        texts.append(t)

    print(f"Embedding {len(texts)} chunks with '{EMBED_MODEL}' via {OLLAMA_URL} …")
    vecs = embed_texts(texts)  # [N, D]
    np.save(os.path.join(args.out, "vectors.npy"), vecs)

    with open(os.path.join(args.out, "meta.jsonl"), "w") as f:
        for m in meta_lines:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    write_json(os.path.join(args.out, "config.json"), {
        "chunks_root": os.path.abspath(args.chunks),
        "created_at": int(time.time()),
        "embed_model": EMBED_MODEL,
        "ollama_url": OLLAMA_URL,
        "count": len(meta_lines),
        "dims": int(vecs.shape[1]),
    })

    print(f"done. wrote {len(meta_lines)} vectors to {args.out}")

if __name__ == "__main__":
    main()
