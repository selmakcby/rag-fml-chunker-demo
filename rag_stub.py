#!/usr/bin/env python3
"""
RAG retriever + chat for Ollama with:
- Fallback to /api/generate if /api/chat is missing (older Ollama)
- Similarity search mode (text->chunks or chunk->chunks)
- Basic filters over chunk metadata/attrs for targeted retrieval
- Export to Markdown / JSON / TXT via --out and --fmt

Usage examples:

# Normal Q&A (RAG):
python rag_stub.py \
  --index  "/path/index" \
  --chunks "/path/fml_chunks" \
  --model  "llama3.1:8b" \
  --query  "List the items and brands in the 'Living' room on the first floor." \
  --out answer.md --fmt md

# Similarity (text):
python rag_stub.py --index /path/index --chunks /path/fml_chunks \
  --similar_text "cozy living room with brown leather sofa" --k 10 --show \
  --out similar.json --fmt json

# Similarity (chunk->chunks):
python rag_stub.py --index /path/index --chunks /path/fml_chunks \
  --similar_path "room/252876b7....json" --k 10 --show \
  --out similar_rooms.md --fmt md

# Attribute filters:
python rag_stub.py --index /path/index --chunks /path/fml_chunks \
  --query "Which projects have a brown sofa?" \
  --filter_type item --filter_kv "type_guess:sofa" --filter_kv "color:brown" --k 15 \
  --out brown_sofas.txt --fmt txt

# Compare two chunks (index NOT needed):
python rag_stub.py --chunks /path/fml_chunks \
  --compare "room/AAA.json,room/BBB.json" \
  --out compare.md --fmt md

Env:
  OLLAMA_URL (default http://localhost:11434)
  OLLAMA_EMBED_MODEL (default nomic-embed-text)
  OLLAMA_CHAT_MODEL  (default llama3.1:8b)
"""

import os, json, argparse, numpy as np, requests
from typing import List, Dict, Any
from collections import Counter
import re

# ---------------- Config ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL","llama3.1:8b")


# --- Category normalization (drop-in) ---
CATEGORY_NORMALIZER = [
    (r"\bcoffee\b|\bcenter table\b", "coffee table"),
    (r"\bside table\b|\bend table\b|\bround side\b", "side table"),
    (r"\bconsole\b", "console table"),
    (r"\barm\s*chair\b|\bhost chair\b|\baccent chair\b|\bclub chair\b", "armchair"),
    (r"\bottoman\b|\bpouf\b", "ottoman"),
    (r"\bfloor lamp\b", "floor lamp"),
    (r"\btable lamp\b|\bdesk lamp\b", "table lamp"),
    (r"\brug\b|\bcarpet\b", "rug"),
    (r"\bbookcase\b|\bshelf\b|\bshelving\b|\bunit\b", "shelving unit"),
    (r"\bmedia\b|\bconsole\b.*media", "media console"),
    (r"\bfireplace\b|\bhearth\b", "fireplace"),
    (r"\bwall unit\b", "shelving unit"),
]

def normalize_type_guess_from_text(name: str, tguess: str) -> str:
    import re
    hay = f"{(name or '').lower()} {(tguess or '').lower()}"
    for patt, norm in CATEGORY_NORMALIZER:
        if re.search(patt, hay):
            return norm
    # sensible fallback buckets
    if "lighting" in hay: return "floor lamp" if "floor" in hay else "table lamp"
    if "table" in hay:    return "side table" if "side" in hay else "coffee table"
    return (tguess or "").lower() or "decor"

def enrich_suggestion_row(r: dict, chunks_root: str) -> dict:
    """Fill missing fields and normalize category using the example item."""
    try:
        ip = os.path.join(chunks_root, r.get("example_item", ""))
        ia = read_json(ip).get("attrs", {}) if ip and os.path.isfile(ip) else {}
    except Exception:
        ia = {}
    name = ia.get("name") or r.get("example_name") or ""
    tguess = ia.get("type_guess") or ia.get("category") or r.get("type") or ""
    r["type"] = normalize_type_guess_from_text(name, tguess)
    if not r.get("brand"):
        r["brand"] = ia.get("brand") or ia.get("vendor") or ia.get("source") or "—"
    return r


def normalize_type_guess(name: str, tguess: str) -> str:
    import re
    hay = f"{name or ''} {tguess or ''}".lower()
    for patt, norm in CATEGORY_NORMALIZER:
        if re.search(patt, hay):
            return norm
    return (tguess or "").lower() or "decor"


def _coerce_item_id(spec: str) -> str | None:
    # Accept "item/AAA.json" or bare "AAA"
    s = spec.strip()
    if not s:
        return None
    if s.endswith(".json"):
        s = os.path.basename(s)  # "AAA.json"
        s = s[:-5]               # "AAA"
    if "/" in s:
        _, s = s.rsplit("/", 1)
    return s or None

def seed_doc_from_items(chunks_root: str, csv: str):
    ids = []
    for part in (csv or "").split(","):
        iid = _coerce_item_id(part)
        if not iid:
            continue
        p = os.path.join(chunks_root, "item", f"{iid}.json")
        if os.path.isfile(p):
            ids.append(iid)
    if len(ids) < 1:
        raise SystemExit("No valid items found for --complete_from_items.")
    # virtual seed room doc
    return {
        "type": "room",
        "id": "__VIRTUAL_SEED__",
        "attrs": {"name": "Virtual Seed", "role": "Living"},
        "items": ids,
    }, "room/__VIRTUAL_SEED__.json"


def all_room_docs(chunks_root: str):
    room_dir = os.path.join(chunks_root, "room")
    if not os.path.isdir(room_dir): return []
    out = []
    for fn in os.listdir(room_dir):
        if fn.endswith(".json"):
            rel = f"room/{fn}"
            try:
                out.append((rel, read_json(os.path.join(room_dir, fn))))
            except Exception:
                pass
    return out

def sig_from_items_set(items_set):
    # items_set = {(name,brand,type_guess), ...}
    # normalize to (brand,type) – this matches your relaxed-overlap idea
    return set((b, t) for (_, b, t) in items_set if (b or t))

def complete_room_from_neighbors(chunks_root: str, seed_doc: dict, neighbors: int = 12, topn: int = 6, style: str | None = None):
    # 1) seed signature
    seed_items = set(item_keys_for_room(seed_doc, chunks_root))
    seed_sig   = sig_from_items_set(seed_items)

    # 2) score all rooms by relaxed overlap size (brand×type shared)
    nbrs = []
    for (r_rel, r_doc) in all_room_docs(chunks_root):
        if r_doc is seed_doc: 
            continue
        items = set(item_keys_for_room(r_doc, chunks_root))
        sig   = sig_from_items_set(items)
        shared = len(seed_sig & sig)
        if shared > 0:
            nbrs.append((shared, r_rel, r_doc, items, sig))
    nbrs.sort(key=lambda x: x[0], reverse=True)
    nbrs = nbrs[:neighbors]

    # 3) mine missing (brand×type) pairs; keep an exemplar item path+name
    #    exemplar = first item we see that fits the pair
    counts = {}   # (brand,type) -> {"count": c, "example": (name, item_rel, room_rel)}
    seed_pairs = set((b,t) for (_,b,t) in seed_items)
    for (_, r_rel, r_doc, items, sig) in nbrs:
        for (n,b,t) in items:
            pair = (b,t)
            if pair in seed_pairs:
                continue
            # ignore ultra-generic noise
            if not (b or t): 
                continue
            slot = counts.setdefault(pair, {"count":0, "example":None})
            slot["count"] += 1
            if slot["example"] is None:
                # find the concrete item file to cite
                for iid in (r_doc.get("items") or []):
                    ip = os.path.join(chunks_root, "item", f"{iid}.json")
                    if not os.path.isfile(ip): 
                        continue
                    ia = (read_json(ip).get("attrs") or {})
                    nb = (ia.get("brand") or "").lower()
                    nt = (ia.get("type_guess") or ia.get("category") or "").lower()
                    if (nb, nt) == pair:
                        nm = ia.get("name") or ia.get("sku") or ia.get("type_guess") or ia.get("category") or "item"
                        slot["example"] = (nm, f"item/{iid}.json", r_rel)
                        break

    # 4) rank suggestions
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1]["count"], kv[0][1], kv[0][0]))[:topn]

    # 5) optional: tag style for the seed to bias wording
    style_tag = None
    if style:
        try:
            style_tag = tag_style_for_room(DEFAULT_CHAT_MODEL, seed_doc, chunks_root, preferred=style)
        except Exception:
            style_tag = None

    # 6) build a compact payload
    recs = []
    for (b,t), rec in ranked:
        ex = rec.get("example")
        recs.append({
            "brand": b or "",
            "type":  t or "",
            "seen_in_neighbors": rec["count"],
            "example_name": ex[0] if ex else "",
            "example_item": ex[1] if ex else "",
            "example_room": ex[2] if ex else "",
        })

    neighbors_summary = [
        {"room": summarise_room(chunks_root, r_rel), "path": r_rel, "shared_pairs": shared}
        for (shared, r_rel, r_doc, _, _) in nbrs
    ]

    return {
        "kind": "complete",
        "seed": summarise_room(chunks_root, seed_doc.get('path', 'seed')),
        "neighbors_used": neighbors_summary,
        "recommendations": recs,
        "style_tag": style_tag,
    }

def room_has_sofa(chunks_root: str, room_doc: dict) -> bool:
    SOFA = ("sofa", "sectional", "loveseat", "chaise")
    for iid in (room_doc.get("items") or []):
        ip = os.path.join(chunks_root, "item", f"{iid}.json")
        if not os.path.isfile(ip): continue
        a = (read_json(ip).get("attrs") or {})
        hay = " ".join(str(a.get(k,"")) for k in ("type_guess","category","name")).lower()
        if any(tok in hay for tok in SOFA):
            return True
    return False

def role_is_living(room_doc: dict) -> bool:
    role = ((room_doc.get("attrs") or {}).get("role") or "").lower()
    return "living" in role


# -----------# === ADD: product lookup + sofa evidence =====================================
def fp_products_by_ids(ids: list[str], editor_version: str | None = None) -> dict[str, dict]:
    """
    POST /products/ids -> { id -> {name, category, brand, ...} }
    This endpoint is public (used by floorplanner.com/demo).
    """
    if not ids:
        return {}
    url = "https://search.floorplanner.com/products/ids"
    body = {"ids": ids}
    if editor_version:
        url += f"?editor_version={editor_version}"
    try:
        r = requests.post(url, json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Some deployments return a list; normalize to dict
        if isinstance(data, list):
            # best effort map by known 'id' field
            return {str(d.get("id") or d.get("_id") or d.get("sku")): d for d in data if isinstance(d, dict)}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def possible_product_ids_from_item_attrs(attrs: dict) -> list[str]:
    # Best-effort: collect any plausible id keys you stored in item chunks
    keys = ["productId", "product_id", "id", "sku", "fp_id"]
    out = []
    for k in keys:
        v = attrs.get(k)
        if v is None: continue
        if isinstance(v, (int, float)): v = str(v)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out

def sofa_evidence_for_room(chunks_root: str, room_doc: dict, editor_version: str | None = None) -> dict:
    """
    Returns {ok: bool, hits: [ ... ]} where hits are resolved product records
    proving 'sofa' category (sofa/sectional/loveseat).
    Falls back to local 'type_guess' if API yields nothing.
    """
    SOFA_CATS = {"sofa", "sectional", "loveseat", "chaise sofa", "sofa bed"}
    product_ids = []
    fallback_hits = []
    for iid in (room_doc.get("items") or []):
        ip = os.path.join(chunks_root, "item", f"{iid}.json")
        if not os.path.isfile(ip): continue
        it = read_json(ip)
        attrs = (it.get("attrs") or {})
        product_ids += possible_product_ids_from_item_attrs(attrs)
        # local fallback
        cat = (attrs.get("category") or attrs.get("type_guess") or "").lower()
        name = (attrs.get("name") or "").lower()
        if any(tok in (cat + " " + name) for tok in SOFA_CATS):
            fallback_hits.append({"name": attrs.get("name"), "category": attrs.get("category") or attrs.get("type_guess"), "brand": attrs.get("brand")})

    # Resolve via public API
    resolved = fp_products_by_ids(list({pid for pid in product_ids if pid}), editor_version=editor_version)
    api_hits = []
    for pid, rec in resolved.items():
        cat = str(rec.get("category","")).lower()
        name = str(rec.get("name",""))
        if any(c in cat for c in SOFA_CATS) or "sofa" in name.lower():
            api_hits.append({"productId": pid, "name": name, "brand": rec.get("brand"), "category": rec.get("category")})

    hits = api_hits or fallback_hits
    return {"ok": len(hits) > 0, "hits": hits[:3]}  # cap to 3

# === ADD: style tagging =======================================================
def tag_style_for_room(model: str, room_doc: dict, chunks_root: str, preferred: str | None = None) -> dict:
    """
    Returns {"label": "...", "score": float, "why": "..."}.
    If 'preferred' is provided, score is for that label in [0..1].
    """
    items = []
    for iid in (room_doc.get("items") or [])[:12]:
        ip = os.path.join(chunks_root, "item", f"{iid}.json")
        if os.path.isfile(ip):
            ia = (read_json(ip).get("attrs") or {})
            items.append(f"{ia.get('name','?')} | {ia.get('brand','?')} | {ia.get('type_guess', ia.get('category',''))}")

    sys = "You are a strict interior style classifier. Be terse and numeric."
    labels = ["classic","modern","contemporary","mid-century","industrial","scandinavian","transitional","boho"]
    asked = preferred or "best_fit"
    user = f"""Given these room items, output JSON: {{"label": "<one of {labels}>", "score": 0..1, "why": "<<=110 chars>"}}.
If a preferred style is given ("{preferred or ''}"), set label to that and score how well it fits (0..1).
Items:
- """ + "\n- ".join(items)
    raw = chat_ollama(model, sys, user)
    data = _json_skim(raw)
    if not isinstance(data, dict):
        data = {}
    data.setdefault("label", preferred or "classic")
    try:
        s = float(data.get("score", 0.0))
        data["score"] = max(0.0, min(1.0, s))
    except Exception:
        data["score"] = 0.0
    data.setdefault("why", "")
    return data

def aggregate_brand_type(items_set):
    """
    Summarize items by brand, by category (type_guess), and brand×category pairs.
    items_set is a set of (name, brand, type_guess) all lowercased.
    """
    brands = Counter()
    types  = Counter()
    pairs  = Counter()
    for (n,b,t) in items_set:
        if b: brands[b] += 1
        if t: types[t]  += 1
        if b and t: pairs[(b,t)] += 1

    def top_n(counter, n=10):
        return counter.most_common(n)

    return {
        "brands_top": top_n(brands),
        "types_top":  top_n(types),
        "pairs_top":  [(f"{b}×{t}", c) for ((b,t), c) in top_n(pairs)],
    }


def relaxed_overlap(itemsA, itemsB):
    """
    Relaxed match on (brand, type_guess) — ignores exact SKU/name differences.
    Returns shared/onlyA/onlyB tuples plus counts.
    """
    def sigs(items):
        return Counter((b, t) for (_, b, t) in items if (b or t))

    A = sigs(itemsA)
    B = sigs(itemsB)
    shared_keys = set(A) & set(B)
    only_a_keys = set(A) - set(B)
    only_b_keys = set(B) - set(A)

    # one example name per signature (brand,type) for flavor
    def name_by_sig(items):
        d = {}
        for (n,b,t) in items:
            k = (b,t)
            if (b or t) and k not in d:
                d[k] = n
        return d

    nameA = name_by_sig(itemsA)
    nameB = name_by_sig(itemsB)

    return {
        "shared": [(b or "—", t or "—", min(A[(b,t)], B[(b,t)])) for (b,t) in sorted(shared_keys)],
        "only_a": [(b or "—", t or "—", A[(b,t)], nameA.get((b,t),"")) for (b,t) in sorted(only_a_keys)],
        "only_b": [(b or "—", t or "—", B[(b,t)], nameB.get((b,t),"")) for (b,t) in sorted(only_b_keys)],
        "counts": {
            "shared": sum(min(A[k], B[k]) for k in shared_keys),
            "a_total": sum(A.values()),
            "b_total": sum(B.values()),
        },
    }

# ---------- Insight generation ----------
def build_compare_insight_prompt(exp: dict, style: str | None = None) -> str:
    a_title = exp.get("A", {}).get("title", "A")
    b_title = exp.get("B", {}).get("title", "B")
    overlap = exp.get("overlap_shared", 0)
    union   = exp.get("overlap_union", 0)
    jacc    = exp.get("jaccard", 0.0)

    def fmt_tuples(tups, limit=8):
        return "\n".join(f"- {k}: {c}" for (k,c) in (tups or [])[:limit]) or "(none)"

    aggA = exp.get("aggA", {})
    aggB = exp.get("aggB", {})
    brandsA = fmt_tuples(aggA.get("brands_top"))
    brandsB = fmt_tuples(aggB.get("brands_top"))
    typesA  = fmt_tuples(aggA.get("types_top"))
    typesB  = fmt_tuples(aggB.get("types_top"))

    rel = exp.get("relaxed", {})
    rel_shared = rel.get("counts", {}).get("shared", 0)

    shared = "\n".join(f"- {s}" for s in exp.get("shared", [])[:10]) or "(none)"
    only_a = "\n".join(f"- {s}" for s in exp.get("only_a", [])[:10]) or "(none)"
    only_b = "\n".join(f"- {s}" for s in exp.get("only_b", [])[:10]) or "(none)"
    style_line = f"\n- User style preference: {style}" if style else ""

    return f"""
You are an analyst for an interior design retail team. Be concise (6–10 bullets). Do three things:
1) Differences & commonalities (mention brand/type patterns and relaxed-overlap signal)
2) Why it matters (merchandising/design implications)
3) Concrete next actions

Entities:
- A: {a_title}
- B: {b_title}
- Exact overlap: {overlap} / {union} (Jaccard={jacc:.3f})
- Relaxed overlap (brand+type): {rel_shared}

Top brands:
A:
{brandsA}

B:
{brandsB}

Top categories:
A:
{typesA}

B:
{typesB}

Shared SKUs (sample):
{shared}

Unique to A (sample):
{only_a}

Unique to B (sample):
{only_b}
""".strip()


def generate_insight(model: str, exp: dict, style: str | None = None) -> str:
    sys = "You generate crisp, actionable merchandising/design insights from structured comparisons."
    user = build_compare_insight_prompt(exp, style=style)
    try:
        return chat_ollama(model, sys, user)
    except Exception as e:
        return f"(insight generation failed: {e})"

def _sample_item_lines(items_set, maxn=10):
    rows = []
    for (n,b,t) in list(items_set)[:maxn]:
        rows.append(f"- {n or '—'} | {b or '—'} | {t or '—'}")
    more = len(items_set) - min(len(items_set), maxn)
    if more > 0:
        rows.append(f"- (+{more} more)")
    return "\n".join(rows)

def load_design(chunks_root: str, design_id_or_rel: str):
    """Load a design given a bare id or 'design/<id>.json'. Returns (doc, relpath) or (None, rel)."""
    rel = design_id_or_rel
    if not rel.endswith(".json"):
        rel = f"{design_id_or_rel}.json"
    if not rel.startswith("design/"):
        rel = f"design/{rel}"
    p = os.path.join(chunks_root, rel)
    if os.path.exists(p):
        try:
            return read_json(p), rel
        except Exception:
            pass
    return None, rel

def collect_room_ids_from_design(design_doc: dict):
    """Extract room IDs from a design doc (tries several typical shapes)."""
    if not design_doc:
        return []
    # common fields
    for key in ("rooms", "room_ids"):
        v = design_doc.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())
    # sometimes nested under attrs
    a = (design_doc.get("attrs") or {})
    for key in ("rooms", "room_ids"):
        v = a.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())
    # sometimes nested under 'areas' with 'type' == 'room'
    areas = design_doc.get("areas") or a.get("areas")
    if isinstance(areas, list):
        out = []
        for ar in areas:
            if isinstance(ar, dict) and (ar.get("type") == "room" or ar.get("kind") == "room"):
                rid = ar.get("id") or ar.get("room_id")
                if rid: out.append(rid)
        if out:
            return out
    return []

def find_project_by_name(chunks_root: str, name: str):
    """Scan project/*.json and return (doc, relpath) whose attrs.name == name (case-insensitive)."""
    proj_dir = os.path.join(chunks_root, "project")
    if not os.path.isdir(proj_dir):
        return None, None
    want = (name or "").strip().lower()
    for fn in os.listdir(proj_dir):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(proj_dir, fn)
        try:
            d = read_json(p)
        except Exception:
            continue
        if (d.get("type") == "project") and ((d.get("attrs", {}) or {}).get("name","").strip().lower() == want):
            return d, f"project/{fn}"
    return None, None

def load_floor(chunks_root: str, floor_id_or_rel: str):
    """Load a floor given a bare id or 'floor/<id>.json'. Returns (doc, relpath) or (None, rel)."""
    rel = floor_id_or_rel
    if not rel.endswith(".json"):
        rel = f"{floor_id_or_rel}.json"
    if not rel.startswith("floor/"):
        rel = f"floor/{rel}"
    p = os.path.join(chunks_root, rel)
    if os.path.exists(p):
        try:
            return read_json(p), rel
        except Exception:
            pass
    return None, rel

def collect_room_ids_from_floor(floor_doc: dict, chunks_root: str = None):
    """Extract room IDs from a floor; if absent, walk floor.designs -> design.rooms."""
    if not floor_doc:
        return []
    # Try direct fields first
    for key in ("rooms", "room_ids"):
        v = floor_doc.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())
    a = (floor_doc.get("attrs") or {})
    for key in ("rooms", "room_ids"):
        v = a.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())

    # Walk designs to collect rooms
    if chunks_root:
        designs = floor_doc.get("designs") or a.get("designs") or []
        if isinstance(designs, dict):
            designs = list(designs.values())
        out = []
        for did in designs:
            Ddoc, Drel = load_design(chunks_root, did)
            out.extend(collect_room_ids_from_design(Ddoc))
        if out:
            return out

    return []


def load_room(chunks_root: str, room_id_or_rel: str):
    """Accept either a bare room id or 'room/<id>.json'. Returns (doc, relpath) or (None, rel)."""
    rel = room_id_or_rel
    if not rel.endswith(".json"):
        rel = f"{room_id_or_rel}.json"
    if not rel.startswith("room/"):
        rel = f"room/{rel}"
    p = os.path.join(chunks_root, rel)
    if os.path.exists(p):
        try:
            return read_json(p), rel
        except Exception:
            pass
    return None, rel

def read_room_items(room_doc, chunks_root):
    """Return a set of (name,brand,type_guess) triples for a room document."""
    return set(item_keys_for_room(room_doc, chunks_root))

def load_project(chunks_root: str, project_name: str):
    """Load project/{project_name}.json if present; else return None."""
    p = os.path.join(chunks_root, "project", f"{project_name}.json")
    if os.path.exists(p):
        try:
            return read_json(p)
        except Exception:
            pass
    return None

def room_rel(room_id: str) -> str:
    return f"room/{room_id}.json"


def load_json(root, rel):
    p = os.path.join(root, rel)
    with open(p, "r") as f:
        return json.load(f)

def load_chunk(chunks_root, rel):
    return load_json(chunks_root, rel)

def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def load_index(index_dir: str):
    vecs = np.load(os.path.join(index_dir, "vectors.npy"))
    meta = []
    with open(os.path.join(index_dir, "meta.jsonl"), "r") as f:
        for line in f:
            meta.append(json.loads(line))
    return vecs.astype(np.float32), meta


# ---------------- Embeddings / similarity ----------------
def embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    emb = r.json().get("embedding")
    if not emb:
        raise RuntimeError("no embedding from Ollama")
    return np.array(emb, dtype=np.float32)

def cosine(q: np.ndarray, mat: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-8)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return (mn @ qn).astype(np.float32)

# ---------------- Chunk utilities ----------------
def chunk_text_for_embed(d: Dict[str,Any]):
    return (d.get("summary") or d.get("doc") or json.dumps(d.get("attrs", {}))).strip()

def item_keys_for_room(room_doc, chunks_root):
    keys = []
    for iid in room_doc.get("items", []):
        ipath = os.path.join(chunks_root, "item", f"{iid}.json")
        try:
            it = read_json(ipath).get("attrs", {})
        except Exception:
            it = {}
        name = it.get("name") or ""
        brand = it.get("brand") or ""
        tguess = it.get("type_guess") or it.get("category") or ""
        keys.append((name.lower(), brand.lower(), tguess.lower()))
    return keys

def chunk_full_path(chunks_root: str, relpath: str) -> str:
    return os.path.join(chunks_root, relpath)

def fetch_chunk_text(chunks_root: str, meta_row: Dict[str,Any]) -> str:
    full = chunk_full_path(chunks_root, meta_row["path"])
    d = read_json(full)
    text = d.get("summary") or d.get("doc") or ""
    attrs = d.get("attrs",{}) or {}
    title = meta_row.get("title","")
    crumb = meta_row.get("breadcrumb","")
    bits = [f"[title] {title}", f"[where] {crumb}", text]
    for k in ("name","label","role","brand","type_guess","color"):
        if attrs.get(k): bits.append(f"{k}: {attrs[k]}")
    return "\n".join(bits).strip()

def attrs_for(chunks_root: str, meta_row: Dict[str,Any]) -> Dict[str,Any]:
    full = chunk_full_path(chunks_root, meta_row["path"])
    d = read_json(full)
    return d.get("attrs",{}) or {}

# ---------------- Filters ----------------
def passes_filters(chunks_root: str, m: Dict[str,Any], type_whitelist: List[str], kv_filters: List[str]) -> bool:
    if type_whitelist and m.get("type") not in type_whitelist:
        return False
    if not kv_filters:
        return True
    a = attrs_for(chunks_root, m)
    hay = {**{k.lower(): str(v).lower() for k,v in a.items()},
           "_title": (m.get("title") or "").lower(),
           "_breadcrumb": (m.get("breadcrumb") or "").lower()}
    for spec in kv_filters:
        if ":" not in spec:
            needle = spec.strip().lower()
            if not any(needle in str(v) for v in hay.values()):
                return False
            continue
        k,v = spec.split(":",1)
        k, v = k.strip().lower(), v.strip().lower()
        found = False
        for hk, hv in hay.items():
            if hk==k and v in hv:
                found = True; break
        if not found:
            return False
    return True

# ---------------- Chat with Ollama (with fallback) ----------------
def chat_ollama(model: str, system: str, user: str) -> str:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def _consume_ndjson(resp) -> str:
        out = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "message" in obj and isinstance(obj["message"], dict):
                piece = obj["message"].get("content")
                if piece: out.append(piece)
            if "response" in obj and isinstance(obj["response"], str):
                out.append(obj["response"])
            if obj.get("done") is True:
                break
        return "".join(out).strip()

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model,
                  "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                  ],
                  "stream": True,
                  "options": {"temperature": 0.2}},
            stream=True, timeout=300)
        if r.status_code == 404:
            raise FileNotFoundError("chat endpoint missing")
        r.raise_for_status()
        return _consume_ndjson(r)
    except FileNotFoundError:
        prompt = f"System:\n{system}\n\nUser:\n{user}"
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True, "options": {"temperature": 0.2}},
            stream=True, timeout=300)
        if r.status_code >= 400:
            try: err = r.json().get("error")
            except Exception: err = r.text
            raise RuntimeError(f"Ollama /api/generate error {r.status_code}: {err}")
        return _consume_ndjson(r)


def _json_skim(text: str):
    """Try to pull a JSON object from a chat response (robust to extra prose)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def interpret_assist_intent(model: str, user_text: str) -> dict:
    sys = "Extract structured intent for a retrieval+comparison task over interior-design rooms."
    user = f"""
User request:
{user_text}

Return ONLY a compact JSON with keys:
- item_keywords: string (e.g. "sofa, lamp, white")
- k: integer number of rooms to shortlist (default 2)
- style_prefs: string (e.g. "classic", "mid-century", "modern")
- constraints: string (optional; anything else user cares about)
Example:
{{"item_keywords":"sofa","k":2,"style_prefs":"classic","constraints":""}}
"""
    raw = chat_ollama(model, sys, user)
    data = _json_skim(raw)
    # defaults
    if not isinstance(data, dict): data = {}
    data.setdefault("item_keywords", "sofa")
    data.setdefault("k", 2)
    try:
        data["k"] = max(1, int(data["k"]))
    except Exception:
        data["k"] = 2
    data.setdefault("style_prefs", "")
    data.setdefault("constraints", "")
    return data

def tokenize_keywords(s: str):
    return [t.strip().lower() for t in re.split(r"[,\s]+", s or "") if t.strip()]

def item_matches_tokens(attrs: dict, tokens: list[str]) -> bool:
    hay = " ".join(str(attrs.get(k, "")) for k in ("name","brand","type_guess","category","color","sku")).lower()
    return all(t in hay for t in tokens)

def scan_rooms_with(chunks_root: str, tokens: list[str]):
    """Return list of rows: (room_rel, room_doc, sample_item_name) for rooms that match."""
    room_dir = os.path.join(chunks_root, "room")
    out = []
    if not os.path.isdir(room_dir): return out
    for fn in os.listdir(room_dir):
        if not fn.endswith(".json"): continue
        r_rel = f"room/{fn}"
        try:
            r_doc = read_json(os.path.join(room_dir, fn))
        except Exception:
            continue
        sample = None
        for iid in (r_doc.get("items") or []):
            ipath = os.path.join(chunks_root, "item", f"{iid}.json")
            if not os.path.isfile(ipath): continue
            try:
                it = read_json(ipath)
            except Exception:
                continue
            attrs = (it.get("attrs") or {})
            if item_matches_tokens(attrs, tokens):
                sample = attrs.get("name") or attrs.get("sku") or attrs.get("type_guess") or "item"
                break
        if sample:
            out.append((r_rel, r_doc, sample))
    return out

def score_room_for_style_llm(model: str, room_doc: dict, chunks_root: str, style: str) -> float:
    """Ask the LLM to score 0–10 for how well this room fits the style."""
    # Build a tiny description the model can judge from
    a = (room_doc.get("attrs") or {})
    items = []
    for iid in (room_doc.get("items") or [])[:12]:
        ipath = os.path.join(chunks_root, "item", f"{iid}.json")
        if os.path.isfile(ipath):
            it = read_json(ipath)
            ia = (it.get("attrs") or {})
            items.append(f"{ia.get('name','?')} | {ia.get('brand','?')} | {ia.get('type_guess', ia.get('category',''))}")
    sys = "You are a precise style scorer."
    user = f"""Rate from 0 to 10 how well this room fits the style "{style}". Only return a number.
Room name: {(a.get('name') or a.get('label') or 'room')}
Items:
- """ + "\n- ".join(items[:12])
    raw = chat_ollama(model, sys, user).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    try:
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0

def pick_top_k_rooms(model: str, rows: list[tuple], chunks_root: str, k: int, style: str):
    """rows = [(room_rel, room_doc, sample_item)], returns top-k sorted by style score desc."""
    if not rows: return []
    if not style:
        # no style preference, just take first k
        return rows[:k]
    scored = []
    for (r_rel, r_doc, sample) in rows:
        s = score_room_for_style_llm(model, r_doc, chunks_root, style)
        scored.append((s, r_rel, r_doc, sample))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(r_rel, r_doc, sample) for (s, r_rel, r_doc, sample) in scored[:k]]

def compare_two_room_docs(chunks_root: str, RA: dict, a_rel: str, RB: dict, b_rel: str):
    """Build the same exp dict your --compare flow produces for rooms."""
    va = embed(chunk_text_for_embed(RA))
    vb = embed(chunk_text_for_embed(RB))
    cos = float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8))

    Ia = set(item_keys_for_room(RA, chunks_root))
    Ib = set(item_keys_for_room(RB, chunks_root))
    shared = Ia & Ib
    only_a = Ia - Ib
    only_b = Ib - Ia
    union = Ia | Ib
    jacc = (len(shared) / len(union)) if union else 0.0
    aggA = aggregate_brand_type(Ia)
    aggB = aggregate_brand_type(Ib)
    rel  = relaxed_overlap(Ia, Ib)

    return {
        "kind": "compare",
        "a_rel": a_rel, "b_rel": b_rel,
        "A": summarise_room(chunks_root, a_rel),
        "B": summarise_room(chunks_root, b_rel),
        "cosine": cos,
        "overlap_shared": len(shared),
        "overlap_union": len(union),
        "jaccard": jacc,
        "count_a": len(Ia), "count_b": len(Ib),
        "shared": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(shared)[:50]],
        "only_a": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_a)[:50]],
        "only_b": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_b)[:50]],
        "aggA": aggA, "aggB": aggB, "relaxed": rel,
    }

def recommend_between_two_rooms(model: str, exp: dict, style: str) -> str:
    """Ask the LLM to pick one and say why, with a couple of tweaks to match the style."""
    sys = "You are a retail/design assistant. Be decisive and practical."
    user = f"""
User style: {style or 'unspecified'}

Two rooms were compared. Choose ONE room for the user and justify in 3–5 bullets.
Then give 2–3 quick tweaks to better fit the user's style.

Comparison summary (machine):
- Cosine: {exp.get('cosine')}
- Exact overlap: {exp.get('overlap_shared')}/{exp.get('overlap_union')} (Jaccard {exp.get('jaccard')})
- A name: {exp.get('A',{}).get('title')}
- B name: {exp.get('B',{}).get('title')}
- Shared items (sample): {', '.join(exp.get('shared',[])[:6])}
- A uniques (sample): {', '.join(exp.get('only_a',[])[:4])}
- B uniques (sample): {', '.join(exp.get('only_b',[])[:4])}

Return concise markdown with a final line: Recommendation: A or B.
"""
    return chat_ollama(model, sys, user).strip()


# ---------------- Prompt scaffolding ----------------
SYSTEM_PROMPT = """You are a helpful assistant that answers strictly using the provided context chunks from a Floorplanner export.
If the answer is not in the context, say so and suggest what to search for. Cite chunk indices like [#3]."""

USER_TEMPLATE = """Question:
{q}

Context:
{ctx}

Instructions:
- Cite chunk indices like [#3] when used.
- Prefer items/rooms/designs that match the user's intent.
- If conflicting info appears, mention it concisely."""

def build_context(chunks_root: str, meta: List[Dict[str,Any]], idxs: List[int]) -> str:
    blobs = []
    for rank, i in enumerate(idxs, start=1):
        m = meta[i]
        chunk_text = fetch_chunk_text(chunks_root, m)
        header = f"[#{rank}] ({m['type']}) {m['title']} — {m['path']}"
        blobs.append(header + "\n" + chunk_text)
    return "\n\n".join(blobs)

def topk_indices(vecs: np.ndarray, sims: np.ndarray, k: int) -> List[int]:
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()

# ---------------- Pretty formatters / exporters ----------------
def _safe(x, n=120):
    s = str(x or "")
    return s if len(s) <= n else s[:n-1] + "…"

def summarise_room(chunks_root, relpath):
    try:
        d = read_json(os.path.join(chunks_root, relpath))
        a = (d.get("attrs") or {})
        name = a.get("name") or a.get("label") or "room"
        role = a.get("role")
        breadcrumb = d.get("breadcrumb", "")
        title = f"{name}" + (f" ({role})" if role else "")
        items = d.get("items") or []
        names = []
        for iid in items[:5]:
            ip = os.path.join(chunks_root, "item", f"{iid}.json")
            it = read_json(ip) if os.path.exists(ip) else {}
            ia = (it.get("attrs") or {})
            nm = ia.get("name") or ia.get("sku") or ia.get("type_guess") or ia.get("category")
            if nm: names.append(nm)
        return {
            "title": title,
            "role": role,
            "project": a.get("project"),
            "floor": a.get("floor"),
            "design": a.get("design"),
            "breadcrumb": breadcrumb,
            "items_count": len(items),
            "sample_items": names
        }
    except Exception:
        return {"title":"room", "items_count": None, "sample_items":[]}
def generate_completion_brief(model: str, payload: dict, style: str | None, user_hint: str | None) -> str:
    seed = payload.get("seed", {}) or {}
    recs = payload.get("recommendations", []) or []
    nbrs = payload.get("neighbors_used", []) or []

    # compact lines for the prompt
    seed_line = seed.get("title","room")
    rec_lines = []
    for r in recs:
        cat = (r.get("type") or "—")
        br  = r.get("brand") or ""
        seen = r.get("seen_in_neighbors", 0)
        ex   = r.get("example_name") or ""
        cites = ", ".join(x for x in [r.get("example_room"), r.get("example_item")] if x)
        rec_lines.append(f"- {cat} / {br} (seen {seen}); e.g. {ex} [{cites}]")
    nbr_lines = [f"- {n['room'].get('title','room')} [{n['path']}]"
                 for n in nbrs[:8]]

    sys = (
        "You are an interior designer. Write a concise, practical plan. "
        "Be specific on items, materials, palette, and layout. Keep it to 10–14 bullets, "
        "then a short shopping list. Include brief citations to example room/item paths."
    )
    user = f"""
Style: {style or 'unspecified'}
User notes: {user_hint or 'n/a'}

Seed room: {seed_line}

Suggested complements (data-derived):
{chr(10).join(rec_lines) or '(none)'}

Neighbor rooms considered (for visual precedent):
{chr(10).join(nbr_lines) or '(none)'}

Instructions:
- Propose a coherent classic-style living room completion for the seed.
- Group bullets by: Layout, Key Pieces, Textiles, Lighting, Finishing Touches.
- Justify choices briefly (why they fit classic + the seed).
- End with a compact shopping list (Category — Example — Why), with citations in backticks to example paths.
- Keep it actionable; avoid fluff.
""".strip()

    return chat_ollama(model, sys, user)



def to_markdown(obj):
    def _fmt_counts(pairs):
        return "\n".join(f"- {k}: {c}" for (k, c) in (pairs or [])) or "(none)"

    kind = obj.get("kind")

    # ---------- RAG Q&A ----------
    if kind == "rag":
        md = f"## Q&A\n\n**Question**\n\n{obj['question']}\n\n**Answer**\n\n{obj['answer']}\n\n**Sources**\n"
        for i, s in enumerate(obj["sources"], 1):
            md += f"- [#{i}] `{s['type']}` — {_safe(s.get('title',''))}  \n  `{s.get('path','')}`\n"

            # NEW: Sofa evidence (if present)
            se = s.get("sofa_evidence")
            if isinstance(se, dict):
                hits = se.get("hits") or []
                if hits:
                    md += "   - **Sofa evidence:**\n"
                    for h in hits:
                        nm = h.get("name","?"); cat = h.get("category","?"); br = h.get("brand","?")
                        pid = h.get("productId")
                        pid_txt = f" [{pid}]" if pid else ""
                        md += f"     • {nm} ({cat}, {br}){pid_txt}\n"

            # NEW: Style tag (if present)
            st = s.get("style_tag")
            if isinstance(st, dict):
                lab = st.get("label","?")
                try:
                    sc = float(st.get("score", 0.0))
                    sc_txt = f"{sc:.2f}"
                except Exception:
                    sc_txt = str(st.get("score","0"))
                why = st.get("why","")
                if lab or sc_txt or why:
                    md += f"   - **Style:** {lab} (score={sc_txt})"
                    if why:
                        md += f" — {why}"
                    md += "\n"

        if obj.get("insight_text"):
            md += f"\n---\n\n### LLM Insight\n{obj['insight_text']}\n"
        return md

    # ---------- Similarity results ----------
    if kind == "similar":
        md = f"## Similarity Results\n\n**Query**: {_safe(obj.get('query',''))}\n\n"
        for i, row in enumerate(obj.get("results", []), 1):
            score = row.get("score")
            score_txt = f" (_score: {score:.4f}_)" if isinstance(score, (float,int)) else ""
            md += f"{i}. `{row.get('type','')}` — {_safe(row.get('title',''))}{score_txt}  \n`{row.get('path','')}`\n"
            r = row.get("room")
            if r:
                md += f"   - **Room**: {r.get('title','room')} ({r.get('role','') or ''})\n"
                if r.get("project"): md += f"   - **Project**: {r['project']}\n"
                if r.get("floor"):   md += f"   - **Floor**: {r['floor']}\n"
                if r.get("design"):  md += f"   - **Design**: {r['design']}\n"
                if r.get("breadcrumb"): md += f"   - **Breadcrumb**: {r['breadcrumb']}\n"
                md += f"   - **Items**: {r.get('items_count',0)}"
                sample = r.get("sample_items") or []
                if sample:
                    md += f"  (sample: {', '.join(map(_safe, sample))})"
                md += "\n"
        if obj.get("insight_text"):
            md += f"\n---\n\n### LLM Insight\n{obj['insight_text']}\n"
        return md

    # ---------- Compare (rooms/projects) ----------
    if kind == "compare":
        a, b = obj.get("A", {}), obj.get("B", {})
        md = "## Comparison\n\n"

        if obj.get("exec_summary"):
            es = obj["exec_summary"]
            md += f"**Exec Summary:** overlap {es['overlap']} (Jaccard {es['jaccard']:.2f}). {es['headline']}\n\n"

        md += f"**A**: {a.get('title','A')}  \n`{obj.get('a_rel','')}`\n"
        if a.get("project"): md += f"- Project: {a['project']}\n"
        if a.get("floor"):   md += f"- Floor: {a['floor']}\n"
        if a.get("design"):  md += f"- Design: {a['design']}\n"
        md += "\n"

        md += f"**B**: {b.get('title','B')}  \n`{obj.get('b_rel','')}`\n"
        if b.get("project"): md += f"- Project: {b['project']}\n"
        if b.get("floor"):   md += f"- Floor: {b['floor']}\n"
        if b.get("design"):  md += f"- Design: {b['design']}\n"
        md += "\n"

        cos = obj.get("cosine", None)
        if isinstance(cos, (float,int)):
            md += f"**Cosine (summary/doc)**: {cos:.3f}  \n"
        md += f"**Item overlap**: {obj.get('overlap_shared',0)} shared / {obj.get('overlap_union',0)} union"
        jacc = obj.get("jaccard", None)
        if isinstance(jacc, (float,int)):
            md += f"  (Jaccard {jacc:.3f})"
        md += "  \n"
        md += f"**Counts**: |A|={obj.get('count_a',0)}  |B|={obj.get('count_b',0)}\n\n"

        if obj.get("shared"):
            md += "**Shared items (sample):**\n"
            for n in obj["shared"][:10]:
                md += f"- {n}\n"
        if obj.get("only_a"):
            md += "\n**Unique to A (sample):**\n"
            for n in obj["only_a"][:10]:
                md += f"- {n}\n"
        if obj.get("only_b"):
            md += "\n**Unique to B (sample):**\n"
            for n in obj["only_b"][:10]:
                md += f"- {n}\n"

        md += "\n### Brand summary\n"
        md += "**A (top):**\n" + _fmt_counts((obj.get("aggA") or {}).get("brands_top")) + "\n\n"
        md += "**B (top):**\n" + _fmt_counts((obj.get("aggB") or {}).get("brands_top")) + "\n\n"

        md += "### Category summary\n"
        md += "**A (top):**\n" + _fmt_counts((obj.get("aggA") or {}).get("types_top")) + "\n\n"
        md += "**B (top):**\n" + _fmt_counts((obj.get("aggB") or {}).get("types_top")) + "\n\n"

        md += "### Brand×Category pairs (top)\n"
        md += "**A (top):**\n" + _fmt_counts((obj.get("aggA") or {}).get("pairs_top")) + "\n\n"
        md += "**B (top):**\n" + _fmt_counts((obj.get("aggB") or {}).get("pairs_top")) + "\n\n"

        rel = obj.get("relaxed") or {}
        rc  = rel.get("counts") or {}
        md += "### Relaxed overlap (brand×category)\n"
        md += f"- Shared (brand×type, count): {rc.get('shared',0)}\n"
        if rel.get("shared"):
            md += "  - " + ", ".join(f"{b}×{t} ({c})" for (b,t,c) in rel["shared"][:10]) + "\n"
        if rel.get("only_a"):
            md += "- Only in A (brand×type, count):\n"
            md += "  " + ", ".join(f"{b}×{t} ({c})" for (b,t,c,_) in rel["only_a"][:10]) + "\n"
        if rel.get("only_b"):
            md += "- Only in B (brand×type, count):\n"
            md += "  " + ", ".join(f"{b}×{t} ({c})" for (b,t,c,_) in rel["only_b"][:10]) + "\n"

        if obj.get("insight_text"):
            md += f"\n---\n\n### LLM Insight\n{obj['insight_text']}\n"

        return md

    return "Unsupported export."


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="Path to index dir (vectors.npy, meta.jsonl)")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--query", help="Natural language question for RAG")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL)

    # similarity modes
    ap.add_argument("--similar_text", help="Find chunks similar to this text")
    ap.add_argument("--similar_path", help="Find chunks similar to this chunk path (relative under chunks root)")

    # filters
    ap.add_argument("--filter_type", action="append", help="Restrict to types (room,item,design,floor,project); repeatable")
    ap.add_argument("--filter_kv", action="append", help="Key:val contains match (e.g., role:Living, brand:Ethan Allen, type_guess:sofa, color:brown). Repeatable.")
    ap.add_argument("--show", action="store_true", help="Print retrieved chunks (for similarity/debug)")
    ap.add_argument("--compare", help="Comma-separated rel paths to two chunks, e.g. room/AAA.json,room/BBB.json")
    ap.add_argument("--assist", help="Natural-language task: I will search rooms, compare, and recommend (no index required).")


    # exporters
    ap.add_argument("--out", help="Write results to this file (e.g., report.md, results.json, out.txt)")
    ap.add_argument("--fmt", default="md", choices=["md","json","txt"], help="Output format when --out is provided")
    ap.add_argument("--compare_projects", help="Comma-separated project names, e.g. PRESNELL2025,JAGGIE")
    ap.add_argument("--explain", action="store_true", help="Ask the LLM to summarize differences and suggest next actions.")
    ap.add_argument("--explain_model", default=DEFAULT_CHAT_MODEL, help="Model to use when --explain is set.")
    ap.add_argument("--compare_rooms", help="Comma-separated room IDs or room JSON paths to compare two rooms")
    ap.add_argument("--rooms_with", help="Find rooms that contain items matching these keywords (name/brand/type/color/sku). No index required.")
    ap.add_argument("--task", choices=["pick_and_recommend"], help="End-to-end NL flow.")
    ap.add_argument("--style", help="User style preference, e.g. 'classic'", default=None)
    ap.add_argument("--need", help="What the user needs, e.g. 'living room with sofa'", default=None)
    ap.add_argument("--nrooms", type=int, default=2, help="How many rooms to consider (default 2)")
    ap.add_argument("--verify_sofa", action="store_true", help="Call /products/ids to prove sofa presence.")
    ap.add_argument("--editor_version", help="Optional ?editor_version=... for /products/ids", default=None)
    ap.add_argument("--style_tag", help="Ask LLM to tag/score style (e.g., 'classic').", default=None)
    ap.add_argument("--complete_room", help="Room ID or room/<id>.json to complete from co-occurring items (no index required).")
    ap.add_argument("--neighbors", type=int, default=12, help="How many similar rooms to mine for co-occurrences.")
    ap.add_argument("--suggest", type=int, default=6, help="How many item suggestions to return.")
    ap.add_argument(
    "--complete_from_items",
    help="Comma-separated item IDs or item/*.json paths to use as a virtual seed (no file needed)."
)
    # near other argparse adds
    ap.add_argument("--complete_nl", help="Freeform instruction to turn completion suggestions into a natural-language plan (markdown).")






    args = ap.parse_args()

    # ---------- EARLY: compare mode (no index needed) ----------
    if args.compare:
        a_rel, b_rel = [s.strip() for s in args.compare.split(",", 1)]
        A = load_chunk(args.chunks, a_rel)
        B = load_chunk(args.chunks, b_rel)

        va = embed(chunk_text_for_embed(A))
        vb = embed(chunk_text_for_embed(B))
        cos = float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8))

        def collect_items(doc):
            if doc.get("type") == "room":
                return set(item_keys_for_room(doc, args.chunks))
            return set()

        Ia, Ib = collect_items(A), collect_items(B)
        shared = Ia & Ib
        only_a = Ia - Ib
        only_b = Ib - Ia
        union = Ia | Ib
        jacc = (len(shared) / len(union)) if union else 0.0
        # NEW: roll-ups and relaxed overlap
        aggA = aggregate_brand_type(Ia)
        aggB = aggregate_brand_type(Ib)
        rel  = relaxed_overlap(Ia, Ib)


        # Build export object
        exp = {
            "kind": "compare",
            "a_rel": a_rel, "b_rel": b_rel,
            "A": summarise_room(args.chunks, a_rel),
            "B": summarise_room(args.chunks, b_rel),
            "cosine": cos,
            "overlap_shared": len(shared),
            "overlap_union": len(union),
            "jaccard": jacc,
            "count_a": len(Ia), "count_b": len(Ib),
            "shared": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(shared)[:50]],
            "only_a": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_a)[:50]],
            "only_b": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_b)[:50]],
            "aggA": aggA,
            "aggB": aggB,
            "relaxed": rel,

        }

        # Console summary
        print("=== Comparison ===")
        print(f"A: {exp['A']['title']}  →  {a_rel}")
        print(f"B: {exp['B']['title']}  →  {b_rel}")
        print(f"Cosine similarity (summaries/docs): {exp['cosine']:.3f}")
        print(f"Item overlap: {exp['overlap_shared']} shared / {exp['overlap_union']} union  (Jaccard: {exp['jaccard']:.3f})")
        print(f"Item counts:  |A|={exp['count_a']}  |B|={exp['count_b']}")
        if exp["shared"]:
            print("\nShared items (sample):")
            for n in exp["shared"][:10]: print(f"- {n}")
        if exp["only_a"]:
            print("\nUnique to A (sample):")
            for n in exp["only_a"][:10]: print(f"- {n}")
        if exp["only_b"]:
            print("\nUnique to B (sample):")
            for n in exp["only_b"][:10]: print(f"- {n}")
                # Optional LLM insight
        if args.explain:
            insight = generate_insight(args.explain_model, exp, style=args.style if hasattr(args, "style") else None)
            exp["insight_text"] = insight
            print("\n---\nLLM Insight:\n" + insight)

        # File export
            # Export
        if args.out:
            # Enrich with sofa/style tags when requested
            if args.verify_sofa or args.style_tag:
                for s in srcs:
                    if s.get("type") == "room":
                        room_doc = load_json(args.chunks, s["path"])
                        if args.verify_sofa:
                            s["sofa_evidence"] = sofa_evidence_for_room(
                                args.chunks, room_doc, editor_version=args.editor_version
                            )
                        if args.style_tag:
                            s["style_tag"] = tag_style_for_room(
                                args.model, room_doc, args.chunks, preferred=args.style_tag
                            )

            exp = {"kind":"rag", "question": args.query, "answer": answer, "sources": srcs}
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")



        # ---------- EARLY: compare_projects mode (no index needed) ----------
        # ---------- EARLY: compare_projects mode (no index needed) ----------
        # ---------- EARLY: compare_projects mode (no index needed) ----------
    if args.compare_projects:
        projA_name, projB_name = [s.strip() for s in args.compare_projects.split(",", 1)]

        def collect_project_items_by_name(project_name: str):
            items = set()

            # A) Find the project doc by human name (attrs.name)
            P, _P_rel = find_project_by_name(args.chunks, project_name)

            # B1) Preferred traversal: project -> floors -> (designs) -> rooms -> items
            if P:
                floors = P.get("floors") or (P.get("attrs", {}) or {}).get("floors") or []
                if isinstance(floors, dict):
                    floors = list(floors.values())
                for fid in floors:
                    Fdoc, _ = load_floor(args.chunks, fid)
                    if not Fdoc:
                        continue
                    room_ids = collect_room_ids_from_floor(Fdoc, chunks_root=args.chunks)
                    for rid in room_ids:
                        Rdoc, _ = load_room(args.chunks, rid)
                        if Rdoc:
                            items |= set(item_keys_for_room(Rdoc, args.chunks))

            # B2) Fallback: scan rooms whose breadcrumb mentions the project name
            if not items:
                proj_lower = project_name.lower()
                room_root = os.path.join(args.chunks, "room")
                if os.path.isdir(room_root):
                    for fn in os.listdir(room_root):
                        if not fn.endswith(".json"):
                            continue
                        p = os.path.join(room_root, fn)
                        try:
                            d = read_json(p)
                        except Exception:
                            continue
                        crumb = (d.get("breadcrumb") or "").lower()
                        if f"project:{proj_lower}" in crumb or proj_lower in crumb:
                            items |= set(item_keys_for_room(d, args.chunks))

            return items

        Ia = collect_project_items_by_name(projA_name)
        Ib = collect_project_items_by_name(projB_name)

        shared = Ia & Ib
        only_a = Ia - Ib
        only_b = Ib - Ia
        union = Ia | Ib
        jacc = (len(shared) / len(union)) if union else 0.0

        exp = {
            "kind": "compare",
            "a_rel": projA_name, "b_rel": projB_name,
            "A": {"title": projA_name, "items_count": len(Ia)},
            "B": {"title": projB_name, "items_count": len(Ib)},
            "cosine": None,  # not meaningful for whole projects
            "overlap_shared": len(shared),
            "overlap_union": len(union),
            "jaccard": jacc,
            "count_a": len(Ia), "count_b": len(Ib),
            "shared": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(shared)[:50]],
            "only_a": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_a)[:50]],
            "only_b": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_b)[:50]],
        }
        if args.explain:
            insight = generate_insight(args.explain_model, exp)
            exp["insight_text"] = insight
            print("\n---\nLLM Insight:\n" + insight)


        # Console print: show samples even when no overlap
        print("=== Project Comparison ===")
        print(f"{projA_name}: {len(Ia)} items")
        print(f"{projB_name}: {len(Ib)} items")
        print(f"Shared: {len(shared)} / {len(union)} (Jaccard {jacc:.3f})")

        print("\nTop items in", projA_name, "(sample):")
        print(_sample_item_lines(Ia))

        print("\nTop items in", projB_name, "(sample):")
        print(_sample_item_lines(Ib))


        if shared:
            print("\nShared items (sample):")
            for n in exp["shared"][:10]:
                print("-", n)

        if args.out:
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")
        return


    # ---------- EARLY: compare_rooms mode (no index needed) ----------
    if args.compare_rooms:
        roomA_id, roomB_id = [s.strip() for s in args.compare_rooms.split(",", 1)]
        RA, a_rel = load_room(args.chunks, roomA_id)
        RB, b_rel = load_room(args.chunks, roomB_id)
        if not RA or not RB:
            raise SystemExit(f"Could not load both rooms:\n  A: {roomA_id} -> {a_rel if RA else 'NOT FOUND'}\n  B: {roomB_id} -> {b_rel if RB else 'NOT FOUND'}")

        va = embed(chunk_text_for_embed(RA))
        vb = embed(chunk_text_for_embed(RB))
        cos = float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8))

        Ia = set(item_keys_for_room(RA, args.chunks))
        Ib = set(item_keys_for_room(RB, args.chunks))
        shared = Ia & Ib
        only_a = Ia - Ib
        only_b = Ib - Ia
        union = Ia | Ib
        jacc = (len(shared) / len(union)) if union else 0.0

        aggA = aggregate_brand_type(Ia)
        aggB = aggregate_brand_type(Ib)
        rel  = relaxed_overlap(Ia, Ib)

        exp = {
            "kind": "compare",
            "a_rel": a_rel, "b_rel": b_rel,
            "A": summarise_room(args.chunks, a_rel),
            "B": summarise_room(args.chunks, b_rel),
            "cosine": cos,
            "overlap_shared": len(shared),
            "overlap_union": len(union),
            "jaccard": jacc,
            "count_a": len(Ia), "count_b": len(Ib),
            "shared": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(shared)[:50]],
            "only_a": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_a)[:50]],
            "only_b": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(only_b)[:50]],
            "aggA": aggA, "aggB": aggB, "relaxed": rel,
        }

        print("=== Room Comparison ===")
        print(f"A: {exp['A']['title']}  →  {a_rel}")
        print(f"B: {exp['B']['title']}  →  {b_rel}")
        print(f"Cosine similarity (summaries/docs): {exp['cosine']:.3f}")
        print(f"Item overlap: {len(shared)} / {len(union)} (Jaccard {jacc:.3f})")
        print(f"Item counts: |A|={len(Ia)} |B|={len(Ib)}")
        if shared:
            print("\nShared items (sample):")
            for n in exp["shared"][:10]:
                print("-", n)

        if args.explain:
            insight = generate_insight(args.explain_model, exp)
            exp["insight_text"] = insight
            print("\n---\nLLM Insight:\n" + insight)

        if args.out:
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")
        raise SystemExit(0)
    if args.complete_from_items:
        seed_doc, seed_rel = seed_doc_from_items(args.chunks, args.complete_from_items)
        payload = complete_room_from_neighbors(
            args.chunks, seed_doc,
            neighbors=args.neighbors if hasattr(args, "neighbors") else 12,
            topn=args.suggest if hasattr(args, "suggest") else 6,
            style=args.style if hasattr(args, "style") else None,
        )
        nl_md = ""
        if args.complete_nl:
            try:
                nl_md = generate_completion_brief(args.model, payload, style=args.style, user_hint=args.complete_nl)
            except Exception as e:
                nl_md = f"(could not generate NL plan: {e})"


        print("=== Room Completion (virtual seed) ===")
        print("Seed:", payload["seed"].get("title","room"), "→", seed_rel)
        for r in payload["recommendations"]:
            line = f"- {r['type'] or '—'}"
            if r["brand"]: line += f" / {r['brand']}"
            line += f"  (seen in {r['seen_in_neighbors']} neighbors)"
            if r["example_name"]: line += f"  e.g. {r['example_name']}"
            print(line)

        if args.out:
            md = ["## Room Completion",
                f"**Seed:** `{seed_rel}` — {payload['seed'].get('title','room')}",
                "",
                "### Suggestions"]
            for r in payload["recommendations"]:
                row = f"- **{r['type'] or '—'}**"
                if r["brand"]: row += f" / *{r['brand']}*"
                row += f" — seen in **{r['seen_in_neighbors']}** neighbor rooms"
                cites = []
                if r["example_room"]: cites.append(f"`{r['example_room']}`")
                if r["example_item"]: cites.append(f"`{r['example_item']}`")
                if r["example_name"]: row += f"; e.g. *{r['example_name']}*"
                if cites: row += "  \n  sources: " + ", ".join(cites)
                md.append(row)
            md.append("")
            md.append("### Neighbors considered")
            for n in payload["neighbors_used"]:
                md.append(f"- `{n['path']}` — {n['room'].get('title','room')} (shared pairs: {n['shared_pairs']})")
            with open(args.out, "w") as f:
                f.write("\n".join(md) if (args.fmt if hasattr(args,"fmt") else "md")=="md"
                        else "\n".join(md).replace("**",""))
            print(f"\nSaved to {args.out}")
        raise SystemExit(0)

    if args.complete_room:
        seed_doc, seed_rel = load_room(args.chunks, args.complete_room)
        if not seed_doc:
            raise SystemExit(f"Seed room not found: {seed_rel}")

        # compute!
        payload = complete_room_from_neighbors(
            args.chunks, seed_doc,
            neighbors=args.neighbors, topn=args.suggest, style=args.style
        )

        # pretty print
        print("=== Room Completion ===")
        print("Seed:", payload["seed"].get("title","room"), "→", seed_rel)
        for r in payload["recommendations"]:
            line = f"- {r['type'] or '—'}"
            if r["brand"]: line += f" / {r['brand']}"
            line += f"  (seen in {r['seen_in_neighbors']} neighbors)"
            if r["example_name"]: line += f"  e.g. {r['example_name']}"
            print(line)

        if args.out:
            # quick md block rather than touching to_markdown everywhere
            md = ["## Room Completion",
                f"**Seed:** `{seed_rel}` — {payload['seed'].get('title','room')}",
                "",
                "### Suggestions"]
            for r in payload["recommendations"]:
                row = f"- **{r['type'] or '—'}**"
                if r["brand"]: row += f" / *{r['brand']}*"
                row += f" — seen in **{r['seen_in_neighbors']}** neighbor rooms"
                cites = []
                if r["example_room"]: cites.append(f"`{r['example_room']}`")
                if r["example_item"]: cites.append(f"`{r['example_item']}`")
                if r["example_name"]: row += f"; e.g. *{r['example_name']}*"
                if cites: row += "  \n  sources: " + ", ".join(cites)
                md.append(row)
            md.append("")
            md.append("### Neighbors considered")
            for n in payload["neighbors_used"]:
                md.append(f"- `{n['path']}` — {n['room'].get('title','room')} (shared pairs: {n['shared_pairs']})")

            with open(args.out, "w") as f:
                f.write("\n".join(md) if args.fmt=="md" else "\n".join(md).replace("**",""))
            print(f"\nSaved to {args.out}")
        raise SystemExit(0)

    
    # ---------- EARLY: rooms_with (no index needed; scans items directly) ----------
    if args.rooms_with:
        query = args.rooms_with.strip().lower()
        tokens = [t for t in query.replace(",", " ").split() if t]

        def item_matches(attrs: dict) -> bool:
            hay = " ".join(str(attrs.get(k, "")) for k in ("name","brand","type_guess","category","color","sku")).lower()
            return all(tok in hay for tok in tokens)

        room_dir = os.path.join(args.chunks, "room")
        if not os.path.isdir(room_dir):
            raise SystemExit(f"No room directory at {room_dir}")

        seen = set()
        rows = []  # (room_name, room_rel, sample_item, item_rel)
        for fn in os.listdir(room_dir):
            if not fn.endswith(".json"): continue
            r_rel = f"room/{fn}"
            r_doc = read_json(os.path.join(room_dir, fn))
            item_ids = r_doc.get("items") or []
            sample = None
            for iid in item_ids:
                ipath = os.path.join(args.chunks, "item", f"{iid}.json")
                if not os.path.isfile(ipath): continue
                it = read_json(ipath)
                attrs = (it.get("attrs") or {})
                if item_matches(attrs):
                    if not sample:
                        sample = attrs.get("name") or attrs.get("sku") or attrs.get("type_guess") or "item"
                    # record once per room
                    if r_rel not in seen:
                        seen.add(r_rel)
                        a = r_doc.get("attrs") or {}
                        rname = a.get("name") or a.get("label") or "room"
                        rows.append((rname, r_rel, sample, f"item/{iid}.json"))
                    # do not break; but we only collect one sample per room
            # end items loop

        if not rows:
            print(f"No rooms matched: {args.rooms_with}")
        else:
            print(f"=== Rooms with: \"{args.rooms_with}\" ===")
            for i,(rname, rrel, sample, irel) in enumerate(rows, 1):
                rdoc = read_json(os.path.join(args.chunks, rrel))
                attrs = rdoc.get("attrs", {}) or {}
                print(f"{i}. {rname}  →  {os.path.basename(rrel)}")
                if attrs.get("project"): print(f"   project: {attrs['project']}")
                if attrs.get("floor"):   print(f"   floor: {attrs['floor']}")
                if attrs.get("design"):  print(f"   design: {attrs['design']}")
                if rdoc.get("breadcrumb"): print(f"   breadcrumb: {rdoc['breadcrumb']}")
                print(f"   sample item: {sample}")
                print(f"   sources: room={rrel}  item={irel}")

        if args.out:
            exp = {
                "kind": "similar",
                "query": f'rooms_with("{args.rooms_with}")',
                "results": [
                    {"rank": i+1, "type":"room", "title": rname, "path": rrel, "room": summarise_room(args.chunks, rrel)}
                    for i,(rname,rrel,_,_) in enumerate(rows)
                ],
            }
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")
        raise SystemExit(0)
    # ---------- EARLY: natural-language assistant (no index needed) ----------
    if args.assist:
        # 1) Interpret intent
        intent = interpret_assist_intent(args.explain_model, args.assist)
        tokens = tokenize_keywords(intent.get("item_keywords","sofa"))
        k = max(1, int(intent.get("k", 2)))
        style = intent.get("style_prefs","").strip()

        # 2) Find candidate rooms by keywords
        rows = scan_rooms_with(args.chunks, tokens)
        if not rows:
            raise SystemExit(f"No rooms matched the keywords: {', '.join(tokens)}")

        # 3) Pick top-k by style (if provided) using LLM scoring
        chosen = pick_top_k_rooms(args.explain_model, rows, args.chunks, k, style)
        if len(chosen) < 2 and k >= 2 and len(rows) >= 2:
            # fallback: just take first two
            chosen = (rows[:2])

        # 4) If we have at least two, compare the best two and ask for a rec
        if len(chosen) >= 2:
            (a_rel, a_doc, _), (b_rel, b_doc, _) = chosen[0], chosen[1]
            exp = compare_two_room_docs(args.chunks, a_doc, a_rel, b_doc, b_rel)

            # Optional: add your existing LLM “insight”
            if args.explain:
                exp["insight_text"] = generate_insight(args.explain_model, exp)

            # Tailored recommendation
            rec = recommend_between_two_rooms(args.explain_model, exp, style)
            exp["insight_text"] = (exp.get("insight_text","\n").rstrip()+"\n\n---\n### Recommendation\n"+rec).strip()

            # Console
            print("=== NL Assist ===")
            print(f"Query: {args.assist}")
            print(f"Interpreted: keywords={tokens}, k={k}, style='{style}'")
            print(f"Picked rooms:\n- {a_rel}\n- {b_rel}")
            if args.out:
                if args.fmt == "json":
                    with open(args.out, "w") as f: json.dump(exp, f, indent=2)
                else:
                    md = to_markdown(exp)
                    with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
                print(f"\nSaved to {args.out}")
            raise SystemExit(0)

        # 5) If only one room requested, or we only found one, just print it
        print("Found rooms:")
        for (r_rel, r_doc, sample) in chosen:
            a = r_doc.get("attrs",{}) or {}
            print("-", r_rel, "—", a.get("name") or a.get("label") or "room", "| sample:", sample)
        raise SystemExit(0)



    if args.task == "pick_and_recommend":
        # 1) Build a similarity query from NL
        nl = args.query or ""
        if args.need:
            nl += f" need: {args.need}"
        if args.style:
            nl += f" style: {args.style}"

        # 2) We need the index to rank rooms
        if not args.index:
            raise SystemExit("Provide --index for --task pick_and_recommend.")
        vecs, meta = load_index(args.index)

        # Restrict to rooms; loosely require 'living' and 'sofa' if user hinted
        candidates = [i for i,m in enumerate(meta) if m.get("type")=="room"]
        sims_full = cosine(embed(nl), vecs)
        # Soft post-filter by checking chunk text for hints
                # Soft post-filter by checking room text AND item attributes for hints
        need_text = (args.need or "").lower().strip()
        need_tokens = [t for t in re.split(r"[,\s]+", need_text) if t]  # e.g., ["living","room","with","sofa"]

        def ok(i):
            if not need_tokens:
                return True
            m = meta[i]
            # 1) Room-level text (title/breadcrumb/summary)
            room_txt = fetch_chunk_text(args.chunks, m).lower()

            # 2) Item-level text (aggregate item attrs)
            items_txt_parts = []
            try:
                room_doc = load_json(args.chunks, m["path"])
            except Exception:
                room_doc = {}
            for iid in (room_doc.get("items") or []):
                ip = os.path.join(args.chunks, "item", f"{iid}.json")
                if not os.path.isfile(ip):
                    continue
                try:
                    it = read_json(ip).get("attrs", {}) or {}
                except Exception:
                    it = {}
                items_txt_parts.append(" ".join(str(it.get(k,"")) for k in ("name","brand","type_guess","category","color","sku")))
            items_txt = " ".join(items_txt_parts).lower()

            haystack = room_txt + " " + items_txt

            # Heuristic: require ALL “strong” tokens; ignore stop-ish filler like "with"
            weak = {"with","a","an","the","room","rooms","and","or"}
            strong_tokens = [t for t in need_tokens if t not in weak]

            # If everything got filtered out, fall back to original tokens
            tokens = strong_tokens or need_tokens

            return all(t in haystack for t in tokens)


        ranked = [i for i in sorted(candidates, key=lambda j: -sims_full[j]) if ok(i)]
        top = ranked[:args.nrooms]
        if len(top) < 2:
            print("Not enough matches after filtering."); return

        # 3) Compare the top two and produce an LLM explanation
        a_rel = meta[top[0]]["path"]
        b_rel = meta[top[1]]["path"]

        # Reuse your compare flow
        A = load_chunk(args.chunks, a_rel)
        B = load_chunk(args.chunks, b_rel)
        Ia = set(item_keys_for_room(A, args.chunks)) if A.get("type")=="room" else set()
        Ib = set(item_keys_for_room(B, args.chunks)) if B.get("type")=="room" else set()
        shared, union = Ia & Ib, Ia | Ib
        jacc = (len(shared)/len(union)) if union else 0.0

        exp = {
            "kind":"compare",
            "a_rel":a_rel, "b_rel":b_rel,
            "A": summarise_room(args.chunks, a_rel),
            "B": summarise_room(args.chunks, b_rel),
            "cosine": None,
            "overlap_shared": len(shared),
            "overlap_union": len(union),
            "jaccard": jacc,
            "count_a": len(Ia), "count_b": len(Ib),
            "shared": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(shared)[:50]],
            "only_a": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(Ia - Ib)[:50]],
            "only_b": [f"{n or '—'} | {b or '—'} | {t or '—'}" for (n,b,t) in list(Ib - Ia)[:50]],
        }

        # Optional: add roll-ups if you like (aggregate_brand_type, relaxed_overlap)
        exp["aggA"] = aggregate_brand_type(Ia)
        exp["aggB"] = aggregate_brand_type(Ib)
        exp["relaxed"] = relaxed_overlap(Ia, Ib)

        if args.explain:
            exp["insight_text"] = generate_insight(args.explain_model, exp, style=args.style)

        # Print and export
        print("Picked rooms:", a_rel, "vs", b_rel)
        if args.out:
            md = to_markdown(exp) if args.fmt=="md" else json.dumps(exp, indent=2)
            with open(args.out, "w") as f: f.write(md if args.fmt!="txt" else md.replace("**",""))
            print(f"Saved to {args.out}")
        return

    # ---------- From here on, we need the index ----------
    if not args.index:
        raise SystemExit("Provide --index for RAG/similarity modes (or use --compare).")

    vecs, meta = load_index(args.index)

    # Prepare candidate mask from filters
    type_whitelist = args.filter_type or []
    kv_filters = args.filter_kv or []
    candidates = [i for i, m in enumerate(meta) if passes_filters(args.chunks, m, type_whitelist, kv_filters)]
    if not candidates:
        print("No chunks matched the provided filters."); return

    # Embedding for query/similarity
    if args.similar_text:
        qvec = embed(args.similar_text)
    elif args.similar_path:
        full = os.path.join(args.chunks, args.similar_path)
        if not os.path.isfile(full):
            raise SystemExit(f"similar_path not found: {full}")
        d = read_json(full)
        text = d.get("summary") or d.get("doc") or json.dumps(d.get("attrs",{}))
        qvec = embed(text)
    elif args.query:
        qvec = embed(args.query)
    else:
        raise SystemExit("Provide --query (RAG) OR --similar_text/--similar_path (similarity) OR --compare")

    # Similarity on filtered subset
    sims_full = cosine(qvec, vecs)
    sims = np.array([sims_full[i] for i in candidates], dtype=np.float32)
    top_rel = topk_indices(vecs, sims, args.k)
    top_idxs = [candidates[i] for i in top_rel]

    # ---------- Similarity-only mode ----------
    if args.similar_text or args.similar_path:
        results = []
        for rank, i in enumerate(top_idxs, 1):
            m = meta[i]
            row = {
                "rank": rank,
                "score": float(sims_full[i]),
                "type": m["type"],
                "title": m.get("title",""),
                "path": m["path"],
            }
            if m["type"] == "room":
                row["room"] = summarise_room(args.chunks, m["path"])
            results.append(row)
            # console
            print(f"[#{rank}] {row['score']:.4f}  {row['type']:7}  {row['title']}  →  {row['path']}")
            if args.show:
                txt = fetch_chunk_text(args.chunks, m)
                print(txt[:800] + ("..." if len(txt) > 800 else ""))
                print("-"*80)

        if args.out:
            exp = {"kind":"similar", "query": args.similar_text or f"similar_to:{args.similar_path}", "results": results}
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")
        return

    # ---------- Special: item→room grouping ----------
    def parent_room_row(item_meta, chunks_root):
        item_path = item_meta["path"]              # e.g. "item/....json"
        item_doc  = load_json(chunks_root, item_path)
        room_id   = item_doc.get("parent_id")
        if not room_id: return None
        room_path = f"room/{room_id}.json"
        room_doc  = load_json(chunks_root, room_path)
        rattrs    = room_doc.get("attrs", {})
        room_name = rattrs.get("name") or rattrs.get("label") or "room"
        iname     = item_doc.get("attrs", {}).get("name") or (item_doc.get("doc","").split(" | ")[0].replace("name: ",""))
        return (room_name, room_path, iname, item_path)

    if args.filter_type and "item" in args.filter_type:
        seen = set()
        rows = []
        for m in [meta[i] for i in top_idxs]:
            row = parent_room_row(m, args.chunks)
            if not row: continue
            room_name, room_path, item_name, item_path = row
            if room_path in seen: continue
            seen.add(room_path)
            rows.append((room_name, room_path, item_name, item_path))

        print("=== Rooms with matching items ===")
        for i,(rname, rpath, iname, ipath) in enumerate(rows, 1):
            room_doc = load_json(args.chunks, rpath)
            breadcrumb = room_doc.get("breadcrumb", "")
            attrs = room_doc.get("attrs", {}) or {}
            floor = attrs.get("floor")
            design = attrs.get("design")
            project = attrs.get("project")

            print(f"{i}. {rname}  →  {os.path.basename(rpath)}")
            if project: print(f"   project: {project}")
            if floor:   print(f"   floor: {floor}")
            if design:  print(f"   design: {design}")
            if breadcrumb: print(f"   breadcrumb: {breadcrumb}")
            print(f"   sample item: {iname}")
            print(f"   sources: room={rpath}  item={ipath}")

        # Export a simple list if needed
        if args.out:
            exp = {"kind":"similar", "query": args.query or "(items filter)", "results":[
                {"rank":i+1, "type":"room", "title":rname, "path":rpath,
                 "room": summarise_room(args.chunks, rpath)} for i,(rname,rpath,_,_) in enumerate(rows)
            ]}
            if args.fmt == "json":
                with open(args.out, "w") as f: json.dump(exp, f, indent=2)
            else:
                md = to_markdown(exp)
                with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
            print(f"\nSaved to {args.out}")
        return

    # ---------- Default RAG Q&A ----------
    ctx = build_context(args.chunks, meta, top_idxs)
    user = USER_TEMPLATE.format(q=args.query, ctx=ctx)
    answer = chat_ollama(args.model, SYSTEM_PROMPT, user)

    # Sources
    srcs = []
    for rank, i in enumerate(top_idxs, start=1):
        m = meta[i]
        srcs.append({"rank": rank, "type": m["type"], "title": m.get("title",""), "path": m["path"]})

    # Console
    print("=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for s in srcs:
        print(f"[#{s['rank']}] {s['type']:7}  {s['title']}  →  {s['path']}")

    # Export
    if args.out:
        exp = {"kind":"rag", "question": args.query, "answer": answer, "sources": srcs}
        if args.fmt == "json":
            with open(args.out, "w") as f: json.dump(exp, f, indent=2)
        else:
            md = to_markdown(exp)
            with open(args.out, "w") as f: f.write(md if args.fmt=="md" else md.replace("**",""))
        print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
