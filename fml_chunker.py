import json, os, math, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---------- helpers ----------

def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def polygon_area(points: List[Dict[str, float]]) -> float:
    """Shoelace; returns area in screen/plan units^2."""
    n = len(points)
    if n < 3: return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]["x"], points[i]["y"]
        x2, y2 = points[(i+1) % n]["x"], points[(i+1) % n]["y"]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0

def point_in_polygon(x: float, y: float, poly: List[Dict[str, float]]) -> bool:
    """Ray casting; works for simple polygons."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]["x"], poly[i]["y"]
        x2, y2 = poly[(i+1) % n]["x"], poly[(i+1) % n]["y"]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if intersects:
            inside = not inside
    return inside

def units2_to_m2(area_units2: float, use_metric: bool) -> Optional[float]:
    # If your project marks metric, coords are often centimeters; cm² -> m².
    if use_metric:
        return area_units2 / 10000.0
    return None

# crude item-name/type helpers
def _lc(s): return (s or "").lower()

def guess_room_type(item_names: List[str]) -> str:
    names = [_lc(n) for n in item_names]
    joined = " ".join(names)
    # very light heuristic; extend as needed
    if any(k in joined for k in ["bed", "nightstand", "dresser", "wardrobe"]):
        return "bedroom"
    if ("sofa" in joined or "couch" in joined) and ("tv" in joined or "television" in joined):
        return "living"
    if any(k in joined for k in ["stove", "sink", "fridge", "kitchen", "hood", "cabinet"]):
        return "kitchen"
    if any(k in joined for k in ["toilet", "washbasin", "shower", "bathtub", "bath", "wc"]):
        return "washroom"
    if any(k in joined for k in ["desk", "office", "workstation"]):
        return "office"
    return "unknown"

def item_display_name(it: Dict[str, Any]) -> str:
    return it.get("name") or it.get("class_name") or it.get("refid") or "item"

def stable_id_for(kind, **parts):
    import hashlib, json
    key = {"kind": kind, **parts}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

# ---------- normalizers ----------

def normalize_item(it: Dict[str, Any], parent_design_id: str, room_id: Optional[str]) -> Dict[str, Any]:
    chunk_id = stable_id_for("item", parent=parent_design_id, ref=it.get("refid"), xy=(it.get("x"), it.get("y")))
    payload = {
        "chunk_id": chunk_id,
        "level": "item",
        "parent_id": parent_design_id,
        "room_id": room_id,                 # <-- linked if inside a polygon
        "type": it.get("class_name") or "item",
        "name": item_display_name(it),
        "refid": it.get("refid"),
        "position": {k: it.get(k) for k in ("x", "y", "z") if k in it},
        "size": {k: it.get(k) for k in ("width", "height", "z_height") if k in it},
        "rotation": it.get("rotation"),
        "summary_text": f"Item '{item_display_name(it)}' ({it.get('class_name') or 'item'}) at ({it.get('x')}, {it.get('y')}).",
        "refs": {"parent_id": parent_design_id},
        "raw": it
    }
    return payload

def normalize_area_to_room(area: Dict[str, Any],
                           items: List[Dict[str, Any]],
                           use_metric: bool,
                           design_id: str,
                           floor_id: str,
                           project_id: str,
                           design_name: Optional[str]) -> Tuple[Dict[str, Any], List[str]]:
    poly = [{"x": p["x"], "y": p["y"]} for p in (area.get("poly") or []) if "x" in p and "y" in p]
    area_units2 = polygon_area(poly)
    area_m2 = units2_to_m2(area_units2, use_metric)
    area_ref = area.get("refid") or sha({"poly": poly})
    room_id = stable_id_for("room", project=project_id, floor=floor_id, design=design_id, area=area_ref)

    # assign items to this room by point-in-polygon
    inside_items = []
    for it in items:
        x, y = it.get("x"), it.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and point_in_polygon(x, y, poly):
            inside_items.append(it)

    item_names = [item_display_name(it) for it in inside_items]
    room_type = guess_room_type(item_names)

    # summary text (RAG-friendly)
    parts = [f"{design_name or 'Design'} — {room_id[-8:]}: polygon with {len(poly)} points."]
    if area_m2 is not None:
        parts.append(f"Approx. area {area_m2:.1f} m².")
    else:
        parts.append(f"Approx. area (units²) {area_units2:.0f}.")
    if room_type != "unknown":
        parts.append(f"Likely {room_type}.")
    if item_names:
        parts.append("Contains: " + ", ".join(sorted(set(item_names))[:12]) + ".")

    room_chunk = {
        "chunk_id": room_id,
        "level": "room",
        "room_id": room_id,
        "design_id": design_id,
        "floor_id": floor_id,
        "project_id": project_id,
        "name": f"Area {area_ref}",
        "room_type": room_type,
        "num_walls": None,
        "num_doors": None,
        "num_windows": None,
        "area_m2": area_m2,
        "area_units2": area_units2 if area_m2 is None else None,
        "summary_text": " ".join(parts),
        "refs": {"parent_id": design_id, "area_refid": area_ref},
        "raw": {"area": area}
    }
    # return also the refids of items that were inside, for linking later
    return room_chunk, [it.get("refid") for it in inside_items if it.get("refid")]

def normalize_design(design: Dict[str, Any],
                     floor_id: str,
                     project_id: str,
                     use_metric: bool) -> List[Dict[str, Any]]:
    did = stable_id_for("design", floor=floor_id, name=design.get("name"))
    items = list(design.get("items") or [])
    areas = list(design.get("areas") or [])
    out: List[Dict[str, Any]] = []

    # design chunk
    out.append({
        "chunk_id": did,
        "level": "design",
        "design_id": did,
        "floor_id": floor_id,
        "project_id": project_id,
        "name": design.get("name") or "Design",
        "item_counts": [],  # optional aggregate later
        "summary_text": f"Design '{design.get('name') or 'Design'}' with {len(areas)} areas and {len(items)} items.",
        "refs": {"parent_id": floor_id},
        "raw": {"name": design.get("name"), "counts": {"areas": len(areas), "items": len(items)}}
    })

    # room chunks (areas) + collect which items belong to which room
    room_chunks: List[Dict[str, Any]] = []
    item_to_room: Dict[str, str] = {}  # refid -> room_id

    for a in areas:
        room_chunk, item_refs = normalize_area_to_room(a, items, use_metric, did, floor_id, project_id, design.get("name"))
        room_chunks.append(room_chunk)
        for ref in item_refs:
            item_to_room[ref] = room_chunk["room_id"]

    out += room_chunks

    # item chunks (with room_id if assigned)
    for it in items:
        room_id = item_to_room.get(it.get("refid"))
        out.append(normalize_item(it, did, room_id))

    return out

def normalize_floor(floor: Dict[str, Any], project_id: str, use_metric: bool) -> List[Dict[str, Any]]:
    fid = stable_id_for("floor", project=project_id, name=floor.get("name"), level=floor.get("level"))
    out = [{
        "chunk_id": fid,
        "level": "floor",
        "floor_id": fid,
        "project_id": project_id,
        "name": floor.get("name"),
        "level_num": floor.get("level"),
        "height": floor.get("height"),
        "summary_text": f"Floor '{floor.get('name')}', level {floor.get('level')}, {len(floor.get('designs',[]))} designs.",
        "refs": {"parent_id": project_id},
        "raw": {"id": floor.get("id"), "name": floor.get("name"), "level": floor.get("level")}
    }]
    for d in floor.get("designs", []) or []:
        out += normalize_design(d, fid, project_id, use_metric)
    return out

def normalize_project(fml: Dict[str, Any]) -> List[Dict[str, Any]]:
    pid = stable_id_for("project", id=fml.get("id"), name=fml.get("name"))
    settings = fml.get("settings") or {}
    use_metric = bool(settings.get("useMetric", True))
    out = [{
        "chunk_id": pid,
        "level": "project",
        "project_id": pid,
        "name": fml.get("name"),
        "settings": settings,
        "summary_text": f"Project '{fml.get('name')}' with {len(fml.get('floors',[]))} floors.",
        "refs": {"parent_id": None},
        "raw": {"id": fml.get("id"), "features": fml.get("features")}
    }]
    for fl in fml.get("floors", []) or []:
        out += normalize_floor(fl, pid, use_metric)
    return out

# ---------- entry ----------

def process_fml_file(fml_path: Path, out_dir: Path) -> int:
    try:
        fml = json.loads(fml_path.read_text())
    except Exception as e:
        print(f"[ERROR] Failed to read {fml_path.name}: {e}")
        return 0
    chunks = normalize_project(fml)
    for ch in chunks:
        out_file = out_dir / f"{ch['chunk_id']}.json"
        write_json(out_file, ch)
    print(f"[OK] {fml_path.name} -> {len(chunks)} chunks")
    return len(chunks)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_fml", help="Single .fml JSON file")
    ap.add_argument("--in_dir", help="Directory with .fml files")
    ap.add_argument("--out_dir", default="sample_chunks", help="Output directory for chunks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    if args.in_fml:
        total += process_fml_file(Path(args.in_fml), out_dir)
    elif args.in_dir:
        for p in Path(args.in_dir).glob("*.fml"):
            total += process_fml_file(p, out_dir)
    else:
        print("Provide --in_fml or --in_dir")
        return
    print(f"Wrote {total} chunks to {out_dir}")

if __name__ == "__main__":
    main()
