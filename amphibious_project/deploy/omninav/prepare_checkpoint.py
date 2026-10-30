"""Fetch a pinned public OmniNav checkpoint; no model loading or GPU access.

Only inference files are downloaded. Resumable parts and SHA-256 checks keep a
failed transfer from being mistaken for a complete trained navigation model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[2]
MODEL = "chongchongjj/OmniNav_Slowfast"
ENDPOINT = "https://modelscope.ai/api/v1/models/" + MODEL
LOCK = PROJECT / "configs/omninav_checkpoint.lock.json"
FILES = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
         "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
         "added_tokens.json", "chat_template.json", "generation_config.json",
         "vocab.json", "merges.txt", "README.md",
         "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
HEADS = ("wp_predictor.", "wp_predictor_angle.", "input_wp_encoder.",
         "query_action", "query_multihead_attn.", "arrive_predictor.")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def make_lock():
    with urlopen(ENDPOINT + "/repo/files?Revision=master&Recursive=true", timeout=30) as response:
        data = json.load(response)
    if not data.get("Success"): raise RuntimeError("ModelScope did not return a successful file listing")
    selected = [{"path": f["Path"], "bytes": int(f["Size"]), "sha256": f["Sha256"]}
                for f in data["Data"]["Files"] if f["Type"] == "blob" and f["Path"] in FILES]
    if {f["path"] for f in selected} != FILES: raise RuntimeError("Checkpoint inference files are missing")
    revision = data["Data"]["LatestCommitter"]["ShortId"]
    full = {f["Revision"] for f in data["Data"]["Files"] if f.get("Revision", "").startswith(revision)}
    if len(full)==1: revision=next(iter(full))
    return {"schema": "L17.omninav_checkpoint.v1", "model": MODEL, "revision": revision,
            "source": "https://www.modelscope.ai/models/" + MODEL,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "files": sorted(selected, key=lambda f: (f["path"].endswith(".safetensors"), f["path"]))}


def download_one(entry, revision, destination):
    target = destination / entry["path"]
    if target.parent != destination or target.name != entry["path"]: raise ValueError("Unexpected checkpoint path")
    if target.is_file() and target.stat().st_size == entry["bytes"] and sha(target) == entry["sha256"]:
        return
    part = target.with_suffix(target.suffix + ".part")
    if part.is_file() and part.stat().st_size==entry["bytes"]:
        if sha(part)!=entry["sha256"]:raise RuntimeError("A complete partial file has an invalid hash: "+part.name)
        part.replace(target);return
    url = ENDPOINT + "/repo?" + urlencode({"Revision": revision, "FilePath": entry["path"]})
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry["bytes"]: raise RuntimeError("Partial file exceeds the pinned size")
            request = Request(url, headers={"User-Agent": "L17-OmniNav-checkpoint/1"})
            if offset: request.add_header("Range", "bytes=" + str(offset) + "-")
            with urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith("bytes " + str(offset) + "-"):
                        raise RuntimeError("Server returned a different resume range")
                elif response.status == 200:
                    offset = 0
                else: raise RuntimeError("Unexpected download status " + str(response.status))
                print(json.dumps({"downloading": target.name, "resume_bytes": offset,
                                  "total_bytes": entry["bytes"]}), flush=True)
                with part.open("ab" if offset else "wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if part.stat().st_size != entry["bytes"] or sha(part) != entry["sha256"]:
                raise RuntimeError("Pinned size/SHA-256 check failed: " + target.name)
            part.replace(target)
            print(json.dumps({"verified": target.name, "bytes": entry["bytes"]}), flush=True)
            return
        except (OSError, RuntimeError) as error:
            if attempt == 2: raise
            print(json.dumps({"retry": attempt + 1, "file": target.name, "error": str(error)}), flush=True)
            time.sleep(2)


def inspect_checkpoint(destination, require_weights=True):
    config = json.loads((destination / "config.json").read_text())
    if config.get("waypoint_number") != 5 or not config.get("predict_angle") or config.get("cross_attention_flow_match"):
        raise ValueError("This is not the required five-waypoint Slowfast/angle checkpoint")
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    keys = index["weight_map"]
    for prefix in HEADS:
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError("Trained head is absent from the checkpoint index: " + prefix)
    headers = {}
    if require_weights:
        for name in sorted(set(keys.values())):
            path = destination / name
            if not path.is_relative_to(destination) or not path.is_file(): raise ValueError("Missing weight shard")
            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                if not 2 <= length <= 16_000_000: raise ValueError("Invalid safetensors header size")
                header = json.loads(handle.read(length))
            for key, value in header.items():
                if key != "__metadata__" and key.startswith(HEADS): headers[key] = value["shape"]
        for key in keys:
            if key.startswith(HEADS) and key not in headers: raise ValueError("Indexed navigation-head tensor is missing: " + key)
    return {"checkpoint": str(destination), "waypoint_number": 5, "predict_angle": True,
            "use_arrive_list": bool(config.get("use_arrive_list")), "trained_head_shapes": headers,
            "weights_present": require_weights, "model_loaded": False, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-lock", action="store_true", help="Pin the current public repository once")
    parser.add_argument("--download", action="store_true", help="Include both large weight shards")
    parser.add_argument("--metadata-only", action="store_true", help="Download inference metadata and tokenizer files only")
    parser.add_argument("--verify", action="store_true", help="Offline size/SHA-256 and trained-head checks")
    parser.add_argument("--destination", type=Path, default=PROJECT / "checkpoints/OmniNav_Slowfast")
    args = parser.parse_args()
    if args.create_lock:
        if LOCK.exists(): raise ValueError("Preserve the existing pinned manifest instead of silently updating it")
        LOCK.write_text(json.dumps(make_lock(), indent=2) + "\n", encoding="utf-8")
    lock = json.loads(LOCK.read_text())
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    entries = [e for e in lock["files"] if args.download or not e["path"].endswith(".safetensors")]
    if args.download or args.metadata_only:
        def stored(entry):
            path=destination/entry["path"]
            part=path.with_suffix(path.suffix+".part")
            if path.is_file():return min(entry["bytes"],path.stat().st_size)
            return min(entry["bytes"],part.stat().st_size) if part.is_file() else 0
        remaining = sum(max(0,e["bytes"]-stored(e)) for e in entries)
        if shutil.disk_usage(destination).free < remaining + 2 * 1024**3: raise RuntimeError("Leave at least 2 GiB free after downloading")
        for entry in entries: download_one(entry, lock["revision"], destination)
    if args.verify or args.download:
        for entry in lock["files"]:
            path = destination / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise RuntimeError("Incomplete or modified checkpoint file: " + entry["path"])
    report = inspect_checkpoint(destination, require_weights=args.verify or args.download)
    (destination / "l17_checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
