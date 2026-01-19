from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import lmdb
import pickle

META_KEYS = {
    "len": [b"__len__", b"__length__"],
    "subject_ids": [b"__subject_ids__", b"__subjects__"],
    "subject_ranges": [b"__subject_ranges__", b"__subj_ranges__", b"__subject_ranges__".replace(b"subject", b"subj")],
}

def _get_first(txn: lmdb.Transaction, keys: List[bytes]) -> Optional[bytes]:
    for k in keys:
        v = txn.get(k)
        if v is not None:
            return v
    return None

@dataclass
class LMDBIndex:
    total_len: int
    subject_ids: List[str]
    sub2range: Dict[str, List[Tuple[int, int]]]

def read_lmdb_index(lmdb_path: str | Path) -> LMDBIndex:
    lmdb_path = str(lmdb_path)
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=False) as txn:
            blob_len = _get_first(txn, META_KEYS["len"])
            if blob_len is None:
                raise KeyError("LMDB missing __len__ metadata")
            total_len = int(pickle.loads(blob_len))

            blob_subj = _get_first(txn, META_KEYS["subject_ids"])
            subject_ids = pickle.loads(blob_subj) if blob_subj is not None else []

            blob_ranges = _get_first(txn, META_KEYS["subject_ranges"])
            if blob_ranges is None:
                # allow LMDB without ranges
                sub2range = {}
            else:
                sub2range = pickle.loads(blob_ranges)
    finally:
        env.close()

    # ensure keys are str
    sub2range2 = {}
    for k, v in (sub2range or {}).items():
        sub2range2[str(k)] = [(int(a), int(b)) for a, b in v]

    return LMDBIndex(total_len=total_len, subject_ids=[str(s) for s in subject_ids], sub2range=sub2range2)

def read_lmdb_record(lmdb_path: str | Path, global_idx: int) -> Dict[str, Any]:
    lmdb_path = str(lmdb_path)
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=False) as txn:
            key = int(global_idx).to_bytes(8, "big")
            blob = txn.get(key)
            if blob is None:
                raise KeyError(f"Index {global_idx} not found in LMDB")
            rec = pickle.loads(blob)
    finally:
        env.close()
    return rec

def subject_to_indices(index: LMDBIndex, subject_id: str) -> List[int]:
    subject_id = str(subject_id)
    out: List[int] = []
    for start, end in index.sub2range.get(subject_id, []):
        if end > start:
            out.extend(range(start, end))
    # filter bounds
    return [i for i in out if 0 <= i < index.total_len]
