import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset  # type: ignore


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_text_raw: str
    source_url: str
    source_domain: str
    date_accessed: str
    source_type: str
    license_or_usage_note: str
    domain: str
    predicate_family_guess: str
    style_band: str
    notes: str


def _today_iso() -> str:
    return date.today().isoformat()


def _stable_id(prefix: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _hf_url(dataset_name: str, config_name: str | None, split_name: str, row_id: str, field: str) -> str:
    cfg = config_name or "default"
    return f"hf://{dataset_name}/{cfg}/{split_name}#{row_id}/{field}"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _iter_text_candidates_from_row(dataset_name: str, config: str | None, split: str, row_index: int, row: dict) -> Iterator[tuple[str, str, str]]:
    """
    Yield (field, role_hint, text) triplets from a dataset row.

    This is intentionally conservative and schema-driven:
    - Prefer role-tagged message arrays when available.
    - Otherwise emit the most obvious free-text fields.
    """
    # Common chat schema: {"messages":[{"role":"user","content":"..."}, ...]}
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg_i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue
            role = _safe_str(msg.get("role")).lower()
            content = msg.get("content")
            if content is None and "text" in msg:
                content = msg.get("text")
            text = _safe_str(content).strip()
            if not text:
                continue
            field = f"messages[{msg_i}]"
            yield field, role, text
        return

    # ShareGPT-like schema: {"conversations":[{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    conv = row.get("conversations")
    if isinstance(conv, list):
        for msg_i, msg in enumerate(conv):
            if not isinstance(msg, dict):
                continue
            role = _safe_str(msg.get("from")).lower()
            text = _safe_str(msg.get("value")).strip()
            if not text:
                continue
            field = f"conversations[{msg_i}]"
            yield field, role, text
        return

    # LMSYS-Chat-1M schema often contains "conversation" as JSON (string) or already-parsed list.
    lmsys_conv = row.get("conversation")
    if isinstance(lmsys_conv, str) and lmsys_conv.strip().startswith(("[", "{")):
        try:
            lmsys_conv = json.loads(lmsys_conv)
        except Exception:
            lmsys_conv = None
    if isinstance(lmsys_conv, list):
        for msg_i, msg in enumerate(lmsys_conv):
            if not isinstance(msg, dict):
                continue
            role = _safe_str(msg.get("role")).lower()
            text = _safe_str(msg.get("content")).strip()
            if not text:
                continue
            field = f"conversation[{msg_i}]"
            yield field, role, text
        return

    # OpenAssistant schema: usually "role" + "text"
    if "text" in row and "role" in row:
        text = _safe_str(row.get("text")).strip()
        if text:
            yield "text", _safe_str(row.get("role")).lower(), text
        return

    # DailyDialog-like: dialog is a list of utterances
    for key in ("dialog", "dialogue", "utterances", "turns"):
        v = row.get(key)
        if isinstance(v, list):
            for u_i, u in enumerate(v):
                text = _safe_str(u).strip()
                if not text:
                    continue
                yield f"{key}[{u_i}]", "", text
            return

    # PersonaChat-like: persona lines and dialog lines
    for key in ("persona", "personality", "profile"):
        v = row.get(key)
        if isinstance(v, list):
            for p_i, p in enumerate(v):
                text = _safe_str(p).strip()
                if not text:
                    continue
                yield f"{key}[{p_i}]", "persona", text

    # Generic: emit a few well-known text fields if present.
    for key in ("prompt", "question", "input", "instruction", "output", "response", "utterance", "content"):
        if key in row:
            text = _safe_str(row.get(key)).strip()
            if text:
                yield key, "", text


def _role_to_source_type(role_hint: str) -> str:
    # Closed-class role tags in dataset schemas, not natural-language parsing.
    if role_hint in ("user", "prompter", "human"):
        return "public_dataset_user_turn"
    if role_hint in ("assistant", "gpt", "bot"):
        return "public_dataset_assistant_turn"
    if role_hint == "persona":
        return "public_dataset_persona_line"
    return "public_dataset_text_field"


def _role_to_style_band(role_hint: str) -> str:
    # Prefer not to guess; keep a consistent default for downstream curation.
    if role_hint == "persona":
        return "direct"
    return "casual"


def _emit_candidates(
    dataset_name: str,
    config: str | None,
    split: str,
    rows: Iterable[dict],
    limit: int | None,
    *,
    domain: str,
    predicate_family_guess: str,
    notes_prefix: str,
    dataset_license: str | None,
) -> Iterator[Candidate]:
    count = 0
    accessed = _today_iso()
    for row_index, row in enumerate(rows):
        if limit is not None and count >= limit:
            break
        if not isinstance(row, dict):
            continue
        for field, role_hint, text in _iter_text_candidates_from_row(dataset_name, config, split, row_index, row):
            if limit is not None and count >= limit:
                break
            row_id = str(row.get("id") or row.get("_id") or row.get("message_id") or row_index)
            source_url = _hf_url(dataset_name, config, split, row_id, field)
            candidate_id = _stable_id("hf", f"{source_url}|{role_hint}|{text}")
            source_type = _role_to_source_type(role_hint)
            style_band = _role_to_style_band(role_hint)
            license_note = f"HF dataset={dataset_name} license={dataset_license or 'unknown'}"
            yield Candidate(
                candidate_id=candidate_id,
                source_text_raw=text,
                source_url=source_url,
                source_domain="huggingface.co",
                date_accessed=accessed,
                source_type=source_type,
                license_or_usage_note=license_note,
                domain=domain,
                predicate_family_guess=predicate_family_guess,
                style_band=style_band,
                notes=f"{notes_prefix} role_hint={role_hint}".strip(),
            )
            count += 1
            if limit is not None and count >= limit:
                break


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
            n += 1
    return n


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Import conversational text from HF datasets into extractor_v7 raw candidate JSONL format.")
    ap.add_argument("--dataset", action="append", required=True, help="HF dataset name, e.g. OpenAssistant/oasst1 (repeatable)")
    ap.add_argument("--config", action="append", help="Optional config name; repeat to match --dataset order")
    ap.add_argument(
        "--split",
        action="append",
        help="Split name; repeat to match --dataset order (default per dataset: train)",
    )
    ap.add_argument("--limit-per-dataset", type=int, default=2000, help="Max candidates per dataset (default: 2000)")
    ap.add_argument("--out-raw", default=None, help="Output raw candidates JSONL path")
    ap.add_argument("--out-inventory", default=None, help="Output source inventory JSONL path")
    ap.add_argument("--domain", default="mixed", help="Domain label to assign (default: mixed)")
    ap.add_argument("--predicate-family-guess", default="mixed", help="Predicate family guess (default: mixed)")
    ap.add_argument("--append", action="store_true", help="Append to output files instead of overwriting")
    ap.add_argument("--streaming", action="store_true", help="Use HF streaming mode (recommended for very large datasets)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow HF datasets with custom loading code")
    args = ap.parse_args()

    configs = args.config or []
    while len(configs) < len(args.dataset):
        configs.append(None)

    splits = args.split or []
    while len(splits) < len(args.dataset):
        splits.append("train")

    ts = date.today().strftime("%Y%m%d")
    out_raw = Path(args.out_raw or f"data/extractor_v7/raw_web/raw_candidates_{ts}_hf.jsonl")
    out_inv = Path(args.out_inventory or f"data/extractor_v7/source_inventory/source_inventory_{ts}_hf.jsonl")

    # Write incrementally per dataset so a later dataset failure doesn't drop earlier work.
    raw_mode = "a" if args.append else "w"
    inv_mode = "a" if args.append else "w"
    total_raw = 0
    total_inv = 0
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_inv.parent.mkdir(parents=True, exist_ok=True)

    with out_raw.open(raw_mode, encoding="utf-8") as raw_f, out_inv.open(inv_mode, encoding="utf-8") as inv_f:
        for ds_name, cfg, split in zip(args.dataset, configs, splits):
            ds = load_dataset(ds_name, cfg, split=split, streaming=args.streaming, trust_remote_code=args.trust_remote_code)
            ds_license = getattr(getattr(ds, "info", None), "license", None)
            notes_prefix = f"hf_dataset={ds_name} cfg={cfg or 'default'} split={split}"

            written = 0
            for cand in _emit_candidates(
                ds_name,
                cfg,
                split,
                ds,
                args.limit_per_dataset,
                domain=args.domain,
                predicate_family_guess=args.predicate_family_guess,
                notes_prefix=notes_prefix,
                dataset_license=_safe_str(ds_license) if ds_license else None,
            ):
                raw_f.write(json.dumps(cand.__dict__, ensure_ascii=True) + "\n")
                written += 1
            total_raw += written

            inv_row = {
                "dataset": ds_name,
                "config": cfg or "default",
                "split": split,
                "date_accessed": _today_iso(),
                "license": _safe_str(ds_license) if ds_license else None,
                "candidates_written": written,
                "streaming": bool(args.streaming),
            }
            inv_f.write(json.dumps(inv_row, ensure_ascii=True) + "\n")
            total_inv += 1

    print(f"wrote_raw {out_raw} {total_raw}")
    print(f"wrote_inventory {out_inv} {total_inv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
