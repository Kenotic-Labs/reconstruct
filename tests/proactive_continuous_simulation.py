import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.proactive import proactive_engine


REPORT_PATH = r"S:\Nura\Docs\PROACTIVE_CONTINUOUS_SIMULATION_REPORT.md"
RUN_LOG_PATH = r"S:\Nura\Docs\PROACTIVE_CONTINUOUS_RUNS.jsonl"
RUN_SUMMARY_PATH = r"S:\Nura\Docs\PROACTIVE_CONTINUOUS_RUNS_SUMMARY.json"
TRACE_SEED = 1005
TRACE_BEFORE_PATH = r"S:\Nura\Docs\PROACTIVE_CONTINUOUS_TRACE_seed1005_before.json"
TRACE_AFTER_PATH = r"S:\Nura\Docs\PROACTIVE_CONTINUOUS_TRACE_seed1005_after.json"
_SIGNIFICANCE_PRIORITY = {
    "TASK_CLOSURE": 0,
    "NARRATIVE_EXPECTATION": 1,
    "TASK_BLOCKING": 2,
    "EMOTIONAL_CHECK_IN": 3,
    "HABIT": 4,
}


@dataclass
class Scenario:
    seed: int
    fingerprint: str
    base_time: datetime
    memories: List[Dict[str, Any]]


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _fingerprint(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("ascii", errors="ignore")).hexdigest()


def _make_task(rng: random.Random, base: datetime, idx: int) -> Dict[str, Any]:
    due_offset_hours = rng.randint(-8, 36)
    due_at = base + timedelta(hours=due_offset_hours)
    created_at = base - timedelta(hours=rng.randint(1, 48))
    task_id = f"task_{idx}"
    risk_window_hours = rng.choice([6, 12, 24])
    return {
        "memory_id": task_id,
        "memory_type": "task",
        "created_at": _iso(created_at),
        "due_at": _iso(due_at),
        "resolved": False,
        "is_resolved": False,
        "status": "open",
        "risk_window_hours": risk_window_hours,
        "outcome_known": rng.choice([True, False]),
        "prerequisites": [],
        "focus": f"task focus {idx}",
    }


def _make_prereq(rng: random.Random, base: datetime, idx: int) -> Dict[str, Any]:
    created_at = base - timedelta(hours=rng.randint(1, 72))
    prereq_id = f"prereq_{idx}"
    return {
        "memory_id": prereq_id,
        "memory_type": "task",
        "created_at": _iso(created_at),
        "resolved": False,
        "is_resolved": False,
        "status": "open",
        "outcome_known": rng.choice([True, False]),
        "focus": f"prereq focus {idx}",
    }


def _make_emotional(rng: random.Random, base: datetime, idx: int) -> Dict[str, Any]:
    created_at = base - timedelta(hours=rng.randint(1, 48))
    return {
        "memory_id": f"emo_{idx}",
        "memory_type": "personal",
        "created_at": _iso(created_at),
        "resolved": False,
        "is_resolved": False,
        "status": "open",
        "category": "emotional_state",
        "focus": f"emotion focus {idx}",
    }


def _make_narrative(rng: random.Random, base: datetime, idx: int) -> Dict[str, Any]:
    created_at = base - timedelta(hours=rng.randint(1, 72))
    boundary = base + timedelta(hours=rng.randint(-6, 24))
    return {
        "memory_id": f"narr_{idx}",
        "memory_type": "event",
        "created_at": _iso(created_at),
        "narrative_boundary_at": _iso(boundary),
        "acknowledged": rng.choice([True, False]),
        "focus": f"narrative focus {idx}",
    }


def _generate_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, 9, 0, 0) + timedelta(days=rng.randint(0, 300), hours=rng.randint(0, 10))

    task_count = rng.randint(0, 4)
    emo_count = rng.randint(0, 3)
    narrative_count = rng.randint(0, 2)

    tasks = [_make_task(rng, base, i + 1) for i in range(task_count)]
    prereqs: List[Dict[str, Any]] = []
    for task in tasks:
        if rng.random() < 0.5:
            prereq = _make_prereq(rng, base, len(prereqs) + 1)
            prereqs.append(prereq)
            task["prerequisites"].append({"id": prereq["memory_id"], "status": rng.choice(["unknown", "unknown", "done"])})

    memories: List[Dict[str, Any]] = []
    memories.extend(tasks)
    memories.extend(prereqs)
    for i in range(emo_count):
        memories.append(_make_emotional(rng, base, i + 1))
    for i in range(narrative_count):
        memories.append(_make_narrative(rng, base, i + 1))

    fingerprint = _fingerprint({
        "tasks": tasks,
        "prereqs": prereqs,
        "emo_count": emo_count,
        "narrative_count": narrative_count,
    })
    return Scenario(seed=seed, fingerprint=fingerprint, base_time=base, memories=memories)


def _mandatory_candidates(memories: List[Dict[str, Any]], now: datetime, disable_task_closure: bool) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    memory_index = {str(m.get("memory_id")): m for m in memories}

    for mem in memories:
        if str(mem.get("memory_type", "")).lower() != "task":
            continue
        due_at = _parse_iso(mem.get("due_at"))
        if not disable_task_closure:
            if due_at and now > due_at and not _bool(mem.get("outcome_known")):
                candidates.append({
                    "memory_id": str(mem.get("memory_id")),
                    "obligation_class": "TASK_CLOSURE",
                    "violation_at": due_at,
                    "due_at": due_at,
                    "created_at": _parse_iso(mem.get("created_at")),
                })

        due_at = _parse_iso(mem.get("due_at"))
        risk_window = mem.get("risk_window_hours")
        if due_at and isinstance(risk_window, (int, float)):
            risk_start = due_at - timedelta(hours=float(risk_window))
            if now >= risk_start and now < due_at:
                if _bool(mem.get("asked_before")):
                    continue
                for prereq in mem.get("prerequisites") or []:
                    if not isinstance(prereq, dict):
                        continue
                    status = str(prereq.get("status", "")).lower()
                    if status in {"done", "complete", "completed"}:
                        continue
                    prereq_id = str(prereq.get("id") or prereq.get("memory_id") or "")
                    prereq_mem = memory_index.get(prereq_id)
                    if not prereq_mem or _bool(prereq_mem.get("outcome_known")):
                        continue
                    if _bool(prereq_mem.get("asked_before")):
                        continue
                    candidates.append({
                        "memory_id": prereq_id,
                        "obligation_class": "TASK_BLOCKING",
                        "violation_at": risk_start,
                        "due_at": due_at,
                        "created_at": _parse_iso(prereq_mem.get("created_at")),
                    })

    for mem in memories:
        if _bool(mem.get("asked_before")):
            continue
        boundary = _parse_iso(mem.get("narrative_boundary_at"))
        if boundary and now > boundary and not _bool(mem.get("acknowledged")):
            candidates.append({
                "memory_id": str(mem.get("memory_id")),
                "obligation_class": "NARRATIVE_EXPECTATION",
                "violation_at": boundary,
                "due_at": _parse_iso(mem.get("due_at")),
                "created_at": _parse_iso(mem.get("created_at")),
            })

    return candidates


def _select_mandatory(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    def sort_key(item: Dict[str, Any]) -> Tuple[int, datetime, datetime, datetime]:
        # Mirror proactive_engine significance ordering for deterministic validation.
        significance = _SIGNIFICANCE_PRIORITY.get(item.get("obligation_class"), 99)
        violation_at = item.get("violation_at") or datetime.max
        due_at = item.get("due_at") or datetime.max
        created_at = item.get("created_at") or datetime.max
        return (significance, violation_at, due_at, created_at)
    candidates.sort(key=sort_key)
    return candidates[0]


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    return value is True


def _active_window(memories: List[Dict[str, Any]], now: datetime) -> bool:
    for mem in memories:
        start = _parse_iso(mem.get("active_window_start"))
        end = _parse_iso(mem.get("active_window_end"))
        if start and end and start <= now <= end:
            return True
    return False


def _simulate_scenario(scenario: Scenario, disable_task_closure: bool) -> Tuple[bool, Dict[str, Any]]:
    memories = scenario.memories
    base = scenario.base_time
    times: List[datetime] = []

    earliest_due = None
    for mem in memories:
        due_at = _parse_iso(mem.get("due_at"))
        if due_at and (earliest_due is None or due_at < earliest_due):
            earliest_due = due_at

    narrative_boundary = None
    for mem in memories:
        boundary = _parse_iso(mem.get("narrative_boundary_at"))
        if boundary and (narrative_boundary is None or boundary < narrative_boundary):
            narrative_boundary = boundary

    times.append(base - timedelta(hours=6))
    if earliest_due:
        times.append(earliest_due - timedelta(hours=12))
        times.append(earliest_due + timedelta(hours=2))
    if narrative_boundary:
        times.append(narrative_boundary + timedelta(hours=2))

    times = sorted({t for t in times})

    asks_today = 0
    last_asked_at: Optional[str] = None
    asked_obligations: set[Tuple[str, str]] = set()
    mandatory_collision_count = 0
    timeline: List[Dict[str, Any]] = []

    for now in times:
        payload = {
            "user_id": "user-1",
            "now_timestamp": _iso(now),
            "recent_memories": memories,
            "temporal_tags": [],
            "cooldown_state": {
                "last_asked_at": last_asked_at,
                "asks_today": asks_today,
            },
        }
        decision = proactive_engine.evaluate(payload)
        timeline.append({
            "now": _iso(now),
            "decision": decision,
        })

        active_window = _active_window(memories, now)
        mandatory_candidates = _mandatory_candidates(memories, now, disable_task_closure)
        mandatory_selected = _select_mandatory(mandatory_candidates)
        if len(mandatory_candidates) > 1:
            mandatory_collision_count += 1

        if active_window and decision.get("should_ask"):
            return False, {
                "invariant": "ACTIVE_WINDOW_GUARD",
                "timeline": timeline,
            }

        if mandatory_selected and not active_window:
            if decision.get("should_ask") is not True:
                return False, {
                    "invariant": "TASK_CLOSURE_ENFORCEMENT",
                    "timeline": timeline,
                }
            if decision.get("memory_id") != mandatory_selected.get("memory_id"):
                return False, {
                    "invariant": "TASK_BLOCKING_PRECEDENCE",
                    "timeline": timeline,
                }

        if not mandatory_selected and decision.get("should_ask"):
            if decision.get("memory_id") is None:
                return False, {
                    "invariant": "SILENCE_LEGALITY",
                    "timeline": timeline,
                }

        if decision.get("should_ask"):
            memory_id = str(decision.get("memory_id"))
            obligation = mandatory_selected.get("obligation_class") if mandatory_selected else "OPTIONAL"
            key = (obligation, memory_id)
            if key in asked_obligations:
                return False, {
                    "invariant": "SINGLE_QUESTION_RULE",
                    "timeline": timeline,
                }
            asked_obligations.add(key)
            asks_today += 1
            last_asked_at = _iso(now)
            for mem in memories:
                if str(mem.get("memory_id")) == memory_id:
                    mem["asked_before"] = True
                    if obligation == "TASK_CLOSURE":
                        mem["outcome_known"] = True
                        mem["resolved"] = True
                        mem["is_resolved"] = True
                        mem["status"] = "closed"
                    if obligation == "NARRATIVE_EXPECTATION":
                        mem["acknowledged"] = True
                    if obligation == "TASK_BLOCKING":
                        mem["outcome_known"] = True
                        mem["resolved"] = True
                        mem["is_resolved"] = True
                        mem["status"] = "closed"

    return True, {
        "timeline": timeline,
        "mandatory_collision_count": mandatory_collision_count,
    }


def run_simulation(total: int = 1000, disable_task_closure: bool = False, trace_seed: Optional[int] = None) -> Dict[str, Any]:
    scenarios: List[Scenario] = []
    fingerprints = set()
    seed = 1000
    attempts = 0
    while len(scenarios) < total and attempts < total * 10:
        attempts += 1
        scenario = _generate_scenario(seed + attempts)
        if scenario.fingerprint in fingerprints:
            continue
        fingerprints.add(scenario.fingerprint)
        scenarios.append(scenario)

    if len(scenarios) < total:
        return {
            "status": "failure",
            "reason": "INSUFFICIENT_UNIQUE_SCENARIOS",
            "count": len(scenarios),
        }

    obligations_evaluated = 0
    collision_count = 0
    seed_check = 1005
    seed_check_passed = None
    runs: List[Dict[str, Any]] = []
    invariants_checked = [
        "ACTIVE_WINDOW_GUARD",
        "TASK_CLOSURE_ENFORCEMENT",
        "TASK_BLOCKING_PRECEDENCE",
        "SILENCE_LEGALITY",
        "SINGLE_QUESTION_RULE",
    ]
    trace_detail = None
    for scenario in scenarios:
        ok, detail = _simulate_scenario(scenario, disable_task_closure)
        obligations_evaluated += 1
        collision_count += detail.get("mandatory_collision_count", 0)
        if trace_seed is not None and scenario.seed == trace_seed:
            trace_detail = {
                "seed": scenario.seed,
                "fingerprint": scenario.fingerprint,
                "scenario": {
                    "base_time": _iso(scenario.base_time),
                    "memories": scenario.memories,
                },
                "timeline": detail.get("timeline", []),
            }
        violated = None
        if not ok:
            violated = detail.get("invariant")
        runs.append({
            "index": len(runs) + 1,
            "seed": scenario.seed,
            "fingerprint": scenario.fingerprint,
            "violation": bool(violated),
            "violated_invariant": violated,
            "invariants_checked": invariants_checked,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    if not disable_task_closure:
        scenario = _generate_scenario(seed_check)
        ok, detail = _simulate_scenario(scenario, disable_task_closure)
        seed_check_passed = ok

    return {
        "status": "success" if not any(r["violation"] for r in runs) else "failure",
        "total_scenarios": len(scenarios),
        "obligations_evaluated": obligations_evaluated,
        "mandatory_collision_count": collision_count,
        "seed_check": seed_check,
        "seed_check_passed": seed_check_passed,
        "runs": runs,
        "trace_detail": trace_detail,
    }


def write_report(result: Dict[str, Any]) -> None:
    lines = ["# Proactive Continuous Simulation Report", ""]
    if result.get("status") == "success":
        lines.append(f"Total scenarios run: {result['total_scenarios']}")
        lines.append(f"Total obligations evaluated: {result['obligations_evaluated']}")
        lines.append(f"Mandatory collisions: {result.get('mandatory_collision_count', 0)}")
        if result.get("seed_check") is not None:
            lines.append(f"Previous failure seed {result['seed_check']} passed: {result.get('seed_check_passed')}")
        lines.append("Violations: 0")
    else:
        lines.append("Status: FAILURE")
        lines.append("Violations detected. See run logs for details.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines))


def main() -> None:
    sanity_result = run_simulation(1000, disable_task_closure=True, trace_seed=TRACE_SEED)
    if sanity_result.get("status") == "success":
        write_report({
            "status": "failure",
            "seed": None,
            "fingerprint": None,
            "detail": {"invariant": "SANITY_FAIL_NOT_TRIGGERED"},
            "scenario": {},
        })
        raise SystemExit(1)

    result = run_simulation(1000, disable_task_closure=False, trace_seed=TRACE_SEED)
    write_report(result)
    os.makedirs(os.path.dirname(RUN_LOG_PATH), exist_ok=True)
    with open(RUN_LOG_PATH, "w", encoding="ascii", errors="ignore") as f:
        for run in result.get("runs", []):
            f.write(json.dumps(run, ensure_ascii=True) + "\n")

    summary = {
        "total_runs": len(result.get("runs", [])),
        "unique_fingerprints": len({r["fingerprint"] for r in result.get("runs", [])}),
        "violations": [r for r in result.get("runs", []) if r.get("violation")],
        "seed_check": result.get("seed_check"),
        "seed_check_passed": result.get("seed_check_passed"),
    }
    with open(RUN_SUMMARY_PATH, "w", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(summary, ensure_ascii=True, indent=2))

    if sanity_result.get("trace_detail"):
        with open(TRACE_BEFORE_PATH, "w", encoding="ascii", errors="ignore") as f:
            f.write(json.dumps(sanity_result["trace_detail"], ensure_ascii=True, indent=2))
    if result.get("trace_detail"):
        with open(TRACE_AFTER_PATH, "w", encoding="ascii", errors="ignore") as f:
            f.write(json.dumps(result["trace_detail"], ensure_ascii=True, indent=2))

    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
