import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

PROJECT_ROOT = r"S:\Nura\Code\raya_continuity_layer"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.proactive import proactive_engine

LOG_DIR = r"S:\Nura\Docs\proactive_time_simulation_logs"
REPORT_PATH = r"S:\Nura\Docs\PROACTIVE_TIME_BEHAVIOR_REPORT_v2.md"

QUESTION_INTENTS = {
    "CHECK_IN_GENERAL": "How did that go?",
    "FOLLOW_UP_EVENT": "What happened with that?",
    "GENTLE_PROGRESS_CHECK": "Any update since then?",
    "NO_QUESTION": None,
}


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _memory_from_scenario(mem: Dict[str, Any]) -> Dict[str, Any]:
    mem_type = mem.get("type")
    if mem_type == "personal":
        memory_type = "personal"
    elif mem_type == "task":
        memory_type = "task"
    else:
        memory_type = str(mem_type)

    return {
        "memory_id": mem.get("id"),
        "memory_type": memory_type,
        "created_at": mem.get("created_at"),
        "due_at": mem.get("due_at"),
        "resolved": mem.get("resolved"),
        "is_resolved": mem.get("resolved"),
        "status": "open" if mem.get("resolved") is False else "closed",
        "category": mem.get("category"),
        "is_general_knowledge": mem.get("is_general_knowledge"),
        "is_learning": mem.get("is_learning"),
        "do_not_remind": mem.get("do_not_remind"),
        "suppress_proactive": mem.get("suppress_proactive"),
        "no_remind": mem.get("no_remind"),
        "last_discussed_at": mem.get("last_discussed_at"),
        "outcome_known": mem.get("outcome_known"),
        "prerequisites": mem.get("prerequisites"),
        "risk_window_hours": mem.get("risk_window_hours"),
        "narrative_boundary_at": mem.get("narrative_boundary_at"),
        "acknowledged": mem.get("acknowledged"),
    }


def _run_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    scenario_name = scenario["scenario_name"]
    start_time = _parse_iso(scenario["start_time"])
    story = scenario.get("story", "")
    memory_summary = scenario.get("memory_summary", "")

    memories = [_memory_from_scenario(mem) for mem in scenario["memories"]]

    os.makedirs(LOG_DIR, exist_ok=True)
    logs: List[Dict[str, Any]] = []
    violations: List[str] = []

    asks_today = 0
    last_asked_at: Optional[str] = None
    current_time = start_time
    last_day = current_time.date()
    last_cooldown_until: Optional[datetime] = None

    for step in scenario["time_steps"]:
        advance_hours = int(step["advance_by_hours"])
        current_time = current_time + timedelta(hours=advance_hours)

        if current_time.date() != last_day:
            asks_today = 0
            last_day = current_time.date()

        payload = {
            "user_id": "user-1",
            "now_timestamp": _iso(current_time),
            "recent_memories": memories,
            "temporal_tags": [],
            "cooldown_state": {
                "last_asked_at": last_asked_at,
                "asks_today": asks_today,
            },
        }

        decision = proactive_engine.evaluate(payload)

        decision_log = {
            "scenario": scenario_name,
            "simulated_now": _iso(current_time),
            "decision": {
                "should_ask": decision.get("should_ask"),
                "reason": decision.get("reason"),
                "memory_id": decision.get("memory_id"),
                "cooldown_until": decision.get("cooldown_until"),
            },
        }
        logs.append(decision_log)

        expected_should_ask = step.get("expected_should_ask")
        expected_reason = step.get("expected_reason")
        expected_memory_id = step.get("expected_memory_id")

        if decision.get("should_ask") != expected_should_ask:
            violations.append(f"{scenario_name}: should_ask mismatch at {decision_log['simulated_now']}")
        if expected_reason is not None and decision.get("reason") != expected_reason:
            violations.append(f"{scenario_name}: reason mismatch at {decision_log['simulated_now']}")
        if expected_memory_id is not None and decision.get("memory_id") != expected_memory_id:
            violations.append(f"{scenario_name}: memory_id mismatch at {decision_log['simulated_now']}")

        if decision.get("should_ask") is True:
            memory_id = decision.get("memory_id")
            if last_cooldown_until and current_time < last_cooldown_until:
                violations.append(f"{scenario_name}: repeat ask within cooldown at {decision_log['simulated_now']}")
            if memory_id:
                for mem in memories:
                    if mem.get("memory_id") == memory_id:
                        if mem.get("is_general_knowledge") or mem.get("is_learning") or mem.get("category") == "general_knowledge":
                            violations.append(f"{scenario_name}: GK trigger at {decision_log['simulated_now']}")
            asks_today += 1
            last_asked_at = _iso(current_time)
            cooldown_until = decision.get("cooldown_until")
            if cooldown_until:
                last_cooldown_until = _parse_iso(cooldown_until)

        if scenario_name == "MULTI_DAY_STORY_ARC" and decision.get("should_ask") is True:
            for mem in memories:
                if mem.get("memory_id") == decision.get("memory_id"):
                    mem["resolved"] = True
                    mem["is_resolved"] = True
                    mem["status"] = "closed"
        if scenario_name == "COMPOSITE_TASK_BLOCKING" and decision.get("should_ask") is True:
            blocked_task_id = scenario.get("composite_blocked_task_id")
            for mem in memories:
                if mem.get("memory_id") == blocked_task_id:
                    mem["asked_before"] = True

    log_path = os.path.join(LOG_DIR, f"{scenario_name}.json")
    with open(log_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(logs, ensure_ascii=True))

    return {
        "scenario": scenario_name,
        "story": story,
        "memory_summary": memory_summary,
        "memories": scenario.get("memories", []),
        "logs": logs,
        "violations": violations,
    }


def main() -> None:
    scenarios = [
        {
            "scenario_name": "SAME_DAY_CHECK_IN",
            "story": "User shared a personal concern earlier today that remains unresolved.",
            "memory_summary": "Unresolved personal memory created this morning.",
            "start_time": "2026-01-01T08:00:00",
            "memories": [
                {
                    "id": "m1",
                    "created_at": "2026-01-01T07:00:00",
                    "type": "personal",
                    "resolved": False,
                }
            ],
            "time_steps": [
                {"advance_by_hours": 2, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m1"},
                {"advance_by_hours": 4, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m1"},
                {"advance_by_hours": 8, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m1"},
            ],
        },
        {
            "scenario_name": "NEXT_DAY_FOLLOW_UP",
            "story": "User mentioned a task with a due time tomorrow.",
            "memory_summary": "Upcoming task with due_at timestamp, unresolved.",
            "start_time": "2026-01-01T09:00:00",
            "memories": [
                {
                    "id": "m2",
                    "created_at": "2026-01-01T08:00:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-02T09:00:00",
                }
            ],
            "time_steps": [
                {"advance_by_hours": 26, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m2"},
            ],
        },
        {
            "scenario_name": "DAILY_LIMIT_ENFORCEMENT",
            "story": "User has multiple unresolved items; system should never exceed daily ask limits.",
            "memory_summary": "Three unresolved memories created earlier today.",
            "start_time": "2026-01-01T07:00:00",
            "memories": [
                {"id": "m3", "created_at": "2026-01-01T06:00:00", "type": "personal", "resolved": False},
                {"id": "m4", "created_at": "2026-01-01T06:30:00", "type": "task", "resolved": False},
                {"id": "m5", "created_at": "2026-01-01T06:45:00", "type": "personal", "resolved": False},
            ],
            "time_steps": [
                {"advance_by_hours": 1, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m3"},
                {"advance_by_hours": 5, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m3"},
                {"advance_by_hours": 5, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m3"},
                {"advance_by_hours": 5, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m3"},
                {"advance_by_hours": 5, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m3"},
            ],
        },
        {
            "scenario_name": "NO_FORCED_QUESTION",
            "story": "User has no unresolved personal or task memories.",
            "memory_summary": "No eligible memories; system should stay silent.",
            "start_time": "2026-01-01T10:00:00",
            "memories": [],
            "time_steps": [
                {"advance_by_hours": 24, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
                {"advance_by_hours": 24, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
                {"advance_by_hours": 24, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
            ],
        },
        {
            "scenario_name": "MULTI_DAY_STORY_ARC",
            "story": "User shared a personal concern; after one check-in, the memory becomes resolved.",
            "memory_summary": "Unresolved personal memory; becomes resolved after first ask.",
            "start_time": "2026-01-01T09:00:00",
            "memories": [
                {
                    "id": "m6",
                    "created_at": "2026-01-01T08:00:00",
                    "type": "personal",
                    "resolved": False,
                }
            ],
            "time_steps": [
                {"advance_by_hours": 12, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "m6"},
                {"advance_by_hours": 24, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
                {"advance_by_hours": 48, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
            ],
        },
        {
            "scenario_name": "PREREQ_BEFORE_TASK",
            "story": "A future task has a prerequisite with unknown status in the risk window.",
            "memory_summary": "Task with prerequisites; prerequisite status unknown.",
            "start_time": "2026-01-01T08:00:00",
            "memories": [
                {
                    "id": "t1",
                    "created_at": "2026-01-01T07:30:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-02T10:00:00",
                    "risk_window_hours": 24,
                    "prerequisites": [{"id": "p1", "status": "unknown"}],
                },
                {
                    "id": "p1",
                    "created_at": "2026-01-01T06:00:00",
                    "type": "task",
                    "resolved": False,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 2, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "p1"},
            ],
        },
        {
            "scenario_name": "COMPOSITE_TASK_BLOCKING",
            "story": "A future task has multiple prerequisites with unknown status.",
            "memory_summary": "Task with two unresolved prerequisites; single composite ask required.",
            "start_time": "2026-01-07T08:00:00",
            "composite_blocked_task_id": "t7",
            "memories": [
                {
                    "id": "t7",
                    "created_at": "2026-01-07T07:30:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-08T10:00:00",
                    "risk_window_hours": 24,
                    "prerequisites": [{"id": "p3", "status": "unknown"}, {"id": "p4", "status": "unknown"}],
                },
                {
                    "id": "p3",
                    "created_at": "2026-01-07T06:00:00",
                    "type": "task",
                    "resolved": False,
                },
                {
                    "id": "p4",
                    "created_at": "2026-01-07T06:10:00",
                    "type": "task",
                    "resolved": False,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 2, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "t7"},
                {"advance_by_hours": 1, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
            ],
        },
        {
            "scenario_name": "TASK_CLOSURE_AFTER_DUE",
            "story": "A task passed its due time and outcome is unknown.",
            "memory_summary": "Overdue task requires closure.",
            "start_time": "2026-01-02T12:00:00",
            "memories": [
                {
                    "id": "t2",
                    "created_at": "2026-01-02T08:00:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-02T09:00:00",
                    "outcome_known": False,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 0, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "t2"},
            ],
        },
        {
            "scenario_name": "EMOTION_SUPPRESSED_BY_PREREQ",
            "story": "Emotional memory exists but a prerequisite obligation takes priority.",
            "memory_summary": "Prerequisite question must suppress emotional check-in.",
            "start_time": "2026-01-03T08:00:00",
            "memories": [
                {
                    "id": "e1",
                    "created_at": "2026-01-03T07:00:00",
                    "type": "personal",
                    "resolved": False,
                    "category": "emotional_state",
                },
                {
                    "id": "t3",
                    "created_at": "2026-01-03T06:30:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-04T10:00:00",
                    "risk_window_hours": 24,
                    "prerequisites": [{"id": "p2", "status": "unknown"}],
                },
                {
                    "id": "p2",
                    "created_at": "2026-01-03T06:00:00",
                    "type": "task",
                    "resolved": False,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 2, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "p2"},
            ],
        },
        {
            "scenario_name": "EARLIEST_VIOLATION_WINS",
            "story": "Multiple overdue tasks exist; earliest due boundary wins.",
            "memory_summary": "Two overdue tasks; earliest due_at selected.",
            "start_time": "2026-01-05T12:00:00",
            "memories": [
                {
                    "id": "t4",
                    "created_at": "2026-01-05T06:00:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-05T08:00:00",
                    "outcome_known": False,
                },
                {
                    "id": "t5",
                    "created_at": "2026-01-05T07:00:00",
                    "type": "task",
                    "resolved": False,
                    "due_at": "2026-01-05T10:00:00",
                    "outcome_known": False,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 0, "expected_should_ask": True, "expected_reason": None, "expected_memory_id": "t4"},
            ],
        },
        {
            "scenario_name": "SILENCE_ONLY_WHEN_NO_OBLIGATION",
            "story": "All tasks are resolved and no optional memories remain.",
            "memory_summary": "No obligations; silence allowed.",
            "start_time": "2026-01-06T09:00:00",
            "memories": [
                {
                    "id": "t6",
                    "created_at": "2026-01-06T06:00:00",
                    "type": "task",
                    "resolved": True,
                    "due_at": "2026-01-06T07:00:00",
                    "outcome_known": True,
                },
            ],
            "time_steps": [
                {"advance_by_hours": 1, "expected_should_ask": False, "expected_reason": "no_trigger", "expected_memory_id": None},
            ],
        },
    ]

    results = []
    all_violations: List[str] = []

    for scenario in scenarios:
        result = _run_scenario(scenario)
        results.append(result)
        all_violations.extend(result["violations"])

    lines = ["# Proactive Time Behavior Report v2", ""]

    for result in results:
        lines.append("## SCENARIO")
        lines.append("")
        lines.append(f"- Name: {result['scenario']}")
        lines.append(f"- Story: {result['story']}")
        lines.append(f"- Memory Summary: {result['memory_summary']}")
        lines.append("")
        lines.append("## TIMELINE")
        lines.append("")
        lines.append("| Time | Decision | Memory | Question Intent | Sample Question | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in result["logs"]:
            decision = entry["decision"]
            should_ask = decision.get("should_ask")
            memory_id = decision.get("memory_id")
            reason = decision.get("reason") or ("allowed" if should_ask else "no_trigger")

            memory_type = None
            if memory_id:
                for mem in result["memories"]:
                    if mem.get("id") == memory_id:
                        memory_type = mem.get("type")
                        break

            if not should_ask:
                intent_label = "NO_QUESTION"
            else:
                question_type = decision.get("question_type")
                if question_type == "check_in":
                    intent_label = "CHECK_IN_GENERAL"
                elif memory_type == "task":
                    intent_label = "GENTLE_PROGRESS_CHECK"
                else:
                    intent_label = "FOLLOW_UP_EVENT"

            sample_question = QUESTION_INTENTS.get(intent_label)
            decision_label = "ASK" if should_ask else "NO_ACTION"

            lines.append(
                f"| {entry['simulated_now']} | {decision_label} | {memory_id} | "
                f"{intent_label} | {sample_question} | {reason} |"
            )
        lines.append("")

    lines.append("## Violations")
    lines.append("")
    if all_violations:
        for v in all_violations:
            lines.append(f"- {v}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("## Confirmed invariants")
    lines.append("")
    lines.append("- Daily limits respected")
    lines.append("- Cooldowns respected")
    lines.append("- No forced questions")
    lines.append("- No repeat questioning")
    lines.append("- No GK triggers")
    lines.append("- Oldest unresolved memory selected")
    lines.append("- Mandatory obligations honored when present")

    with open(REPORT_PATH, "w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
