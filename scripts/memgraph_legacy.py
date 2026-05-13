"""Memgraph — BFS proximity on the relationships edge list.

The relationships table IS the graph: each row's (subject, object) is an
undirected edge. No separate graph store; we build adjacency per query
from the full user edge set."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Set


MAX_HOPS = 3


def build_adjacency(edges: Iterable[Dict]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        s = (e.get("subject") or "").strip()
        o = (e.get("object") or "").strip()
        if not s or not o:
            continue
        adj[s].add(o)
        adj[o].add(s)
    return adj


def _bfs_from(start: str, goals: Set[str],
              adj: Dict[str, Set[str]]) -> int:
    if start in goals:
        return 0
    if start not in adj:
        return -1
    visited = {start}
    q = deque([(start, 0)])
    while q:
        node, d = q.popleft()
        if d >= MAX_HOPS:
            continue
        for n in adj.get(node, ()):
            if n in visited:
                continue
            if n in goals:
                return d + 1
            visited.add(n)
            q.append((n, d + 1))
    return -1


def min_hops_to_entities(
    subject: str, object_: str,
    query_entities: Set[str], adj: Dict[str, Set[str]],
) -> int:
    """Minimum BFS hop count from either endpoint of (subject, object_)
    to any name in query_entities, capped at MAX_HOPS. Returns -1 if
    unreachable within the cap."""
    if not query_entities:
        return -1
    best = -1
    for endpoint in (subject, object_):
        if not endpoint:
            continue
        h = _bfs_from(endpoint, query_entities, adj)
        if h >= 0 and (best < 0 or h < best):
            best = h
    return best
