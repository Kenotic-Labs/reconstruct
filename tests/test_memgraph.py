from app.engines.memgraph import build_adjacency, min_hops_to_entities


def test_direct_edge_zero_hops():
    adj = build_adjacency([{"subject": "Maya", "object": "Vantage"}])
    assert min_hops_to_entities(
        subject="Maya", object_="Vantage",
        query_entities={"Maya"}, adj=adj,
    ) == 0


def test_one_hop_via_shared_neighbor():
    adj = build_adjacency([
        {"subject": "Maya", "object": "Vantage"},
        {"subject": "Leon", "object": "Vantage"},
    ])
    assert min_hops_to_entities(
        subject="Leon", object_="Vantage",
        query_entities={"Maya"}, adj=adj,
    ) == 1


def test_disconnected_returns_neg_one():
    adj = build_adjacency([
        {"subject": "Maya", "object": "Vantage"},
        {"subject": "Bob", "object": "Acme"},
    ])
    assert min_hops_to_entities(
        subject="Bob", object_="Acme",
        query_entities={"Maya"}, adj=adj,
    ) == -1


def test_respects_max_hops():
    adj = build_adjacency([
        {"subject": "Maya", "object": "X1"},
        {"subject": "X1", "object": "X2"},
        {"subject": "X2", "object": "X3"},
        {"subject": "X3", "object": "X4"},
        {"subject": "X4", "object": "Far"},
    ])
    assert min_hops_to_entities(
        subject="Far", object_="Far",
        query_entities={"Maya"}, adj=adj,
    ) == -1
