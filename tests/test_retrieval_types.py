from app.engines.retrieval_types import Candidate


def test_candidate_defaults_are_neutral():
    c = Candidate(relationship_id=1, edge={})
    assert c.entry_cosine == 0.0
    assert c.entity_overlap == 0
    assert c.cluster_members == 0
    assert c.hops_to_entity == -1
    assert c.exit_cosine == 0.0
    assert c.source_stages == set()


def test_candidate_source_stages_accumulates():
    c = Candidate(relationship_id=1, edge={})
    c.source_stages.add("entry")
    c.source_stages.add("expand")
    assert c.source_stages == {"entry", "expand"}
