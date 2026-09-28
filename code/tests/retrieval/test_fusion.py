"""Keep the fused search option isolated from the original split protocol."""

from src.retrieval.config import QueryConfig
from src.retrieval.fusion import fused_search
from src.retrieval.tools import execute_tool_call
from src.retrieval.views import tools_for


def test_search_tool_selection_preserves_split_default():
    default_names = {tool["function"]["name"] for tool in tools_for("native")}
    fused_names = {
        tool["function"]["name"]
        for tool in tools_for("native", search_tools="fused")
    }
    assert QueryConfig().search_tools == "split"
    assert {"bm25_search", "embedding_search"} <= default_names
    assert "memory_search" not in default_names
    assert "memory_search" in fused_names
    assert "bm25_search" not in fused_names
    assert "embedding_search" not in fused_names
    assert "memory_search" not in {
        tool["function"]["name"]
        for tool in tools_for("timeline_source", search_tools="fused")
    }


def test_fused_search_merges_evidence_and_filters_embedding_paths(tmp_path):
    class Index:
        def __init__(self, rows):
            self.rows = rows

        def search(self, query, **kwargs):
            assert query == "Pudong"
            assert kwargs["date_from"] == "2024"
            return self.rows

    shared = {
        "path": "topics/travel.md", "line": 3, "event_id": "ev1",
        "content": "Moved to Pudong", "refs": ["D1:1"],
        "dates": ["2024-05-01"],
    }
    bm25 = Index([{**shared, "final_score": 2.0}])
    embedding = Index([
        {**shared, "event": "ev1", "date": "2024-05-01"},
        {"path": "sources/other.md", "line": 1, "event": "ev2",
         "content": "Pudong", "refs": ["D2:1"], "date": "2024-06-01"},
    ])
    rows = fused_search(
        bm25, embedding, "Pudong", path_prefix="topics/",
        date_from="2024", top_k=5,
    )
    assert len(rows) == 1
    assert rows[0]["path"] == "topics/travel.md"
    assert rows[0]["rule_features"] == ["bm25", "embedding"]
    assert rows[0]["refs"] == ["D1:1"]

    output, executed, accepted = execute_tool_call(
        object(), "memory_search",
        {"query": "Pudong", "path_prefix": "topics/", "date_from": "2024"},
        memory_dir=tmp_path, files=[], condition="native",
        include_recent=True, indexes={"bm25": bm25, "embedding": embedding},
    )
    assert executed and accepted is None
    assert "topics/travel.md:3" in output
    assert "sources/other.md" not in output
