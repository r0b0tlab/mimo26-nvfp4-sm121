"""Draft-extend CUDA-graph metadata must fill the sliding-window index.

The MTP draft layers are SWA. Capture dies in Triton when
forward_extend passes window_kv_indices=None into extend_attention.
"""

from pathlib import Path

SRC = Path(__file__).with_name("triton_backend.py")


def _fn(name: str) -> str:
    text = SRC.read_text()
    start = text.index(f"def {name}(")
    nxt = text.find("\n    def ", start + 1)
    return text[start:nxt]


def test_draft_extend_metadata_uses_window_buffer():
    body = _fn("_build_cuda_graph_forward_metadata")
    branch = body.split("elif forward_mode.is_draft_extend_v2():", 1)[1]
    branch = branch.split("else:", 1)[0]
    assert "window_kv_indices=None" not in branch
    assert "self.cuda_graph_window_kv_indices if swa else None" in branch


def test_draft_extend_buffers_fill_sliding_window():
    body = _fn("_update_draft_extend_buffers")
    assert "update_sliding_window_buffer(" in body
    assert "kv_lens" in body
