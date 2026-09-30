"""Diagnostic-only sitecustomize: account for GPU memory after weight load.

Mounted into the serve container and put first on PYTHONPATH for ONE
diagnostic boot.  It chains the image's own /usr/lib/python3.12/sitecustomize.py,
then wraps sglang's ModelRunner.load_model so that, right after the
"Load weight end" measurement, it writes a JSON breakdown:
  torch allocated / reserved, parameter + buffer bytes by category and dtype,
  CUDA tensors hanging off modules that are not params/buffers, every live
  CUDA tensor the GC can see (deduped by storage), and caching-allocator
  segment occupancy.  Nothing in the serve path is modified.
r0b0tlab mimo26.
"""
import os
import sys

_SYS_SITECUSTOMIZE = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_SYS_SITECUSTOMIZE):
    try:
        exec(compile(open(_SYS_SITECUSTOMIZE).read(), _SYS_SITECUSTOMIZE, "exec"), {})
    except Exception:  # noqa: BLE001
        pass

import importlib.abc  # noqa: E402

_TARGET = "sglang.srt.model_executor.model_runner"
_OUTDIR = os.environ.get("MEMDIAG_OUTDIR", "/tmp")


def _category(name: str) -> str:
    if ".experts." in name or name.endswith("experts.w13_weight") or "experts" in name:
        return "moe_experts"
    if "visual" in name or "vision" in name:
        return "vision"
    if "audio" in name:
        return "audio"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "self_attn" in name or "attn" in name:
        return "attention"
    if ".mlp." in name:
        return "dense_mlp_or_gate"
    return "other"


def _report(runner):
    import gc
    import json
    import socket
    import time
    from collections import defaultdict

    import torch

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    model = runner.model
    rep = {"host": socket.gethostname(), "pid": os.getpid(),
           "is_draft_worker": bool(getattr(runner, "is_draft_worker", False)),
           "tp_rank": getattr(runner, "tp_rank", None),
           "weight_load_mem_usage_gb": getattr(runner, "weight_load_mem_usage", None),
           "torch_allocated_gb": torch.cuda.memory_allocated() / 2**30,
           "torch_reserved_gb": torch.cuda.memory_reserved() / 2**30,
           "torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
    try:
        import psutil
        rep["sys_available_gb"] = psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001
        pass
    seen_ptr = set()
    by_cat = defaultdict(int)
    by_dtype = defaultdict(int)
    big = []
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if not p.is_cuda:
            continue
        ptr = p.untyped_storage().data_ptr()
        nbytes = p.untyped_storage().nbytes()
        if ptr in seen_ptr:
            continue
        seen_ptr.add(ptr)
        by_cat[_category(n)] += nbytes
        by_dtype[str(p.dtype)] += nbytes
        big.append((nbytes, n, str(p.dtype), list(p.shape)))
    rep["model_param_buffer_gb_by_cat"] = {k: v / 2**30 for k, v in sorted(by_cat.items())}
    rep["model_param_buffer_gb_by_dtype"] = {k: v / 2**30 for k, v in sorted(by_dtype.items())}
    rep["model_param_buffer_gb_total"] = sum(by_cat.values()) / 2**30
    big.sort(reverse=True)
    rep["largest_params"] = [{"mb": b / 2**20, "name": n, "dtype": d, "shape": s}
                             for b, n, d, s in big[:12]]
    # moe per-layer detail for the first MoE layer found
    for mn, m in model.named_modules():
        if hasattr(m, "w13_weight") and hasattr(m, "w2_weight"):
            det = {}
            for k, v in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
                det[k] = {"dtype": str(v.dtype), "shape": list(v.shape),
                          "mb": v.untyped_storage().nbytes() / 2**20}
            for k, v in vars(m).items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    det["attr:" + k] = {"dtype": str(v.dtype), "shape": list(v.shape),
                                        "mb": v.untyped_storage().nbytes() / 2**20}
            rep["first_moe_layer"] = {"module": mn, "tensors": det}
            break
    # module attributes that are CUDA tensors but not params/buffers
    attr_extra = defaultdict(int)
    for mn, m in model.named_modules():
        for k, v in vars(m).items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                ptr = v.untyped_storage().data_ptr()
                if ptr not in seen_ptr:
                    seen_ptr.add(ptr)
                    attr_extra[k] += v.untyped_storage().nbytes()
    rep["module_attr_tensor_gb"] = {k: v / 2**30 for k, v in
                                    sorted(attr_extra.items(), key=lambda x: -x[1])[:15]}
    # every GC-visible CUDA tensor not yet counted
    gc_extra = 0
    gc_items = defaultdict(int)
    gc_ptrs = set()
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                ptr = o.untyped_storage().data_ptr()
                if ptr in seen_ptr or ptr in gc_ptrs:
                    continue
                gc_ptrs.add(ptr)
                nb = o.untyped_storage().nbytes()
                gc_extra += nb
                gc_items[f"{o.dtype}{tuple(o.shape)}"] += nb
        except Exception:  # noqa: BLE001
            continue
    rep["gc_uncounted_cuda_tensor_gb"] = gc_extra / 2**30
    rep["gc_uncounted_top"] = {k: v / 2**30 for k, v in
                               sorted(gc_items.items(), key=lambda x: -x[1])[:15]}
    # allocator segments
    try:
        segs = torch.cuda.memory_snapshot()
        tot = sum(s["total_size"] for s in segs)
        act = sum(s.get("allocated_size", 0) for s in segs)
        rep["segments"] = {"count": len(segs), "total_gb": tot / 2**30,
                           "allocated_gb": act / 2**30,
                           "free_in_segments_gb": (tot - act) / 2**30}
    except Exception as e:  # noqa: BLE001
        rep["segments"] = {"error": repr(e)}
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        rep["cuda_mem_get_info_free_gb"] = free_b / 2**30
        rep["cuda_mem_get_info_total_gb"] = total_b / 2**30
    except Exception:  # noqa: BLE001
        pass
    fn = os.path.join(_OUTDIR, f"memdiag.{rep['host']}.{os.getpid()}."
                               f"{'draft' if rep['is_draft_worker'] else 'target'}.json")
    with open(fn, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[MEMDIAG] wrote {fn}: alloc={rep['torch_allocated_gb']:.2f} "
          f"reserved={rep['torch_reserved_gb']:.2f} params={rep['model_param_buffer_gb_total']:.2f} "
          f"gc_extra={rep['gc_uncounted_cuda_tensor_gb']:.2f} GB", flush=True)
    return rep


def _patch(module):
    cls = getattr(module, "ModelRunner", None)
    if cls is None or getattr(cls, "_memdiag_patched", False):
        return
    orig = cls.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            _report(self)
        except Exception as e:  # noqa: BLE001
            print(f"[MEMDIAG] failed: {e!r}", flush=True)
        return r

    cls.load_model = load_model
    cls._memdiag_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(mod, _orig=orig_exec):
            _orig(mod)
            _patch(mod)

        loader.exec_module = exec_module
        return spec


if os.environ.get("MEMDIAG", "0") == "1":
    sys.meta_path.insert(0, _Finder())
