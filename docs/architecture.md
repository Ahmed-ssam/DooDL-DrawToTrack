# Architecture

**Layers.** `static/` (canvas UI, no build step) calls `api.py` (FastAPI, thin). It delegates to `service.py` (`DooDLService`: sessions, objects, jobs), which talks to a `SegmentationBackend` (`backends/base.py`). `pipeline.py` runs a tracking job and renders outputs.

**Backend contract.** `open_session`, `set_prompt` (replace semantics; the full accumulated outline and clicks are resent), `propagate(start, reverse, obj_ids)`, `remove_object`, `close_session`, plus optional stats. SAM 2.1 and the classical baseline implement it; SAM 3.1 could too.

**Tracking passes.** One forward pass from the earliest prompted frame covers all objects. Each distinct later start frame adds one reverse pass for the objects prompted there.

**Storage.** Frames are JPEGs; masks are indexed PNG label maps (pixel value is object id, higher id wins on overlap). Exports are derived from them.

**Concurrency.** One worker runs jobs; a lock guards the backend. Only a few sessions keep model state live (`max_active_handles`); evicted ones are rebuilt by replaying prompts. A sweeper deletes expired sessions.

**Security.** Raw-body uploads with a size cap enforced mid-stream, sanitized filenames, optional token, no framing restrictions so the Colab iframe works.

## Design decisions

- SAM 2.1 over SAM 3.1: ungated weights and a stable video API; text prompts are not needed.
- Polling over SSE: simpler behind Colab proxies and tunnels.
- No front-end build: easy to read, nothing to install.
- Model deps optional: CI and the CPU baseline need no PyTorch.
