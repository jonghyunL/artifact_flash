"""
ckpt_engine_state.py — serialize / restore vLLM V1 engine state (P5)

Companion to libckpt_vllm.so. Handles two independent pieces of state:

  A) Prefix cache hash map (block_pool.cached_block_hash_to_block)
     → enables multi-turn conversation resume: after restore, new requests
       whose prefix matches the checkpointed cache skip prefill.
     → implemented, tested, default ON.

  B) In-flight request resume (scheduler.running/waiting + per-Request state)
     → enables mid-generation resume: requests that were actively decoding
       at checkpoint time continue from num_computed_tokens.
     → PARTIAL implementation, default OFF. To fully enable B, these vLLM
       model-runner-side states also need to be patched on restore:
         * gpu_model_runner.requests[req_id] = CachedRequestState(...)
         * gpu_model_runner.input_batch.add_request(...) for persistent batch
         * gpu_model_runner.encoder_cache entries if multi-modal
         * sampling / mrope metadata if applicable
         * scheduled_cached_reqs vs scheduled_new_reqs semantics on step 1
         * per-request block-table GPU tensors
       None of these have clean external injection hooks; each requires
       reverse-engineering vLLM's private bookkeeping. Deferred to future
       work. The checkpoint side of B still dumps the data so the JSON is
       forward-compatible with a future restore implementation.

Recommended usage (A-only, conversational workload):
  Checkpoint at turn boundaries (no active running requests). On restore,
  the prefix cache is warm; the next turn's prompt hashes against the
  restored cache and skips prefill for the matching prefix.

Env:
  PYTHONHASHSEED=0  — on BOTH checkpoint and restore runs so NONE_HASH in
                      vllm.v1.core.kv_cache_utils is deterministic.
  CKPT_P5_ENABLE_B=1 — opt-in to the partial B restore (runs but will
                      crash the next scheduler step; for development only).

Usage (checkpoint side, driven by libckpt_vllm.so PyRun_SimpleString):
    import ckpt_engine_state
    ckpt_engine_state.dump_state('/tmp/ckpt_engine_state.json')

Usage (restore side, in the driver):
    from ckpt_engine_state import restore_patch
    restore_patch(llm, '/tmp/ckpt_engine_state.json')
"""

import gc
import json
import os
import sys
import time


# --------------------------------------------------------------------
# Access-path helpers
# --------------------------------------------------------------------
def _find_scheduler_via_gc():
    """Walk gc.get_objects to find the vLLM V1 Scheduler instance.

    Used from the C-side dump hook, which has no direct handle to the
    llm object. The restore side uses the explicit attribute chain
    through `llm.llm_engine.engine_core.scheduler` instead (cleaner).
    """
    for obj in gc.get_objects():
        t = type(obj).__name__
        if t == 'Scheduler' and hasattr(obj, 'running') and hasattr(obj, 'kv_cache_manager'):
            return obj
    return None


def _scheduler_from_llm(llm):
    """Extract Scheduler from a vllm.LLM instance. V1 layout only.

    Two access paths depending on the EngineCoreClient variant:
      - InprocClient: llm.llm_engine.engine_core is the InprocClient,
        and its `.engine_core` attribute is the actual EngineCore.
        Scheduler is at llm.llm_engine.engine_core.engine_core.scheduler.
      - SyncMPClient / AsyncMPClient: Scheduler lives in a subprocess
        (ZMQ). Not accessible from here — fall through to gc walk.
    """
    client = llm.llm_engine.engine_core
    # InprocClient: direct access
    if hasattr(client, 'engine_core') and hasattr(client.engine_core, 'scheduler'):
        return client.engine_core.scheduler
    # Fallback: gc-walk for any other layout
    return _find_scheduler_via_gc()


def _sampling_params_to_dict(sp):
    """SamplingParams is a msgspec.Struct — skip runtime-only fields."""
    skip = {'logits_processors', 'guided_decoding', '_all_stop_token_ids', 'bad_words_token_ids'}
    out = {}
    for f in sp.__struct_fields__:
        if f in skip:
            continue
        v = getattr(sp, f, None)
        try:
            json.dumps(v)
            out[f] = v
        except TypeError:
            pass  # not JSON-serializable, skip
    return out


def _sampling_params_from_dict(d):
    from vllm.sampling_params import SamplingParams
    return SamplingParams(**d)


# --------------------------------------------------------------------
# Dump
# --------------------------------------------------------------------
def dump_state(path):
    sched = _find_scheduler_via_gc()
    if sched is None:
        print('[ckpt_engine_state] dump: Scheduler not found via gc', file=sys.stderr)
        return False

    kvm = sched.kv_cache_manager
    bp  = kvm.block_pool

    # --- A: prefix cache hash map ---
    A = []
    for bhash, id_to_block in bp.cached_block_hash_to_block.items():
        for block_id, block in id_to_block.items():
            A.append({
                'hash_value':  bhash.hash_value,
                'token_ids':   list(bhash.token_ids),
                'extra_keys':  bhash.extra_keys,
                'block_id':    block_id,
                'ref_cnt':     block.ref_cnt,
            })

    # --- B: running + waiting requests ---
    B = []
    for q_name, q in (('running', sched.running), ('waiting', sched.waiting)):
        for req in q:
            block_ids = []
            if req.request_id in kvm.req_to_blocks:
                block_ids = [blk.block_id for blk in kvm.req_to_blocks[req.request_id]]
            B.append({
                'queue':              q_name,
                'request_id':         req.request_id,
                'prompt_token_ids':   list(req.prompt_token_ids),
                'output_token_ids':   list(req._output_token_ids),
                'num_computed_tokens': req.num_computed_tokens,
                'status':             req.status.name,
                'eos_token_id':       req.eos_token_id,
                'sampling_params':    _sampling_params_to_dict(req.sampling_params),
                'block_ids':          block_ids,
                'spec_token_ids':     list(req.spec_token_ids) if hasattr(req, 'spec_token_ids') else [],
            })

    state = {
        'version': 1,
        'dumped_at': time.time(),
        'prefix_cache': A,
        'requests': B,
    }
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
    print('[ckpt_engine_state] dumped %d prefix-cache entries, %d requests → %s'
          % (len(A), len(B), path), file=sys.stderr)
    return True


# --------------------------------------------------------------------
# Restore patch (driver side)
# --------------------------------------------------------------------
def restore_patch(llm, path):
    from vllm.v1.core.kv_cache_utils import BlockHashType
    from vllm.v1.request import Request, RequestStatus

    with open(path, 'r') as f:
        state = json.load(f)

    sched = _scheduler_from_llm(llm)
    kvm   = sched.kv_cache_manager
    bp    = kvm.block_pool

    # --- A: re-populate prefix cache hash map ---
    n_a = 0
    for e in state.get('prefix_cache', []):
        bh = BlockHashType(
            hash_value = e['hash_value'],
            token_ids  = tuple(e['token_ids']),
            extra_keys = e['extra_keys'],
        )
        if e['block_id'] >= len(bp.blocks):
            print('[ckpt_engine_state] restore: block_id %d out of range (%d blocks)'
                  % (e['block_id'], len(bp.blocks)), file=sys.stderr)
            continue
        block = bp.blocks[e['block_id']]
        # Only pull out of free list if not already used by an active request
        was_free = (block.ref_cnt == 0)
        block._block_hash = bh
        if e['ref_cnt'] > 0 and was_free:
            # remove from free list (implicit via incr_ref) and set ref count
            bp.free_block_queue.remove(block)
            block.ref_cnt = e['ref_cnt']
        elif e['ref_cnt'] == 0:
            block.ref_cnt = 0
        else:
            block.ref_cnt = max(block.ref_cnt, e['ref_cnt'])
        bp.cached_block_hash_to_block[bh][e['block_id']] = block
        n_a += 1

    # --- B: reconstruct active requests ---
    # Default OFF: restoring running requests into scheduler.running is only
    # a partial patch — the GPU model runner's CachedRequestState dict and
    # input_batch remain unpopulated, so the next step() crashes with a
    # KeyError on req_id. Opt-in via CKPT_P5_ENABLE_B=1 for development.
    # See module docstring for the full list of what's missing.
    enable_b = os.environ.get('CKPT_P5_ENABLE_B', '0') == '1'
    n_b_running = 0
    n_b_waiting = 0
    b_requests = state.get('requests', []) if enable_b else []
    if state.get('requests') and not enable_b:
        print('[ckpt_engine_state] B (in-flight request resume) disabled — '
              'set CKPT_P5_ENABLE_B=1 to opt in (partial implementation)',
              file=sys.stderr)
    for e in b_requests:
        sp = _sampling_params_from_dict(e['sampling_params'])
        req = Request(
            request_id             = e['request_id'],
            prompt_token_ids       = e['prompt_token_ids'],
            multi_modal_inputs     = None,
            multi_modal_hashes     = None,
            multi_modal_placeholders = None,
            sampling_params        = sp,
            eos_token_id           = e['eos_token_id'],
            arrival_time           = time.time(),
        )
        # Rehydrate decode state
        req._output_token_ids[:] = e['output_token_ids']
        req.num_computed_tokens  = e['num_computed_tokens']
        req.status               = RequestStatus[e['status']]
        if hasattr(req, 'spec_token_ids'):
            req.spec_token_ids[:] = e.get('spec_token_ids', [])

        # Re-attach blocks
        blocks = []
        for bid in e['block_ids']:
            if bid >= len(bp.blocks):
                continue
            blk = bp.blocks[bid]
            if blk.ref_cnt == 0:
                bp.free_block_queue.remove(blk)
            blk.ref_cnt += 1
            blocks.append(blk)
        if blocks:
            kvm.req_to_blocks[req.request_id] = blocks

        # Populate vLLM's per-request caching bookkeeping so that the next
        # scheduler step doesn't try to re-hash blocks that our A patch
        # already hashed. num_cached_block is the count of already-hashed
        # full blocks; req_to_block_hashes is the ordered list of those
        # BlockHashType objects. vLLM's cache_full_blocks starts its work
        # at index num_cached_block, so anything before it must already
        # have block_hash set (which A ensures).
        block_size = kvm.block_size
        num_full = req.num_computed_tokens // block_size
        if num_full > len(blocks):
            num_full = len(blocks)
        prior_hashes = []
        for k in range(num_full):
            h = blocks[k].block_hash  # @property -> _block_hash
            if h is None:
                # Fell off the edge of the prefix cache — truncate
                num_full = k
                break
            prior_hashes.append(h)
        kvm.num_cached_block[req.request_id] = num_full
        kvm.req_to_block_hashes[req.request_id] = prior_hashes

        sched.requests[req.request_id] = req
        if e['queue'] == 'running':
            sched.running.append(req)
            n_b_running += 1
        else:
            sched.waiting.append(req)
            n_b_waiting += 1

    print('[ckpt_engine_state] restored %d prefix-cache entries, %d running + %d waiting requests'
          % (n_a, n_b_running, n_b_waiting), file=sys.stderr)
    return True
