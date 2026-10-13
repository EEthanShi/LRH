# Extraction copies for the OpenLLaMA axis: changes against the LLaMA-axis originals (S0b)

Role `oll-axis-extract`, 2026-10-09, step S0b of `analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710…`, section 2 "Code", section 3, section 11).

**Conventions.**
- Pristine copies of the originals are in `orig_copy/` (chmod 444).
- `X.diff` next to each copy is `diff -u orig_copy/<original> X`.
- Changed lines carry the marker `OL-AXIS`. Added blocks carry it on their header lines.

| copy | original (sha256) | what changed |
|---|---|---|
| `registry_openllama_axis.py` | `extract/registry_llama_axis.py` (`1af88e93…`) | Model list (3b/7b/13b), pins, .bin weight shards, the small files of 13b, the exact download allow-list, the Hub facts S1 must re-observe, the expected architecture (num_params 13,015,864,320), the family `OpenLLaMA`, the 11-LM pooled panel and ARH ranks (13b = 8), the paths, and the stored 3b/7b npy sha256 for E2. Also the shard-total erratum below. |
| `ol_axis_common.py` | `extract/llama_axis/llama_axis_common.py` (`1e4aef6d…`) | Paths, the plan path and sha256, the role tag `[oll-axis-extract]`, and the stored 3b/7b references. Added: the RAM/disk STOP watch (`ResourceWatch`; plan section 2) and the GPU-ledger reader. |
| `extract_core_ol.py` | `extract/llama_axis/extract_core_llama.py` (`5b26c082…`) | Registry import, and `code_hashes()` over `openllama_axis/`. `ensure_lm_snapshot` accepts .bin shards (with the index check) and refuses `*.py`. `bin_dtypes` / `weight_dtypes` count tensor dtypes (torch.load with weights_only=True and mmap=True). `WeightsOnlyGuard` makes every torch.load during the model load require `weights_only=True`, and records the calls. The sidecar block is named `openllama_axis`. The forward, pooling and NLL code is unchanged. |
| `download_ol.py` | `extract/llama_axis/download_llama_axis.py` (`7e5e2b81…`) | New `resolve` (S1) with every Hub check of plan section 1. `download` (S3) is gated on CODE FROZEN, S1 and E2, and polls free disk. E1 is built for .bin pickles: per-shard sha256, the weights_only+mmap load, dtypes, and legacy `rotary_emb.inv_freq` buffers in their own field. A mismatching shard is re-downloaded once, then the role STOPs. `verify` re-runs E1. |
| `run_ol.py` | `extract/llama_axis/run_llama_axis.py` (`12bcaa29…`) | `e2` covers 3b and 7b (all layers plus lm_aux, bitwise, records driver and kernel), gated on CODE FROZEN and S1. `primary` covers 13b with the standard fp32 load, gated on CODE FROZEN, S1, E1 and E2. On a load failure it exits 8 and the streamed path is allowed only after `e3`. `e3` checks 7b streamed vs stored. `e5` checks 13b vs 7b vs 3b ids, fast vs slow for 13b, and the tokenizer-file identity. `summary` is new. Threads come from `OL_THREADS`, at most 4. |

**Not copied.**
- `extract/llama_axis/summarize_extraction.py`. Its job is done by `run_ol.py summary`.
- `extract/extract_core.py`. The plan section 3 lists it as an original, for its hash. The copy derives from `extract_core_llama.py`, which is itself a marked copy of it.

**Plan erratum found by the S0b tests** (mocked S1 run; no rule depends on it).
- Plan section 1 says "3 `.bin` shards, 26,032,458,406 B". The three per-shard sizes in the same table sum to 26,031,877,079 B.
- 26,032,458,406 B is the total of all files at the pin. It adds the 7 small files and `.gitattributes`, i.e. +581,327 B.
- S1 checks both totals: shards = 26,031,877,079 and all files = 26,032,458,406.

**S0b tests** (scratch `~/oll_s0b_scratch/lead/`, not citable):
- **Mocked S1.** A fake HfApi built from the designer's Hub JSON: PASS. With `gated="auto"` injected: STOP, exit 2.
- **E1 `verify` on the cached open_llama_7b snapshot.** 2 shards, sha256 = blob names. weights_only+mmap load ok. 291 F16 weights; 32 legacy F32 inv_freq buffers, kept separate. E1 PASS.
- **GPU smoke run of the fp32 path under driver 580.178.04.**
  - open_llama_3b, first 16 items, and open_llama_7b, first 8 items: bitwise equal to the stored features on all layers, and lm_aux bitwise equal.
  - torch.load was observed with weights_only=True (mmap=True, cpu) on every .bin shard.
  - E4 semantics 0.0; rerun max|d| 0.
  - 0.37 GPU-min, logged in the axis ledger.

**Superseded drafts.** `superseded_draft_0949Z/` holds drafts of a second `oll-axis-extract` instance. They are not frozen and must not be run (see its README).
