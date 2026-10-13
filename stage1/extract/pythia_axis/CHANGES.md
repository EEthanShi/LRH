# Extraction-side code of the Pythia axis: changes against the OpenLLaMA-axis originals (S0b, WRITTEN, NOT FROZEN)

Role `pyt-axis-extract`, 2026-10-10, step S0b of `analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (sha256 `6de89603…`, sections 1, 2, 3, 11).

**Status.** Written and tested in S0b test mode. Not frozen: the plan's one `CODE FROZEN` line must cover all S0b code, and the E, I and R implementations, Part B, `guard.py`, feasibility, forecast and the verdict files are not written yet (plan section 3: E, I and R are each written by a separate sub-role).

**Conventions** (as the OpenLLaMA copies).
- Pristine originals are in `orig_copy/` (chmod 444). Their sha256 equal the plan's section 3 table and the OpenLLaMA S0b manifest.
- `X.diff` next to each copy is `diff -u orig_copy/<original> X`.
- Changed lines carry the marker `PYT-AXIS`; added blocks carry it on their header lines. `download_pyt.py` and `run_pyt.py` are restructured copies (S1/S3 and S2/S4 for 7 sizes): their `.diff` is the authoritative record, and markers sit on changed `def`/`import` lines and on block headers.

| copy | original (sha256) | what changed |
|---|---|---|
| `registry_pythia_axis.py` | `extract/openllama_axis/registry_openllama_axis.py` (`885caec2…`) | 7 sizes and size ranks; main pins and step143000 commits; `.bin` files at main with LFS sha256; the 2.8b step143000 `.bin`; small files with blob ids; exact allow-lists per revision; download order with byte totals asserted against plan section 1 (26,007,147,158 / 26,022,088,401 / 5,552,674,156 / 31,574,762,557); Hub facts; expected architecture; fixed depths asserted against plan section 3; loading-check patterns; GPU need per load; E2 reference (bloomz-560m); the 9 PolyPythia directory names for F-cache; nominal GPU minutes. Dropped: pooled ARH panel and ranks (Pythia is not in it), streamed-fp32 switch. |
| `pyt_axis_common.py` | `extract/openllama_axis/ol_axis_common.py` (`58fb4c4e…`) | Paths (features/wit1024_pythia_axis/{,_tok1,_e2_check}, results/S1_2_layermax_pythia/{extraction,logs}, dedicated `HF_HOME` extract/pythia_axis/hf_home); plan path and sha256; role tag; limits 20 GB free disk (statvfs on the HF_HOME filesystem) and 16 GB RAM; STATUS readers for `CODE FROZEN`, `S3 GO` (must cite G0 and the authorisation, and not be followed by `DEFERRED`) and `DEFERRED`; order-gate `require` (exit 3); S0b test mode; no-delete helpers (`keep_aside`, `move_marker_aside`); `compare_features(ref_rows=k)`. |
| `extract_core_pyt.py` | `extract/openllama_axis/extract_core_ol.py` (`425f02ab…`) | Pythia snapshots from the dedicated cache (2.8b source from the E1-PROV decision), exact allow-list check; bloomz-560m (E2) from the standard cache, read-only; fp32 load with `output_loading_info=True`; `loading_check` and `rotary_check` (plan section 2); token-0 diagnostics in the same forward pass (`_tok1` stack, sink statistics); E6 also for `_tok1`; streamed fp32 refused; markers moved aside instead of unlinked; device override only in test mode; sidecar block `pythia_axis`. The forward, pooling and NLL lines are unchanged. |
| `download_pyt.py` | `extract/openllama_axis/download_ol.py` (`93c11d61…`) | Dedicated `HF_HOME`, chunk cache 0, telemetry off, `token=False`. `resolve` (S1) for 7 repos at main, pin and step143000, refs, config.json read into memory. `download` (S3) gated on CODE FROZEN, S1, E2 run, `S3 GO` and G0 = GO; the disk guard before each download (free − this − reserve ≥ 20 GB, reserve = later downloads + 3 GB features + 3 GB results); E1 per snapshot with tensor inventory; one re-download then STOP; E1-PROV right after the 2.8b step143000 E1; `fcache` listing before S3 / after S4. |
| `run_pyt.py` | `extract/openllama_axis/run_ol.py` (`4231c834…`) | `e2` (bloomz-560m, descriptive, diagnostics on); `primary --size` (S4, one size per process, smallest first, G1 = GO, E1/E1-PROV, GPU free-memory check with exit 77, loading check STOP, E4, E6, E7); `e5` (transformers vs `tokenizers` ids, BOS/EOS, n_tokens, ids across sizes); `summary` (E7 bpb flag, F-cache after S4, EXTRACTION DONE); `ledger`; `diskwatch` (10 s watcher, stops the axis's own processes below 20 GB). E3, the streamed path and the tokenizer-sensitivity re-extraction are not ported. |
| `analysis/s12_layermax_pythia/launch_pyt.sh` | `analysis/s12_layermax_openllama/launch_ol.sh` (`3675d6f2…`) | 6 thread tokens, `PYT_THREADS`, Pythia ledger and run directory; an existing `.done`/`.failed`/`.started` marker refuses the launch (exit 65) instead of being removed. |

**New files.**
- `analysis/s12_layermax_pythia/e1prov_pyt.py`: E1-PROV (plan section 2), decision file `results/S1_2_layermax_pythia/extraction/e1prov_2.8b.json`.
- `analysis/s12_layermax_pythia/gpu_gate_pyt.py`: G0, G1 (plan section 11), `memwait` (plan section 2), `s22rate` (read-only S2.2 snapshot).
- `tests/`: `tests_mock_hub.py` (mock Hub from the designer's metadata JSON; fake downloader), `make_synthetic.py` (tiny GPT-NeoX snapshots in the Pythia checkpoint format), `test_extract_pyt.py` (13 CPU tests), `test_gpu_smoke.py` (one GPU step on known data).

**Plan erratum found by the S0b mocked S1 test** (no rule of the plan depends on it; reported to the lead).
- Plan section 1 summarises the card datasets as "EleutherAI/pile" for all 7 repos. The designer's metadata (`design/hub_pythia_main.json`, sha256 `592eca9a…`, the plan's source) shows `['the_pile']` for 1b and `['EleutherAI/the_pile']` for 1.4b. S1 checks the observed per-repo values (`CARD_DATASETS`); a strict reading of the summary would have stopped S1 on a label.

**Checked, no change.** The section 1 rule "expected num_params = Hub F16 parameter count minus L · rotary_ndims/2" holds exactly for all 7 sizes on the designer's metadata (mocked S1), and the GPT-NeoX parameter formula reproduces all 7 expected counts from the architecture (test t03).

**Choices the plan leaves open, made here (all reported).**
- E2 runs with the token-0 diagnostics switched on, so E2 also checks that they leave the primary stack bitwise unchanged; the bloomz `_tok1` stack goes to `_e2_check/_tok1/` (+0.1 GB).
- Sink statistics: the plan's `sink_<size>.json` (per layer: median ratio, max and median of both maxima) plus `sink_<size>.npz` with the per-caption arrays.
- E1 records whether every non-buffer tensor is F16 ("F16 weights expected") but does not stop on it; E1 stops on hashes, blob ids, the weights_only+mmap load, the config architecture and the 6.9b index.
- The GPU free-memory wait runs before the GPU step (`gpu_gate_pyt.py memwait`), so waiting is not counted as GPU minutes; inside the step a shortage exits 77 (deferral).
- Every sub-command that can write an axis output refuses (exit 3) before the CODE FROZEN line; `gpu_gate_pyt.py g0` is the only gate allowed before it (G0 belongs to S0b).
