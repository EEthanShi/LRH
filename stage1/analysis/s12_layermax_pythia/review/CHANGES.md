# Implementation R of the Pythia axis (review recomputation): changes against the OpenLLaMA originals (S0b, WRITTEN AND TESTED, NOT FROZEN)

Role `pyt-code-R`, 2026-10-10 (UTC), step S0b of `analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (sha256 `6de89603…879e7b`), sections 3, 5, 7–11. Written before any Pythia byte existed on ds2213. To be hashed in the one S0b `CODE FROZEN` line of the axis; not chmod-ed here (the freeze does that).

**Independence (plan §3).** R was written from the frozen plan, the OpenLLaMA originals (`analysis/s12_layermax_openllama/review/*`, `VERDICTS_SPEC.md`, `verdict_keys.py`, `compare_verdicts.py`, `guard.py`), the shared extraction-side files (`extract/pythia_axis/pyt_axis_common.py`, `launch_pyt.sh`) and the shared verdict contract of role `pyt-code-infra` (`VERDICTS_SPEC_PYT.md`, `verdict_keys.py`, `compare_verdicts.py`, `guard.py`; the counterpart of the OpenLLaMA contract that the three implementations share). No code and no output of the E or I writers was opened. The only E/I traces seen were directory names under `s0b_scratch/` (`E/run_fx`, `E/run_vp5b`).

**Contract versions R was built and tested against** (sha256 at the final test runs): `VERDICTS_SPEC_PYT.md` `2d5332eb…`, `verdict_keys.py` `6d20e04d…`, `compare_verdicts.py` `86c762ca…`, `guard.py` `a66a5711…`. A later change to any of them requires R's S0b tests to be rerun. After fix round R1: `VERDICTS_SPEC_PYT.md` `96dce74e…` (amendment A1, §15), the other three unchanged; R's battery was rerun on it (last section).

## Files

| file | origin | role |
|---|---|---|
| `pyt_axis_review.py` | copy of `ol_axis_review.py` (`00c98f2a…`; pristine in `orig_copy/`; diff `pyt_axis_review.py.diff`) | Entry point. Gate first (`pyt_review_gate.require`); reads E's (resp. I's) Pythia pair files, the 11 known LMs of its own format, Part B and the 7 feature stacks (clip check); computes every key of `verdict_keys.required_keys("ALL")` (A, B, C, H, R, D, G, F, K) plus `x|` extras; writes `results/S1_2_layermax_pythia/review/{E,I}/verdicts_review.json` (schema `PYT-VERDICTS v1`, impl `R-E`/`R-I`); compares with `early|indep/verdicts_pyt.json` (`--sections EI`) and `forecast/{E,I}/verdicts_forecast.json` (`--sections FK`) through the shared `compare_verdicts.py` (names only) and writes R's own value-level list `cmp/R<impl>_detail_<sec>.json`; F1(c) audit via the shared guard; `recomputed_verdicts.json`, `REVIEW_REPORT.md`. Exit 0 / 1 / 3. |
| `pyt_review_lib.py` | copy of `ol_review_lib.py` (`b224a24e…`; diff) | Pair readers and exact mKNN unchanged. Added: `M_unc` argmax and its entry null mean in both ctm functions; vectorised n-size trend machinery (`rho_matrix`, `TrendParts` with finite and usable masks, `diff_vec` common/legacy, `tr0_vec`); exact ρ equality (`rho_equal_exact`, `degenerate`) and `same_ranks`; `p_le`, `boot_draws`, `p_mag`, `tost_p` (contract §2 tolerances), `holm`. |
| `pyt_review_sections.py` | copy of `ol_review_sections.py` (`e2e41713…`; diff) | `Trends` generalised to n sizes with mode `common` (plan §5.1) or `legacy` (OpenLLaMA rule, V-P8 only); section A with `tr0`, `dtr_nv`, `dtr0` and the common-set D3 in common mode; near ties over every size pair; `order_changes` with the four tie transitions; new `section_B_all` (contract §4). The OpenLLaMA sections B (new/pilot), C, D, F, G, J stay for the V-P8(a) legacy run. |
| `pyt_review_pythia.py` | new | The Pythia rules of contract §§5–10: section C (P-P1, P-P2a(i)/(ii), P-P2a, P-P3, P-P5a, keep identity, per-comparison and per-ViT sign rules with the censoring-tie exclusion, D3, plug-in, OP-style reading, D1, D2, D4–D6), H (`Family`: H1 IUT, H2a/H2b, TOST for every metric × {full, ctm}, gate, Holm), R (leave-one-size-out, five sizes, grid5, leave-one-ViT-family-out, ρ := 0), D (cross-family Tr/ΔTr, 18-LM panel), F (P-P2b, h curve, flat band), K (category, F reasons, premise, M1–M5, readings, counted predictions, knife edge, τ and δ_b curves, `h_consistent`), G (provenance) and R's recheck of Part B (`x|` only). |
| `pyt_review_conf.py` | new | Frozen constants and axis configurations: plan constants (thresholds, margins, curves, band constants, PR* range, fixed depths, grid5 layers, registry), contract names (families, H2/H3/H4, key prefixes), and the S0b configurations (stand-in, OpenLLaMA n = 3, BLOOMZ n = 5). |
| `pyt_review_gate.py` | new | R's gate: the shared `guard.require_partb_frozen` plus R's production order checks (plan hash; a `CODE FROZEN` line that precedes the `Part B FROZEN` line; out_root in `results/S1_2_layermax_pythia/review/`). Test mode: out_root under `s0b_scratch/R/`. |
| `make_test_fixture_pyt.py` | copy of `make_test_fixture.py` (`e565004a…`; diff) | S0b stand-in (5 BLOOMZ + llama-13b + llama-30b as `synth__pyt-<size>`) in the production layout, symlinks to the known E/I pair files and features, a Part B in the contract §11 schema computed with scipy from the frozen `partB_features_ol.json`, clip values from the frozen `clip_thresholds_partB_ol.json`; `--flip-partb-material`, `--no-l2`. Test mode only. |
| `tests/test_vp9_r.py` | new | V-P9 known answers, R's share (114 hand-derived checks). |
| `tests/test_vp8_r.py` | new | V-P8 (a), (b), (c) for R's generalised aggregator. |
| `tests/dryrun_gate_r.sh` | new | 15 dry runs of R's entry points; each must exit 3 and write nothing. |
| `s0b_dryruns_R.json` | new | R's gated commands for `s0b_freeze.py --dryruns` (contract §14). |

Usage (after both main runs and `forecast_pyt.py`): `analysis/s12_layermax_pythia/launch_pyt.sh pyt_R_review 4 -- env PYTHONDONTWRITEBYTECODE=1 venv/bin/python analysis/s12_layermax_pythia/review/pyt_axis_review.py [--impl both|E|I] [--phase all|compute|compare]`. CPU only, no GPU. About 30 s per implementation on the stand-in.

Marker convention: the four copies carry `PYT-AXIS` (or the inherited `OL-AXIS`) on changed lines and on the headers of added blocks; the `.diff` next to each copy (`diff -u orig_copy/<original> <copy>`) is the authoritative record.

## Operational choices of R that neither the plan nor the contract fixes ([R-op])

1. A null p of a Holm member (no valid ViT) is treated as 1, so it never rejects.
2. *Withdrawn by lead ruling R1 (2026-10-10 ~03:05Z; see the last section).* Under x = `unc` all three M4 keys (`M4`, `M4_sizes`, `M4_n_zero`) are null. Until the ruling R wrote `M4_sizes` and `M4_n_zero` for both x, because their inputs do not depend on x; fixture X found that `forecast_pyt.py` and I write null there (4 keys).
3. `K|…|M5_A_sizes` is null outside A, A-att and A-prime (now an instance of ruling R1); inside them it is `""` when no extrapolated model is in band.
4. `K|…|M5_B`: "every failing model of P-P2a(i)" is read as every model outside its band (the `PP2ai|P|x|failing` list), whether or not (i) as a whole fails.
5. `G|clip_equal_partB`: R uses no clip threshold itself. It recomputes the 7 thresholds (`hrhv2.prep.prh_clip_threshold`, float64 all-layer stack) from the primary feature stacks and compares them bitwise with `clip_thresholds_partB_pyt.json`, which makes this an R-level V-P1.
6. l2: R counts l2 as run only if all 119 l2 pair files of the implementation exist. With none, every l2 key is null (contract §9). A partial set is never opened and is recorded in `x|l2_notes`.
7. The legacy `trend()` cross-check of the original R (`x|check|…`) is limited to the `full` and `ctm` protocols, because each call loops over 5040 relabellings in Python.
8. R's recheck of the Part B rules (B3, B4, B6: trends, materiality, directions, bands, extrapolation, B_late = B̄ count) is written as `x|partB_recheck_*` and never used for a verdict (contract §1).

## What R could not implement as written, and open points for the lead

1. **V-P9, the cross-implementation part** ("run end to end through E, I, R and forecast_pyt.py with 0 mismatches") is not R's to run. It needs E's and I's outputs on one fixture, and R did not open them. R's share passed (`s0b_scratch/R/R_s0b_record.json`).
2. **The shared V-P9 fixture is incomplete for the E format.** `s0b_scratch/PB/fixture_vp9/early/pairs/<pre>/` holds only the 119 `.json` files and no `.npz`. R's E path (and any E-format reader that needs the entry arrays) cannot read it. R's I path ran on it with the infra Part B: every one of the 12,954 compared values equals R's run on R's own stand-in Part B.
3. **V-P8(b)** lists no changed key: on the OpenLLaMA pair files no ViT is constant under one score only, so the common-set rule equals the legacy rule there. The rule difference is tested on synthetic rows in V-P9 (section 3 of `test_vp9_r.py`).
4. **Plan §8 per-ViT sign rule.** The rule is false when censoring ties a discordant pair; the example ρ 0.964 → 0.991 is reproduced in V-P9. R applies the contract's exclusion (`nexcl_tie`) and counts excluded ViTs instead of treating them as errors.
5. **D7 (attention sink)** has no contract key and needs the `_tok1` stacks, so R does not recompute it.
6. **ViT-bootstrap CIs of single trends** (plan §5.1 key `S1.2|bootvit|layermax-pythia|<stat>`) are not computed: `<stat>` is unspecified, and the contract makes them optional `x|` extras.
7. **Leftover directory.** `review/s0b_test/` was created by R's first dry run (`pyt_R_s0b_dryrun_1`), before the contract fixed the scratch root at `s0b_scratch/`. It holds launcher logs and markers, a mark file and a stderr file, and no code. It is not deleted (no-delete rule) and should be excluded from the S0b code manifest.
8. **Contract ambiguities other writers must read the same way:** choice 2 is settled by ruling R1 and choice 3 follows from it. Choice 4 (`M5_B`) is not settled by R1. Fixture X has not reached category B, so it has not been compared with `forecast_pyt.py` yet (ruling R3).

## S0b tests (test mode, CPU only, non-Pythia data; records under `analysis/s12_layermax_pythia/s0b_scratch/R/`)

- Dry runs: 15/15 cases exit 3 and write nothing (`pyt_R_s0b_dryrun_2`, `_3`, `_4`). The first run, `_1`, reported rc 1: every case exited 3, but the harness's own no-new-file check counted its scratch directory. That was a harness bug.
- V-P9 known answers: 114/114 (`pyt_R_s0b_vp9_1`, `_2`, `_3`).
- V-P8 (R's share): (a) 4904 A–C keys per implementation, 0 mismatches against the merged OpenLLaMA summary, also through `compare_verdicts.py --legacy`; 0/5230 against the original R outputs. (b) 0 keys change. (c) 0/720 BLOOMZ values differ (`pyt_R_s0b_vp8_2`, `_3`, `_4`). The first run, `_1`, failed: the output directory was not created (harness bug).
- Stand-in end to end, both implementations: 12,974 keys each, 0 required keys missing, 0 implementation-check violations, legacy `trend()` cross-check 0/24, clip 7/7 bitwise equal, Part B recheck consistent. R-E and R-I differ only in the three subsets that the frozen BLOOMZ E-format files never stored (769 keys; a fixture artefact). The flip variant reaches category E and the gate "open (H1 not a member)", and R's recheck flags the flipped Part B as intended. In the no-l2 variant all 6475 l2 keys are null.
- Every run, failed ones included: `s0b_scratch/R/R_test_runs.json`.

## Fix round R1 (lead rulings of 2026-10-10 ~03:05Z; role `pyt-code-R`, 03:14Z to 03:35Z)

**The rulings.** Recorded verbatim by `pyt-code-infra` in contract §15 (amendment A1, `VERDICTS_SPEC_PYT.md` `96dce74e…`, written at 03:19:13Z):
> (R1) GENERAL RULE: a key that the contract defines only for some variants (e.g. a modifier defined only for s_cal) is null in every other variant, in every implementation; so under x=unc all three M4 keys (M4, M4_n_zero, M4_sizes) are null.

R2 (`G|clip_equal_partB`, a type-2 fixture exclusion of I, POST HOC) and R3 (an H1-positive fixture variant, run by the fixture harness) need no change in R. R still evaluates V-P1 itself, from the features.

**What R changed (only `review/`).**
- `pyt_review_pythia.py` (`section_K`): `K|P|unc|M4_sizes` and `K|P|unc|M4_n_zero` are now `null`. Before, they held `""` and the int `R|P|n_scal_zero`. `K|P|unc|M4` was already `null`. Under x = `s_cal` the three keys are unchanged. The comment on `M5_A_sizes` now cites R1. sha256 `0b536455…` → `287893f5e342a1fddfc3292e2ed305a706d643d6a0e5981843bbbd6feef7279e`.
- `tests/test_vp9_r.py`, new section 11 (284 checks, written by hand from §10 and §15.1). It runs `section_K` on hand-made inputs, through all 8 categories and both x. It checks:
  - the M4 trio is non-null only for `s_cal`;
  - `premise`, `M2`, `M5_A` and `M5_A_sizes` are non-null only for A, A-att and A-prime;
  - `F_reason` is non-null only for F, and `M5_B` only for B;
  - `M2_x` and `M5_extrap_sizes` are written in every variant;
  - every null is JSON null.

  sha256 `25aac325…` → `bc3c234d013b42bdc48cf6dbde3343747d13a92ed09c6353eb637c99f6c3497c`.
- Pre-fix copies: `s0b_scratch/R/prev_code/r1_20261010T031400Z/`.

**Other R keys checked against R1 and the §15.1 domain table: no further change needed.** Already `null` outside their domain before the fix:
- `M5_A`, `M5_A_sizes`, `M5_B`, `M2`, `M2_H3`/`M2_H4`, `premise` and `F_reason`;
- `PP2b|…|wrong_all_extrap` and `PP2b|…|ok|…`;
- `H|…|H1|confirmed`, and the Holm keys and `holm_order` while the gate is closed;
- every l2 key when l2 was not run;
- Clopper–Pearson bounds with n = 0.

Written in every variant, as §15.1 item 5 says: `M1`, `M2_x`, `M3`, `M5_extrap_sizes`, `kedge`, `kedge_which`, the readings, and the counted and curve keys. `PP2b|P|unc|n_tie_fc` is 0. The record checks the full domain table on every R output of this round (`R1_domain_violations`, 0 everywhere).

**Open points that R1 does not settle (reported, not changed).**
1. `M5_B` (choice 4 above) has still not been compared with `forecast_pyt.py`: no fixture variant reaches category B.
2. With an empty common ViT set, R writes `H|P|x|H1|nV`, `H|P|x|H2|nV` and hence `K|P|x|nV` as 0, which is the literal "|V|" of §6. The TOST keys, including `H3|nV` and `H4|nV`, are null by §6 text. The §15.1 line "H values over a ViT set … non-null only when the set is non-empty" could be read to cover nV too. This is unreachable on every fixture (it needs all 17 ViTs invalid) and has never been compared.

**Tests of the fix round** (test mode, CPU only, `CUDA_VISIBLE_DEVICES` empty, at most 2 thread tokens; markers in `s0b_scratch/R/logs/`):
1. `pyt_R_s0b_r1_vp9_1`: rc 0, 398/398 (114 earlier checks + 284 new ones).
2. `pyt_R_s0b_r1_mutcheck_1`: rc 1 **by design**, the negative control. It runs the new test file against the pre-fix `pyt_review_pythia.py` (scratch copy `s0b_scratch/R/mutcheck_r1_20261010T031400Z/`). Result 366/398: the 32 failures are exactly `K|P|unc|M4_sizes` and `K|P|unc|M4_n_zero` (8 categories × 2 keys × 2 checks). Every other R1 check already passed on the old code.
3. `pyt_R_s0b_r1_dryrun_1`: rc 0, 15/15 exit 3, 0 new files. `pyt_R_s0b_r1_vp8_1`: rc 0, (a), (b) and (c) pass for E and I.
4. `pyt_R_s0b_r1_e2e_{final,flip,nol2,shared}_1`: rc 0. Against the pre-fix outputs of the same fixtures, the only differences are the unc M4 sub-keys: 4 keys per file, 2 in the no-l2 variant, and 0 `x|` differences.
5. `pyt_R_s0b_r1_X_{base,flipH1,bandfail,bandwide}_1`: R (compute and compare) on the freeze role's four round-2 fixture-X roots (`20261010T024645Z`), outputs in `s0b_scratch/R/r1_X_*`.
   - R vs `forecast_pyt.py` [FK]: **0/4510 on the E side and 0/4510 on the I side in all four variants** (it was 4/4510).
   - R-E vs E [EI]: 504, all type 1. R-I vs I: 1, `G|clip_equal_partB` (type 2, ruling R2). R-E vs R-I [ALL]: 769, all type 1.
   - rc 1 is expected: R exits 1 on any mismatch, and the exclusions are applied by the fixture harness, not by R.
6. Battery 2, the same steps with suffix `_2`, after amendment A1 changed `VERDICTS_SPEC_PYT.md` (`2d5332eb…` → `96dce74e…`). R reads no contract text at run time, and `verdict_keys.py`, `compare_verdicts.py` and `guard.py` are unchanged. Results: vp9 398/398, dry runs 15/15, V-P8 pass, stand-in outputs identical to battery 1 (0 differences), fixture X as in step 5.
7. Records: `R_s0b_record_r1_2.json` (`"pass": true`) and `R_test_runs_r1_2.json`, written by `s0b_scratch/R/write_records_r1_2.py`.
   - The first writer run, `pyt_R_s0b_r1_records_1` (files `…_r1.json`, kept), is superseded. It read the capped `mismatches` field of the comparison files, so it classified only the first 500 of 504 and 769 names.
   - Neither records run has an end marker in its own run list, because each was still running when it wrote the list.
- The round-1 records `R_s0b_record.json` and `R_test_runs.json` are unchanged.

**Coverage note for ruling R3.** In fixture X the descriptive version `prh_clip+l2`/`unc` already reaches category A in base, bandfail and bandwide, with:
- the gate open (H1 confirmed), so H2, H3 and H4 go through Holm;
- `premise` "verified", `M2` false, and `M5_A` false (true in bandfail).

R and `forecast_pyt.py` agree on all of these keys. No variant reaches A-att, A-prime, B, C or D, `premise` "unverified", `M5_B`, the H2b branch, or an A-family category in the confirmatory version (`prh_clip+l2`/`s_cal`).
