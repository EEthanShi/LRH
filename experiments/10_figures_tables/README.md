# 10 · Figures and tables

**What it is.** The makers of the 14 experiment figures of the paper (Figures 2 to 6 and appendix Figures F.1 to F.9) and the scripts that write the appendix tables. Figure 1 is a vector drawing and has no maker here; `assets/fig1_lattice.svg` is its rendering.

**Paper.** All experiment figures and appendix tables.

**Status.** Not preregistered; figures and tables only redraw shipped results. Every value plotted in a figure is listed in its `numbers.json`.

| paper figure | maker | plotted values |
|---|---|---|
| Figure 2 | `figures/v8/F_neighbors/make_F_neighbors.py` | `figures/v8/F_neighbors/numbers.json` |
| Figure 3 | `figures/v8/F_criteria/make_F_criteria.py` | `figures/v8/F_criteria/numbers.json` |
| Figure 4 | `figures/v8/F_aggregation/make_F_aggregation.py` | `figures/v8/F_aggregation/numbers.json` |
| Figure 5 | `figures/v8/F_gauge/make_F_gauge.py` | `figures/v8/F_gauge/numbers.json` |
| Figure 6 | `figures/v8/F_link_top/make_F_link_top.py` | `figures/v8/F_link_top/numbers.json` |
| Figure F.1 | `figures/v8/F_scale_app/make_F_scale_app.py` | `numbers.json` next to it |
| Figure F.2 | `figures/v8/F_e0_app/make_F_e0_app.py` | `numbers.json` next to it |
| Figure F.3 | `figures/v8/F_mst_app/make_F_mst_app.py` | `numbers.json` next to it |
| Figure F.4 | `figures/v8/F_aggregation_app/make_F_aggregation_app.py` | `numbers.json` next to it |
| Figure F.5 | `figures/v8/F_linae_app/make_F_linae_app.py` | `numbers.json` next to it |
| Figure F.6 | `figures/v8/F_gauge_app/make_F_gauge_app.py` | `numbers.json` next to it |
| Figure F.7 | `figures/v8/F_gauge_v11_app/make_F_gauge_v11_app.py` | `numbers.json` next to it |
| Figure F.8 | `figures/v8/F_link_net/make_F_link_net.py` | `numbers.json` next to it |
| Figure F.9 | `figures/v8/F_link_app/make_F_link_app.py` | `numbers.json` next to it |

Shared style: `figures/v8/_style/hrh_style_v8.py` (Paul Tol palette) and `cvdlib.py` (colour-vision checks). The helpers in `figures/F1/`, `figures/v5/` and `figures/v5_1/` are imported by the v8 makers. Tables: `figures/writer/make_tables.py` (most appendix tables, with a manifest of the sha256 of every input), `figures/writer/make_llama_table.py` (LLaMA-1 trends, cross-checked against the second implementation), and `figures/writer/add_s21_compute.py` (a block already merged into `make_tables.py`, kept for provenance; it does not run on its own). The table writer reads `stage2/results/review_final/s22-skeptic/a2b_summary.json` (shipped, with its maker `a2b_summarize.py`), and the LLaMA-1 table reads the second-implementation summary `stage1/results/S1_2_layermax_llama/indep/summary.json` (shipped).

**Steps.**
```bash
python3.12 -m venv figures/venv && figures/venv/bin/pip install -r requirements-figures.txt
scripts/lrh_env.sh bash -c 'OMP_NUM_THREADS=1 ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v8/F_neighbors/make_F_neighbors.py'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_figs/writer && ~/hrh_v2_figs_venv/bin/python make_tables.py'
```

**Limits.** The makers read the shipped summaries, but several also read raw run files that are not shipped (the finite-Gibbs runs for `F_link_top`, `F_link_net` and `F_link_app`, the linear-autoencoder runs for `F_gauge` and `F_linae_app`, the contrastive-network ladders), and every v8 maker also reads files of earlier figure versions (v4 to v7), which are not shipped either. A full rebuild therefore needs the original result trees. The makers document how each panel was drawn, and `numbers.json` holds every plotted value, with the sha256 of each input file. In the shipped `numbers.json` files the absolute home-directory prefix of the experiment machine was replaced by `~/` (see [`results/README.md`](../../results/README.md)).

**Rendering for this README.** The PNGs in `assets/` are 300 dpi renderings of the paper's figure PDFs (PyMuPDF, opaque white background).
