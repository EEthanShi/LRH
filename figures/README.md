# Figure and table code

This folder mirrors the parts of the figure workspace that the paper's figures and tables need. It is reached as `~/hrh_v2_figs` through `scripts/lrh_env.sh`, and its environment is `figures/venv` (reached as `~/hrh_v2_figs_venv`; install it from `requirements-figures.txt`).

| folder | content |
|---|---|
| `v8/F_*/` | one maker per experiment figure of the paper (`make_F_*.py`) and the values it plotted (`numbers.json`) |
| `v8/_style/` | the shared style (`hrh_style_v8.py`, Paul Tol palette) and colour-vision checks (`cvdlib.py`) |
| `F1/`, `v5/`, `v5_1/` | helpers from earlier figure versions that the v8 makers import |
| `writer/` | the scripts that write the appendix tables |

The mapping from paper figures to makers, how to run them, and what a full rebuild needs are in [`experiments/10_figures_tables/README.md`](../experiments/10_figures_tables/README.md). The PNGs shown in the top-level README are in `assets/`.
