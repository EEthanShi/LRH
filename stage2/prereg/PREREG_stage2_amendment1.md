# PREREG_stage2 amendment 1 (2026-10-08, before gate G2 is evaluated)

Applies to PREREG_stage2.md v1.0 (sha256 6ce39abe7094034d9940fdc2acc62c82583a0481b5a3561bcfe484de84fba717), §2.3 gate G2, time criterion only.

**Problem.** v1.0 states "median wall time per run ≤ 300 s". The pilot runs 3 queue workers at once on one GPU, which another role's processes also use (about 14 GB and most of the SMs at the time). A run's own wall-clock time is therefore about 3× its GPU cost, plus contention. The criterion's purpose (final_plan_en.md G2, "≤ 5 min per run") is the cost per run of the program, not the latency of a single run.

**Change.** The time criterion is evaluated as the **effective time per run** = (completion time of the last pilot run − start time of the first pilot run) / number of pilot runs, i.e. the wall-clock throughput of the whole pilot. The literal median wall time per run is still reported. No other gate criterion, tolerance rule, endpoint or decision rule changes.

**What was known when this was written.** Only the first 3 pilot runs (one per arm) had finished: wall 406–408 s each under 3-way concurrency, and kNN accuracy on h of 0.787 (A), 0.795 (B) and 0.803 (A_τ). No pilot ladder or s_N value had been computed. The change is operational (cost accounting) and cannot favour any scientific outcome.
