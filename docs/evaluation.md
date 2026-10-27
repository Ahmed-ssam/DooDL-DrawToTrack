# Evaluation

**Metrics.** J (region IoU), F (boundary F-measure, tolerance 0.8% of the diagonal) and J&F, the mean of both, skipping the prompt frame, as in DAVIS semi-supervised evaluation.

**Synthetic clip.** `doodl.samples` renders a ball and a square crossing over a panning textured background, with exact masks. Prompts are loose circles (1.4x the true radius) on frame 0.

**Run.** `python scripts/benchmark.py --backend sam2 --model tiny --strategy box_points --markdown`. Repeat for `--model small`, and for `--strategy mask` and `grabcut` to compare how outlines become prompts.

**Results.** See the README table. Classical baseline: J&F 0.846 (ball 0.71, square 0.98) at about 5 FPS on CPU. SAM 2.1 numbers must be measured on a GPU and entered by hand.

**Caveats.** One short synthetic clip is a sanity check. For a real claim, evaluate on DAVIS 2017 val (not yet wired into the script). Report GPU, precision and resolution with every number.
