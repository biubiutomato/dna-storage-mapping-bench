# DNA storage mapping benchmark

Code and results for:

> Ye Ye. **A Simple Benchmark of Coding Density and Error Robustness for DNA Data Storage.**
> IEEE BIBM 2026 Workshops (DNA-ISBC, paper S23203).

Six elementary base mappings are compared under the same random-payload model, constraint window
(GC 40–60%, homopolymer ≤ 3 by default), and substitution / insertion / deletion channels, with no outer code.

| Paper name | Name in code / CSVs |
|---|---|
| Naive2Bit | `Naive2Bit` |
| Screened2Bit | `Screened2Bit` |
| Block6to4 | `Block6to4` |
| TernaryRotate | `GoldmanRotate` |
| Pair1Bit | `Church1Bit` |
| CommaRotate | `MarkerResync` |

## Reproduce

Python ≥ 3.9 with numpy, pandas, matplotlib (`pip install -r requirements.txt`). Outputs go to `results/` and `figures/`.

```bash
python dna_storage_bench.py    # Tables I, III; Figs. 2-5      (seed 20260927)
python extend_bench.py         # Table II sweep                (seed 20260928)
python comma_control.py        # Table V, basic comma decoder  (seed 20260929)
python make_extra_figures.py   # Figs. 6-7 (original versions)
python capacity.py             # constrained capacity (transfer matrix, exact counts)
python revision_analyses.py    # CommaRotate rows (seed 20260930), alignment-based similarity (Table IV),
                               # strict comma decoder, Wilson and cluster-bootstrap intervals
python revision_figures.py     # final Figs. 1, 4, 7
```

`revision_analyses.py` consumes the random streams exactly as the original scripts do, so its re-runs of
Tables I, III, and V reproduce the original numbers and add the new columns (`results/rev_*.csv`).

Notes: bit similarity in `error_summary.csv` is positional (Hamming plus length penalty); the alignment-based
score is `edit_similarity` in `results/rev_error_summary.csv`. All validity statements refer to the in-silico
constraint screen; no sequences were synthesized.

## Citation

```bibtex
@inproceedings{ye2026dnabench,
  author    = {Ye, Ye},
  title     = {A Simple Benchmark of Coding Density and Error Robustness for {DNA} Data Storage},
  booktitle = {Proc. IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
