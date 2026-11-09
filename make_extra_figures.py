#!/usr/bin/env python3
"""Figures 6-7 from sensitivity_summary.csv and comma_control.csv."""
import pandas as pd
import dna_storage_bench as B
import plots
out = B.FIG_DIR
plots.fig6(pd.read_csv(f"{B.OUT}/sensitivity_summary.csv"), out)
plots.fig7(pd.read_csv(f"{B.OUT}/comma_control.csv"), out)
