"""Paper tables from the ``metrics.csv`` files of ``run_metrics.py``.

Every table row names a dataset (``dataset_short``), model family, and the player counts whose ``metrics_n{n}`` runs it reads.
Medians and inter-quartile ranges are taken over test samples and the three query points per (row, players, method). Bold marks
the best median and every median within one standard error of it, all rounded to the three printed decimals. The standard error
is taken over test samples, from the median over the query points of each sample. Writes to ``results/tables``
``all_pde_metrics_short.tex`` (main text), ``all_pde_metrics_full_top.tex`` / ``_bot.tex`` (with IQR rows), and the ablation
tables ``holdout.tex``, ``mean_baseline.tex``, ``relu.tex``, and ``midpoint.tex``, and every median of these tables to
``numbers.tex`` as ``\\res{dataset/family/players/method/metric}`` for the text.

Usage
-----
    python make_tables.py
"""

import pandas as pd

from operator_shap.results import RESULTS, load_metrics, write_numbers

LADDER_1D = (32, 64, 128, 256, 512, 1024)
LADDER_ND = (64, 256, 1024)
LADDER_MID = (16, 64, 256, 1024)
BURGERS = ("burgers1d", "fno", (16, *LADDER_1D))
# Every table row: label, dataset_short, model family, and player counts.
MAIN_1D = [
    ("Heat 1D", "heat1d", "fno", LADDER_1D),
    ("Burgers 1D", *BURGERS),
    ("Allen--Cahn 1D", "ac1d", "fno", LADDER_1D),
    ("Diff.--Sorp. 1D", "diffsorp1d", "fno", LADDER_1D),
    ("Advection 1D", "adv1d", "fno", LADDER_1D),
]
MAIN_ND = [
    ("Heat 2D", "heat2d", "localFNO", LADDER_ND),
    ("Darcy 2D", "darcy2d", "localFNO", LADDER_ND),
    ("Reac.--Diff. 2D", "rxndiff2d", "localFNO", LADDER_ND),
    ("Shallow Water 2D", "sw2d", "localFNO", LADDER_ND),
    ("Heat 3D", "heat3d", "fno", (512,)),
    ("MGN Cylinder", "mgn_cyl", "gino", (64,)),
    ("MGN Airfoil", "mgn_air", "gino", (64,)),
]
HOLDOUT = [
    ("Burgers 1D, full ladder", *BURGERS),
    ("Burgers 1D, $n=256$ held out", "burgers1dho256", "fno", (256,)),
    ("Burgers 1D, $n=1024$ held out", "burgers1dho1024", "fno", (1024,)),
    ("Heat 2D, full ladder", "heat2d", "localFNO", LADDER_ND),
    ("Heat 2D, $n=256$ held out", "heat2dho256", "localFNO", (256,)),
    ("Heat 2D, $n=1024$ held out", "heat2dho1024", "localFNO", (1024,)),
]
MEAN_BASELINE = [
    ("Darcy 2D, zero", "darcy2d", "localFNO", LADDER_ND),
    ("Darcy 2D, mean", "darcy2dmean", "localFNO", (64, 1024)),
    ("Diff.--Sorp. 1D, zero", "diffsorp1d", "fno", LADDER_1D),
    ("Diff.--Sorp. 1D, mean", "diffsorp1dmean", "fno", (64, 1024)),
]
MIDPOINT = [
    ("Heat 1D", "heat1dmid", "fno", LADDER_MID),
    ("Burgers 1D", "burgers1dmid", "fno", LADDER_MID),
    ("Allen--Cahn 1D", "ac1dmid", "fno", LADDER_MID),
    ("Diff.--Sorp. 1D", "diffsorp1dmid", "fno", LADDER_MID),
    ("Advection 1D", "adv1dmid", "fno", LADDER_MID),
    ("Heat 2D", "heat2dmid", "localFNO", LADDER_MID),
    ("Darcy 2D", "darcy2dmid", "localFNO", LADDER_MID),
    ("Reac.--Diff. 2D", "rxndiff2dmid", "localFNO", LADDER_MID),
    ("Shallow Water 2D", "sw2dmid", "localFNO", LADDER_MID),
    ("Heat 3D", "heat3dmid", "fno", (512,)),
    ("MGN Cylinder", "mgn_cylmid", "gino", (64,)),
    ("MGN Airfoil", "mgn_airmid", "gino", (64,)),
]
RELU = [("GELU", *BURGERS), ("ReLU", "burgers1d", "fnoRelu", (64, 256))]
# Main-text table: one block per readout at the players of SHORT_PLAYERS, with IQR rows.
POINT_SHORT = [MAIN_1D[1], MAIN_ND[3], MAIN_ND[5]]
DECISION_SHORT = [r for r in MIDPOINT if r[0] in {label for label, *_ in POINT_SHORT}]
SHORT_PLAYERS = LADDER_ND
FULL_PLAYERS = LADDER_1D
METHODS = [("RMSR", "RMSR"), ("KernelSHAP", "KS"), ("IG", "IG"), ("OperatorSHAP", "OS (ours)")]
SEP = "!{\\vrule width 0.3pt}"   # thin rule between the metric blocks
METRICS = [("r2", "R$^2$ $\\uparrow$", True), ("pearson", "Pearson $\\uparrow$", True), ("nrmse", "NRMSE $\\downarrow$", False)]
Row = tuple[str, str, str, tuple[int, ...]]


def load_rows(rows: list[Row]) -> pd.DataFrame:
    """Rows of the ``metrics.csv`` files of every table row, labelled by the row."""
    return pd.concat([load_metrics(ds, fam, n).assign(row=label) for label, ds, fam, ns in rows for n in ns], ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles over samples and query points, and standard error of the median over samples, per (row, n_players, method)."""
    grouped = df.groupby(["row", "n_players", "method"])[[m for m, _, _ in METRICS]]
    agg = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    agg.columns = [f"{m}_{ {0.25: 'q25', 0.5: 'med', 0.75: 'q75'}[q] }" for m, q in agg.columns]
    per_sample = df.groupby(["row", "n_players", "method", "sample"])[[m for m, _, _ in METRICS]].median().groupby(["row", "n_players", "method"])
    # Normal approximation of the standard error of a median, 1.2533 * std / sqrt(N), with N the number of samples.
    se = 1.2533 * per_sample.std() / per_sample.count() ** 0.5
    return agg.join(se.add_suffix("_se")).reset_index()


def medians(agg: pd.DataFrame, rows: list[Row]) -> dict[str, float]:
    """Every median of ``agg`` keyed ``dataset/family/players/method/metric``."""
    keys = {label: f"{ds}/{fam}" for label, ds, fam, _ in rows}
    return {
        f"{keys[r['row']]}/{r['n_players']}/{r['method']}/{m}": float(r[f"{m}_med"])
        for _, r in agg[agg["row"].isin(keys)].iterrows() for m, _, _ in METRICS if pd.notna(r[f"{m}_med"])
    }


def _rows(agg: pd.DataFrame, rows: list[Row], players: tuple[int, ...] | None, iqr: bool) -> list[str]:
    lines = []
    for label, *_ in rows:
        block = agg[agg["row"] == label]
        all_ns = sorted(block["n_players"].unique())
        ns = [n for n in all_ns if players is None or n in players] or all_ns
        for i, n in enumerate(ns):
            cells, subs = [], []
            at_n = block[block["n_players"] == n]
            for m, _, higher in METRICS:
                vals = {
                    k: round(float(v), 3) for k, _ in METHODS
                    if (at_n["method"] == k).any() and pd.notna(v := at_n[at_n["method"] == k][f"{m}_med"].iloc[0])
                }
                best_k = (max if higher else min)(vals, key=vals.get) if vals else None
                tol = round(float(at_n[at_n["method"] == best_k][f"{m}_se"].iloc[0]), 3) if vals else 0.0
                bold = {k for k, v in vals.items() if round(abs(v - vals[best_k]), 3) <= tol}
                for k, _ in METHODS:
                    if k not in vals:
                        cells.append("--")
                        subs.append("")
                        continue
                    r = at_n[at_n["method"] == k].iloc[0]
                    cells.append(("\\textbf{%.3f}" if k in bold else "%.3f") % vals[k])
                    subs.append("\\tiny{[%.3f, %.3f]}" % (r[f"{m}_q25"], r[f"{m}_q75"]))
            lead = ("\\multirow{%d}{*}{%s}" % (len(ns) * (2 if iqr else 1), label)) if i == 0 else ""
            lines.append(f"{lead} & {n} & " + " & ".join(cells) + " \\\\")
            if iqr:
                lines.append(" &  & " + " & ".join(subs) + " \\\\")
        if ns:
            lines.append("\\cmidrule(l){1-%d}" % (2 + len(METHODS) * len(METRICS)))
    return lines[:-1]


def latex_table(
    blocks: list[tuple[str | None, pd.DataFrame, list[Row]]], players: tuple[int, ...] | None = None, iqr: bool = False, first: str = "PDE"
) -> str:
    """Tabular of ``(title, agg, rows)`` blocks separated by midrules; a block with a title starts with that title as a full-width line."""
    n_val = len(METHODS) * len(METRICS)
    head = [
        "\\small", "\\begin{tabular}{ll" + SEP.join("c" * len(METHODS) for _ in METRICS) + "}", "\\toprule",
        " &  & " + " & ".join("\\multicolumn{%d}{c%s}{%s}" % (len(METHODS), SEP if i < len(METRICS) - 1 else "", name) for i, (_, name, _) in enumerate(METRICS))
        + " \\\\",
        f"{first} & $n$ & " + " & ".join(short for _ in METRICS for _, short in METHODS) + " \\\\", "\\midrule",
    ]
    body = []
    for title, agg, rows in blocks:
        if body:
            body.append("\\midrule")
        if title:
            body.append("\\multicolumn{%d}{l}{\\emph{%s}} \\\\" % (2 + n_val, title))
        body += _rows(agg, rows, players, iqr)
    return "\n".join(head + body + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main() -> None:
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    main_agg = aggregate(load_rows(MAIN_1D + MAIN_ND))
    midpoint_df = load_rows(MIDPOINT)
    # IG attributions are identically zero on the decision game, so their Pearson correlation is undefined.
    midpoint_df.loc[midpoint_df["method"] == "IG", "pearson"] = float("nan")
    midpoint_agg = aggregate(midpoint_df)
    short = [("Point readout", main_agg, POINT_SHORT), ("Decision readout", midpoint_agg, DECISION_SHORT)]
    (out / "all_pde_metrics_short.tex").write_text(latex_table(short, SHORT_PLAYERS, iqr=True))
    (out / "all_pde_metrics_full_top.tex").write_text(latex_table([(None, main_agg, MAIN_1D)], FULL_PLAYERS, iqr=True))
    (out / "all_pde_metrics_full_bot.tex").write_text(latex_table([(None, main_agg, MAIN_ND)], FULL_PLAYERS, iqr=True))
    numbers = medians(main_agg, MAIN_1D + MAIN_ND) | medians(midpoint_agg, MIDPOINT)
    for name, rows, players, first in (
        ("holdout", HOLDOUT, (256, 1024), "Setting"), ("mean_baseline", MEAN_BASELINE, (64, 1024), "Baseline"), ("relu", RELU, (64, 256), "Activation"),
        ("midpoint", MIDPOINT, None, "PDE"),
    ):
        agg = midpoint_agg if rows is MIDPOINT else aggregate(load_rows(rows))
        (out / f"{name}.tex").write_text(latex_table([(None, agg, rows)], players, iqr=True, first=first))
        numbers |= medians(agg, rows)
    write_numbers(out / "numbers.tex", numbers)
    print(main_agg.round(3).to_string())


if __name__ == "__main__":
    main()
