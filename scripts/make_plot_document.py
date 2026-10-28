'''Creates a LaTeX document of the change point timecourse plots ordered by AUC.'''

"""
Usage:
    python scripts/make_plot_document.py

Outputs (written to plots/changepoint_timecourses):
    plot_document.csv  - columns model_number, AUC, p50 sorted by AUC, then p50, then model_number
    plot_document.tex  - figures in a 2 x 2 grid per page in the order of the CSV
"""

import src.constants as cn  # type: ignore

import os
import re
import subprocess
import pandas as pd  # type: ignore

PLOT_DIR = os.path.join(cn.PROJECT_DIR, "plots", "changepoint_timecourses")
CSV_PATH = os.path.join(PLOT_DIR, "plot_document.csv")
TEX_PATH = os.path.join(PLOT_DIR, "plot_document.tex")
FILENAME_PATTERN = re.compile(r"changepoint_timecourse_(\d+)\.pdf$")
# Searched separately since the extracted title text may wrap or be out of order
P50_PATTERN = re.compile(r"accuracy=([0-9.]*[0-9])")
AUC_PATTERN = re.compile(r"AUC=([0-9.]*[0-9])")
COL_MODEL_NUMBER = "model_number"
COL_AUC = "AUC"
COL_P50 = "p50"
NUM_FIGURE_PER_PAGE = 4


#################################################################
# Functions
#################################################################
def extractTitleValues(path: str) -> tuple[float, float]:
    """Extracts the p50 accuracy and AUC from the title of a plot.

    Args:
        path: path to the plot PDF

    Returns:
        (p50, AUC)
    """
    text = subprocess.run(["pdftotext", "-l", "1", path, "-"],
            capture_output=True, text=True, check=True).stdout
    p50_match = P50_PATTERN.search(text)
    auc_match = AUC_PATTERN.search(text)
    if p50_match is None or auc_match is None:
        raise ValueError(f"No AUC/accuracy title found in {path}")
    return float(p50_match.group(1)), float(auc_match.group(1))


def makeDataframe() -> pd.DataFrame:
    """Constructs the sorted dataframe of model number, AUC, and p50 for all plots.

    Returns:
        DataFrame with columns model_number, AUC, p50
    """
    rows = []
    for filename in os.listdir(PLOT_DIR):
        match = FILENAME_PATTERN.match(filename)
        if match is None:
            continue
        p50, auc = extractTitleValues(os.path.join(PLOT_DIR, filename))
        rows.append({COL_MODEL_NUMBER: int(match.group(1)), COL_AUC: auc, COL_P50: p50})
    df = pd.DataFrame(rows, columns=[COL_MODEL_NUMBER, COL_AUC, COL_P50])
    return df.sort_values([COL_AUC, COL_P50, COL_MODEL_NUMBER]).reset_index(drop=True)


def makeSubfigure(model_number: int, auc: float, p50: float) -> str:
    """Creates the LaTeX for one plot in the grid."""
    return (
        "  \\begin{subfigure}{0.49\\textwidth}\n"
        f"    \\includegraphics[width=\\linewidth]{{changepoint_timecourse_{model_number}.pdf}}\n"
        f"    \\caption{{Model {model_number}: AUC={auc:.3f}, p50={p50:.3f}}}\n"
        "  \\end{subfigure}\n"
    )


def makeLatex(df: pd.DataFrame) -> str:
    """Creates a LaTeX document with the plots in a 2 x 2 grid per page in the order of df."""
    lines = [
        "\\documentclass{article}\n",
        "\\usepackage[margin=0.75in]{geometry}\n",
        "\\usepackage{graphicx}\n",
        "\\usepackage{subcaption}\n",
        "\\begin{document}\n",
    ]
    for start in range(0, len(df), NUM_FIGURE_PER_PAGE):
        lines.append("\\begin{figure}[p]\n  \\centering\n")
        for _, row in df.iloc[start:start + NUM_FIGURE_PER_PAGE].iterrows():
            lines.append(makeSubfigure(int(row[COL_MODEL_NUMBER]), row[COL_AUC], row[COL_P50]))
        lines.append("\\end{figure}\n\\clearpage\n")
    lines.append("\\end{document}\n")
    return "".join(lines)


#################################################################
# Main
#################################################################
if __name__ == "__main__":
    df = makeDataframe()
    df.to_csv(CSV_PATH, index=False)
    with open(TEX_PATH, "w") as fd:
        fd.write(makeLatex(df))
