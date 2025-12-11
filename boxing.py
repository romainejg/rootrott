import os
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from io import BytesIO


def generate_box_plot_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    plot_title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    """
    Create a box + strip plot figure (no GUI, no saving, just returns a Matplotlib Figure).

    This is the logic from your original `generate_box_plot`, adapted to work
    in a headless environment (e.g. Streamlit).
    """
    if not all([x_column, y_column, plot_title, x_label, y_label]):
        raise ValueError("x_column, y_column, plot_title, x_label, and y_label must all be provided")

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Order categories by mean of y
    order = data.groupby(x_column)[y_column].mean().sort_values(ascending=False).index

    # Color palette
    box_palette = ["#64D273", "#9B91F9", "#ED695D", "#FFD750"]
    box_colors = {key: box_palette[i % len(box_palette)] for i, key in enumerate(order)}

    # Boxplot
    box_plot = sns.boxplot(
        x=x_column,
        y=y_column,
        data=data,
        order=order,
        showmeans=True,
        meanprops={
            "marker": "*",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 15,
        },
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        ax=ax,
    )

    # Color each box edge
    for i, box in enumerate(box_plot.artists):
        box.set_edgecolor(box_colors[order[i]])
        box.set_linewidth(2)

    # Stripplot
    sns.stripplot(
        x=x_column,
        y=y_column,
        data=data,
        order=order,
        jitter=True,
        hue=x_column,
        palette=box_colors,
        edgecolor="none",
        linewidth=0,
        dodge=False,
        legend=False,
        ax=ax,
    )

    # Overlay mean markers
    means = data.groupby(x_column)[y_column].mean().reindex(order)
    for i, mean in enumerate(means):
        ax.plot(i, mean, marker="*", color="black", markersize=15, zorder=10)

    # Titles and labels
    ax.set_title(plot_title, fontsize=16, fontweight="bold", fontname="Urbane Rounded")
    ax.set_xlabel(x_label, fontsize=14, fontname="Rubik")
    ax.set_ylabel(y_label, fontsize=14, fontname="Rubik")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Background and grid
    ax.set_facecolor("#E3E3E2")
    ax.grid(color="white", which="both", linestyle="-", linewidth=1)
    ax.xaxis.grid(True, linestyle="-", linewidth=1, color="white")

    # Optional: adjust y-axis tick spacing (approx half the default step)
    yticks = ax.get_yticks()
    if len(yticks) > 1:
        step = yticks[1] - yticks[0]
        ax.yaxis.set_major_locator(MultipleLocator(step / 2))

    fig.subplots_adjust(bottom=0.2)

    return fig


def generate_box_plot_from_excel(
    file: Union[str, bytes, bytearray],
    x_column: str,
    y_column: str,
    plot_title: str,
    x_label: str,
    y_label: str,
    sheet: Optional[Union[str, int]] = None,
) -> plt.Figure:
    """
    Convenience function:
    - Load Excel (path or bytes)
    - Optionally choose a sheet (by name or index; default = first sheet)
    - Build DataFrame
    - Return a Matplotlib Figure with the box plot
    """
    # Create an ExcelFile object so we can choose sheet
    if isinstance(file, (bytes, bytearray)):
        excel = pd.ExcelFile(BytesIO(file))
    else:
        excel = pd.ExcelFile(file)

    # Default to first sheet if none specified
    if sheet is None:
        sheet = excel.sheet_names[0]

    data = excel.parse(sheet_name=sheet)

    return generate_box_plot_figure(
        data=data,
        x_column=x_column,
        y_column=y_column,
        plot_title=plot_title,
        x_label=x_label,
        y_label=y_label,
    )


def save_box_plot_png(
    fig: plt.Figure,
    output_path: str,
) -> None:
    """
    Save a Matplotlib figure as a PNG file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
