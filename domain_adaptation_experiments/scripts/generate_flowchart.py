"""
Generate Methodology Flowchart for SIST 2026 paper
Shows the complete domain adaptation pipeline

Run: python generate_flowchart.py
Output: figures/figure_methodology.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# IEEE figure settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 9

# Colors
COLOR_SOURCE = "#E8F4F8"  # Light blue
COLOR_TARGET = "#E8F8F0"  # Light green
COLOR_MODEL = "#FFF9E6"  # Light yellow
COLOR_METHOD_WEAK = "#FFE6E6"  # Light red
COLOR_METHOD_GOOD = "#FFE6F0"  # Light pink
COLOR_METHOD_BEST = "#E6FFE6"  # Light green
COLOR_ARROW = "#4A90E2"  # Blue
COLOR_SUCCESS = "#2ECC71"  # Green


def create_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    color,
    border_color="black",
    border_width=1.5,
    fontsize=9,
    fontweight="normal",
    alpha=0.9,
):
    """Create a rounded box with text"""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=border_width,
        alpha=alpha,
    )
    ax.add_patch(box)

    # Add text
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        wrap=True,
    )


def create_arrow(ax, x1, y1, x2, y2, label="", color=COLOR_ARROW, linewidth=2):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->,head_width=0.4,head_length=0.4",
        color=color,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x + 0.2,
            mid_y,
            label,
            fontsize=7,
            style="italic",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )


def create_flowchart():
    """Create the complete methodology flowchart"""

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5,
        9.5,
        "Domain Adaptation Pipeline Overview",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    # ============================================================
    # TOP: Source Domain (PlantVillage)
    # ============================================================
    create_box(
        ax,
        0.5,
        7.5,
        2.5,
        1.2,
        "Source Domain\nPlantVillage\n54,305 images\nLab conditions",
        COLOR_SOURCE,
        fontweight="bold",
    )

    # Accuracy label
    ax.text(
        1.75,
        7.3,
        "98.9% accuracy",
        ha="center",
        fontsize=7,
        style="italic",
        color="green",
        fontweight="bold",
    )

    # ============================================================
    # Arrow down to model
    # ============================================================
    create_arrow(ax, 1.75, 7.5, 1.75, 6.8, "Pre-train")

    # ============================================================
    # MIDDLE: Base Model
    # ============================================================
    create_box(
        ax,
        0.5,
        5.8,
        2.5,
        0.8,
        "MobileNet-V2\n(ImageNet + PlantVillage)",
        COLOR_MODEL,
        border_color="#FFA500",
        border_width=2,
    )

    # ============================================================
    # Arrow down to DA methods
    # ============================================================
    create_arrow(ax, 1.75, 5.8, 1.75, 5.2, "Adapt")

    # ============================================================
    # MIDDLE: Domain Adaptation Methods
    # ============================================================

    # Container box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, 2.5),
            9.4,
            2.5,
            boxstyle="round,pad=0.1",
            facecolor="white",
            edgecolor="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.3,
        )
    )
    ax.text(
        5,
        4.85,
        "Domain Adaptation Strategies",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    # Method 1: Self-Training
    create_box(
        ax,
        0.5,
        3.5,
        2.5,
        1.2,
        "Self-Training\n\nIterative\nPseudo-labeling",
        COLOR_METHOD_WEAK,
        border_color="#E74C3C",
    )
    ax.text(
        1.75, 3.3, "63.0%", ha="center", fontsize=8, fontweight="bold", color="#C0392B"
    )

    # Method 2: Progressive DA
    create_box(
        ax,
        3.5,
        3.5,
        2.5,
        1.2,
        "Progressive DA\n\n3 Stages:\nEasy → Hard → Joint",
        COLOR_METHOD_GOOD,
        border_color="#E67E22",
    )
    ax.text(
        4.75, 3.3, "66.5%", ha="center", fontsize=8, fontweight="bold", color="#D35400"
    )
    ax.text(
        4.75,
        3.05,
        "❌ Forgetting",
        ha="center",
        fontsize=6,
        style="italic",
        color="red",
    )

    # Method 3: Joint Training (BEST)
    create_box(
        ax,
        6.5,
        3.5,
        2.5,
        1.2,
        "Joint Training ✓\n\nAll classes\ntogether",
        COLOR_METHOD_BEST,
        border_color=COLOR_SUCCESS,
        border_width=2.5,
    )
    ax.text(
        7.75, 3.3, "79.1%", ha="center", fontsize=9, fontweight="bold", color="green"
    )
    ax.text(
        7.75,
        3.05,
        "✓ Best!",
        ha="center",
        fontsize=7,
        style="italic",
        color="green",
        fontweight="bold",
    )

    # ============================================================
    # Arrows from methods to target
    # ============================================================
    create_arrow(ax, 1.75, 3.5, 2.5, 2.5, color="#E74C3C", linewidth=1.5)
    create_arrow(ax, 4.75, 3.5, 4.5, 2.5, color="#E67E22", linewidth=1.5)
    create_arrow(ax, 7.75, 3.5, 6.5, 2.5, color=COLOR_SUCCESS, linewidth=2.5)

    # ============================================================
    # BOTTOM: Target Domain (PlantDoc)
    # ============================================================
    create_box(
        ax,
        3.5,
        1.2,
        3,
        1,
        "Target Domain - PlantDoc\n468 field images, 4 classes",
        COLOR_TARGET,
        border_color="green",
        border_width=2,
        fontweight="bold",
    )

    # Baseline accuracy
    ax.text(
        5, 0.95, "Baseline: 41.9%", ha="center", fontsize=7, style="italic", color="red"
    )

    # Final best result
    ax.text(
        5,
        0.75,
        "Best (Joint): 79.1% (+37.2%)",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="green",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#E6FFE6",
            edgecolor="green",
            linewidth=1.5,
        ),
    )

    # ============================================================
    # Side annotation: Key insight
    # ============================================================
    ax.text(
        9.5,
        6.5,
        "Key Insight:",
        ha="right",
        fontsize=8,
        fontweight="bold",
        color="darkblue",
    )
    ax.text(9.5, 6.2, "Joint Training", ha="right", fontsize=7.5, color="darkblue")
    ax.text(
        9.5,
        5.95,
        "avoids catastrophic",
        ha="right",
        fontsize=7,
        style="italic",
        color="darkblue",
    )
    ax.text(
        9.5,
        5.75,
        "forgetting!",
        ha="right",
        fontsize=7,
        style="italic",
        color="darkblue",
    )

    # ============================================================
    # Legend
    # ============================================================
    legend_y = 0.3
    ax.text(
        0.5,
        legend_y,
        "■ Source (Lab)  ■ Target (Field)  ■ Model  ■ DA Method",
        fontsize=6.5,
        color="gray",
    )

    plt.tight_layout()

    # Save
    output_path = "figures/figure_methodology.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300, format="pdf")
    print(f"✓ Methodology flowchart saved to {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING METHODOLOGY FLOWCHART")
    print("=" * 60)
    print()

    create_flowchart()

    print()
    print("=" * 60)
    print("✓ FLOWCHART GENERATED!")
    print("=" * 60)
    print()
    print("Output: figures/figure_methodology.pdf")
    print()
    print("Next steps:")
    print("1. Check figures/figure_methodology.pdf")
    print("2. Upload to Overleaf figures/ folder")
    print("3. Add to LaTeX in Section III (Methodology)")
    print()


if __name__ == "__main__":
    main()
