#!/usr/bin/env python3
"""
COOLANT Architecture Visualization Script

This script generates visual diagrams of the COOLANT model architecture
and exports them as image files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up matplotlib for better quality
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"


class COOLANTVisualizer:
    """Visualizer for COOLANT architecture."""

    def __init__(self):
        self.colors = {
            "text": "#3498db",  # Blue
            "image": "#e74c3c",  # Red
            "fusion": "#2ecc71",  # Green
            "attention": "#f39c12",  # Orange
            "loss": "#9b59b6",  # Purple
            "background": "#ecf0f1",  # Light gray
        }

    def draw_box(self, ax, x, y, width, height, text, color, text_color="white"):
        """Draw a box with text."""
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(box)

        # Add text
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=7,
            fontweight="bold",
        )

    def draw_arrow(self, ax, x1, y1, x2, y2, text="", color="black"):
        """Draw arrow between boxes."""
        arrow = ConnectionPatch(
            (x1, y1),
            (x2, y2),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc=color,
            lw=1.5,
        )
        ax.add_patch(arrow)

        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, text, ha="center", fontsize=6)

    def create_overview_diagram(self):
        """Create high-level overview diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.text(
            6,
            7.5,
            "COOLANT Architecture Overview",
            ha="center",
            fontsize=16,
            fontweight="bold",
        )

        # Input boxes
        self.draw_box(ax, 1, 6, 2, 0.8, "Text Input\n(B, 30, 200)", self.colors["text"])
        self.draw_box(ax, 9, 6, 2, 0.8, "Image Input\n(B, 512)", self.colors["image"])

        # Similarity Module
        self.draw_box(ax, 4, 5, 4, 1, "Similarity Module", self.colors["fusion"])
        self.draw_box(
            ax, 2, 4.5, 1.5, 0.6, "Text Aligned\n(B, 64)", self.colors["text"]
        )
        self.draw_box(
            ax, 8.5, 4.5, 1.5, 0.6, "Image Aligned\n(B, 64)", self.colors["image"]
        )

        # Detection Module
        self.draw_box(ax, 4, 3, 4, 1, "Detection Module", self.colors["attention"])

        # Final Classification
        self.draw_box(
            ax,
            5,
            1.5,
            2,
            0.8,
            "Final Classification\n(B, 2) - [Real/Fake]",
            self.colors["loss"],
        )

        # Arrows
        self.draw_arrow(ax, 2, 6, 4.5, 5.5)
        self.draw_arrow(ax, 10, 6, 7.5, 5.5)
        self.draw_arrow(ax, 2.75, 4.5, 5, 3.5)
        self.draw_arrow(ax, 9.25, 4.5, 7, 3.5)
        self.draw_arrow(ax, 6, 3, 6, 2.3)

        # Add subtitle
        ax.text(
            6,
            0.5,
            "Cross-modal Contrastive Learning for Multimodal Fake News Detection",
            ha="center",
            fontsize=10,
            style="italic",
        )

        plt.tight_layout()
        return fig

    def create_similarity_module_diagram(self):
        """Create detailed similarity module diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.text(
            7,
            9.5,
            "Similarity Module - Detailed Architecture",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Inputs
        self.draw_box(ax, 2, 8, 2, 0.8, "Text Raw\n(B, 30, 200)", self.colors["text"])
        self.draw_box(ax, 10, 8, 2, 0.8, "Image Raw\n(B, 512)", self.colors["image"])

        # Encoding Part
        self.draw_box(ax, 1, 6.5, 4, 1.2, "Encoding Part", self.colors["fusion"])
        self.draw_box(ax, 1.5, 5.8, 1.5, 0.5, "FastCNN", "lightblue")
        self.draw_box(ax, 3.5, 5.8, 1.5, 0.5, "Text Linear", "lightblue")
        self.draw_box(ax, 10.5, 6.5, 2, 0.8, "Image Linear", self.colors["image"])

        # Shared representations
        self.draw_box(
            ax, 2, 4.5, 1.5, 0.6, "Text Shared\n(B, 128)", self.colors["text"]
        )
        self.draw_box(
            ax, 10.5, 4.5, 1.5, 0.6, "Image Shared\n(B, 128)", self.colors["image"]
        )

        # Aligners
        self.draw_box(
            ax, 2, 3.5, 1.5, 0.6, "Text Aligner\n(128→64)", self.colors["text"]
        )
        self.draw_box(
            ax, 10.5, 3.5, 1.5, 0.6, "Image Aligner\n(128→64)", self.colors["image"]
        )

        # Aligned features
        self.draw_box(
            ax, 2, 2.5, 1.5, 0.6, "Text Aligned\n(B, 64)", self.colors["text"]
        )
        self.draw_box(
            ax, 10.5, 2.5, 1.5, 0.6, "Image Aligned\n(B, 64)", self.colors["image"]
        )

        # Similarity classifier
        self.draw_box(
            ax,
            5.5,
            1.5,
            3,
            0.8,
            "Similarity Classifier\nConcat→(B,128)→(B,2)",
            self.colors["attention"],
        )

        # Arrows
        self.draw_arrow(ax, 3, 8, 3, 7.1)
        self.draw_arrow(ax, 11, 8, 11.5, 7.3)
        self.draw_arrow(ax, 2.75, 5.8, 2.75, 5.1)
        self.draw_arrow(ax, 11.25, 6.5, 11.25, 5.1)
        self.draw_arrow(ax, 2.75, 4.5, 2.75, 4.1)
        self.draw_arrow(ax, 11.25, 4.5, 11.25, 4.1)
        self.draw_arrow(ax, 2.75, 2.5, 5.5, 1.9)
        self.draw_arrow(ax, 11.25, 2.5, 8.5, 1.9)

        plt.tight_layout()
        return fig

    def create_detection_module_diagram(self):
        """Create detailed detection module diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.text(
            8,
            11.5,
            "Detection Module - Detailed Architecture",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Inputs
        self.draw_box(ax, 2, 10, 2, 0.8, "Text Raw\n(B, 30, 200)", self.colors["text"])
        self.draw_box(ax, 12, 10, 2, 0.8, "Image Raw\n(B, 512)", self.colors["image"])

        # Encoding
        self.draw_box(ax, 6, 9, 4, 0.8, "Encoding Part", self.colors["fusion"])
        self.draw_box(ax, 3, 8, 1.5, 0.6, "Text Prime\n(B, 128)", self.colors["text"])
        self.draw_box(
            ax, 11.5, 8, 1.5, 0.6, "Image Prime\n(B, 128)", self.colors["image"]
        )

        # Unimodal Detection
        self.draw_box(
            ax, 1, 6.5, 3, 1.2, "Unimodal Detection", self.colors["attention"]
        )
        self.draw_box(ax, 1.5, 5.8, 1, 0.5, "Text Uni\n(128→16)", "lightblue")
        self.draw_box(ax, 2.5, 5.8, 1, 0.5, "Image Uni\n(128→16)", "lightcoral")
        self.draw_box(ax, 1.5, 5.2, 1, 0.5, "Text Uni SE\n(128→64)", "lightblue")
        self.draw_box(ax, 2.5, 5.2, 1, 0.5, "Image Uni SE\n(128→64)", "lightcoral")

        # Cross Module
        self.draw_box(
            ax, 7, 6.5, 2, 0.8, "Cross Module\nText×Image→64", self.colors["fusion"]
        )

        # SE Attention
        self.draw_box(
            ax, 11, 6.5, 3, 0.8, "SE Attention\nWeights (B,3)", self.colors["attention"]
        )

        # Attention Application
        self.draw_box(ax, 2, 4, 1.5, 0.6, "Text Final\n(B, 16)", self.colors["text"])
        self.draw_box(
            ax, 11.5, 4, 1.5, 0.6, "Image Final\n(B, 16)", self.colors["image"]
        )
        self.draw_box(ax, 7, 4, 1.5, 0.6, "Correlation\n(B, 64)", self.colors["fusion"])

        # Feature Concatenation
        self.draw_box(
            ax,
            6,
            2.5,
            4,
            0.8,
            "Feature Concatenation\nConcat→(B,96)",
            self.colors["attention"],
        )

        # Final Classifier
        self.draw_box(
            ax, 7, 1, 2, 0.8, "Final Classifier\n(96→64→64→2)", self.colors["loss"]
        )

        # Arrows
        self.draw_arrow(ax, 3, 10, 6.5, 9.4)
        self.draw_arrow(ax, 13, 10, 9.5, 9.4)
        self.draw_arrow(ax, 3.75, 8, 2.5, 7.1)
        self.draw_arrow(ax, 12.25, 8, 12.5, 7.1)
        self.draw_arrow(ax, 2, 5.8, 2, 4.6)
        self.draw_arrow(ax, 3, 5.8, 7, 6.9)
        self.draw_arrow(ax, 12, 5.8, 11.5, 4.6)
        self.draw_arrow(ax, 8, 6.5, 7.75, 4.6)
        self.draw_arrow(ax, 2.75, 4, 6, 2.9)
        self.draw_arrow(ax, 12.25, 4, 10, 2.9)
        self.draw_arrow(ax, 7.75, 4, 8, 3.3)
        self.draw_arrow(ax, 8, 2.5, 8, 1.8)

        plt.tight_layout()
        return fig

    def create_ambiguity_learning_diagram(self):
        """Create ambiguity learning diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.text(
            6,
            7.5,
            "Ambiguity Learning Module",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Inputs
        self.draw_box(ax, 2, 6.5, 2, 0.8, "Text Aligned\n(B, 64)", self.colors["text"])
        self.draw_box(
            ax, 8, 6.5, 2, 0.8, "Image Aligned\n(B, 64)", self.colors["image"]
        )

        # Encoders
        self.draw_box(
            ax, 1.5, 5, 3, 1, "Text Encoder\n(64→4) → μ₁, σ₁", self.colors["text"]
        )
        self.draw_box(
            ax, 7.5, 5, 3, 1, "Image Encoder\n(64→4) → μ₂, σ₂", self.colors["image"]
        )

        # Sampling
        self.draw_box(
            ax, 2, 3.5, 1.5, 0.6, "z₁ ~ N(μ₁,σ₁)\n(B, 2)", self.colors["text"]
        )
        self.draw_box(
            ax, 8.5, 3.5, 1.5, 0.6, "z₂ ~ N(μ₂,σ₂)\n(B, 2)", self.colors["image"]
        )

        # KL Divergence
        self.draw_box(
            ax,
            4.5,
            2.5,
            3,
            0.8,
            "Symmetric KL Divergence\nSKL = (KL₁₂ + KL₂₁)/2",
            self.colors["attention"],
        )

        # Weight Computation
        self.draw_box(
            ax,
            4.5,
            1,
            3,
            0.8,
            "Weight Computation\nweight_uni = (1-SKL)\nweight_corre = SKL",
            self.colors["fusion"],
        )

        # Arrows
        self.draw_arrow(ax, 3, 6.5, 3, 6)
        self.draw_arrow(ax, 9, 6.5, 9, 6)
        self.draw_arrow(ax, 2.75, 5, 2.75, 4.1)
        self.draw_arrow(ax, 9.25, 5, 9.25, 4.1)
        self.draw_arrow(ax, 2.75, 3.5, 4.5, 2.9)
        self.draw_arrow(ax, 9.25, 3.5, 7.5, 2.9)
        self.draw_arrow(ax, 6, 2.5, 6, 1.8)

        plt.tight_layout()
        return fig

    def create_loss_diagram(self):
        """Create loss components diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.text(
            5,
            7.5,
            "COOLANT Loss Components",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Loss components
        self.draw_box(
            ax,
            1,
            6,
            2.5,
            1,
            "Classification Loss\nCrossEntropy(logits, labels)",
            self.colors["loss"],
        )
        self.draw_box(
            ax,
            3.75,
            6,
            2.5,
            1,
            "Contrastive Loss\nInfoNCE(text, image)",
            self.colors["attention"],
        )
        self.draw_box(
            ax,
            6.5,
            6,
            2.5,
            1,
            "Similarity Loss\nCrossEntropy(sim, sim_labels)",
            self.colors["fusion"],
        )

        # Weights
        self.draw_box(ax, 1, 4.5, 2.5, 0.6, "Weight: 1.0", "lightgray")
        self.draw_box(ax, 3.75, 4.5, 2.5, 0.6, "Weight: 1.0", "lightgray")
        self.draw_box(ax, 6.5, 4.5, 2.5, 0.6, "Weight: 0.5", "lightgray")

        # Total Loss
        self.draw_box(
            ax,
            2.5,
            2.5,
            5,
            1,
            "Total Loss\ncls_loss×1.0 + cont_loss×1.0 + sim_loss×0.5",
            self.colors["loss"],
        )

        # Arrows
        self.draw_arrow(ax, 2.25, 4.5, 3.5, 3.5)
        self.draw_arrow(ax, 5, 4.5, 5, 3.5)
        self.draw_arrow(ax, 7.75, 4.5, 6.5, 3.5)

        # Add note
        ax.text(
            5,
            1,
            "Multi-component loss for comprehensive optimization",
            ha="center",
            fontsize=9,
            style="italic",
        )

        plt.tight_layout()
        return fig

    def export_all_diagrams(self, output_dir="diagrams"):
        """Export all diagrams as image files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        diagrams = [
            ("coolant_overview", self.create_overview_diagram),
            ("coolant_similarity_module", self.create_similarity_module_diagram),
            ("coolant_detection_module", self.create_detection_module_diagram),
            ("coolant_ambiguity_learning", self.create_ambiguity_learning_diagram),
            ("coolant_loss_components", self.create_loss_diagram),
        ]

        exported_files = []

        for name, create_func in diagrams:
            try:
                logger.info(f"Creating {name} diagram...")
                fig = create_func()

                # Save as PNG
                png_path = output_path / f"{name}.png"
                fig.savefig(
                    png_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )

                # Save as SVG for vector graphics
                svg_path = output_path / f"{name}.svg"
                fig.savefig(svg_path, format="svg", bbox_inches="tight")

                exported_files.extend([str(png_path), str(svg_path)])
                plt.close(fig)

                logger.info(f"Exported {name} to PNG and SVG")

            except Exception as e:
                logger.error(f"Error creating {name}: {e}")

        return exported_files


def main():
    """Main function to export COOLANT architecture diagrams."""
    logger.info("Starting COOLANT architecture visualization...")

    visualizer = COOLANTVisualizer()
    exported_files = visualizer.export_all_diagrams()

    logger.info(f"Successfully exported {len(exported_files)} diagram files:")
    for file_path in exported_files:
        logger.info(f"  - {file_path}")

    logger.info("Architecture diagrams exported successfully!")


if __name__ == "__main__":
    main()
