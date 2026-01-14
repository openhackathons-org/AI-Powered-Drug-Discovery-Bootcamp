"""
Visualization Module - Plots and reports for CDK inhibitor design

Generates publication-quality visualizations of design results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import CDKConfig


class CDKVisualizer:
    """Generate visualizations for CDK inhibitor design results."""
    
    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
        self.output_dir = self.config.output_dir
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            "cdk4": "#27ae60",      # Green
            "cdk11": "#9b59b6",     # Purple
            "good": "#2ecc71",      # Bright green
            "warning": "#f39c12",   # Orange
            "bad": "#e74c3c",       # Red
            "neutral": "#3498db",   # Blue
        }
    
    def plot_affinity_scatter(
        self,
        df: pd.DataFrame,
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot CDK4 vs CDK11 affinity scatter.
        
        Args:
            df: DataFrame with IC50 predictions
            save_path: Optional path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cdk4_col = f"{self.config.on_target}_IC50_pred"
        cdk11_col = f"{self.config.anti_target}_IC50_pred"
        
        if cdk4_col not in df.columns or cdk11_col not in df.columns:
            cdk4_col = "cdk4_ic50_nm"
            cdk11_col = "cdk11_ic50_nm"
        
        valid_df = df.dropna(subset=[cdk4_col, cdk11_col])
        
        # Plot selectivity regions
        x_range = np.logspace(-1, 5, 100)
        
        # Excellent selectivity (>100x)
        ax.fill_between(x_range, x_range * 100, 1e6, alpha=0.1, color=self.colors["good"], label="Excellent (>100x)")
        # Good selectivity (>10x)
        ax.fill_between(x_range, x_range * 10, x_range * 100, alpha=0.1, color=self.colors["warning"], label="Good (>10x)")
        # Poor selectivity
        ax.fill_between(x_range, 0.1, x_range * 10, alpha=0.1, color=self.colors["bad"], label="Poor (<10x)")
        
        # Plot compounds
        scatter = ax.scatter(
            valid_df[cdk4_col],
            valid_df[cdk11_col],
            c=valid_df.get("total_score", self.colors["neutral"]),
            cmap="RdYlGn",
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5
        )
        
        # Diagonal line (equal binding)
        ax.plot(x_range, x_range, 'k--', alpha=0.5, label="Equal binding")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 100000)
        ax.set_ylim(0.1, 100000)
        ax.set_xlabel(f"{self.config.on_target} IC50 (nM)", fontsize=12)
        ax.set_ylabel(f"{self.config.anti_target} IC50 (nM)", fontsize=12)
        ax.set_title("CDK4 vs CDK11 Binding Affinity", fontsize=14, fontweight="bold")
        
        # Colorbar
        if "total_score" in valid_df.columns:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Total Score", fontsize=10)
        
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_score_distribution(
        self,
        df: pd.DataFrame,
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot distribution of component scores.
        
        Args:
            df: DataFrame with scores
            save_path: Optional path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        score_cols = [
            "binding_score", "selectivity_score", "avoidance_score",
            "qed_score", "sa_score_norm", "pains_score", "novelty_score_norm"
        ]
        
        available_cols = [c for c in score_cols if c in df.columns]
        
        if not available_cols:
            print("No score columns found in DataFrame")
            return None
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            data = df[col].dropna()
            
            ax.hist(data, bins=20, color=self.colors["neutral"], alpha=0.7, edgecolor="black")
            ax.axvline(data.mean(), color=self.colors["bad"], linestyle="--", label=f"Mean: {data.mean():.2f}")
            ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=8)
        
        # Total score in last panel
        if "total_score" in df.columns and len(axes) > len(available_cols):
            ax = axes[len(available_cols)]
            data = df["total_score"].dropna()
            ax.hist(data, bins=20, color=self.colors["good"], alpha=0.7, edgecolor="black")
            ax.axvline(data.mean(), color=self.colors["bad"], linestyle="--", label=f"Mean: {data.mean():.2f}")
            ax.set_xlabel("Total Score", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=8)
        
        # Hide empty axes
        for i in range(len(available_cols) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Score Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_top_compounds(
        self,
        df: pd.DataFrame,
        n: int = 10,
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot bar chart of top compounds.
        
        Args:
            df: DataFrame with scores (must have rank column)
            n: Number of top compounds to show
            save_path: Optional path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        top_df = df.nsmallest(n, "rank") if "rank" in df.columns else df.head(n)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total score bar chart
        ax1 = axes[0]
        compounds = top_df["compound_id"].tolist() if "compound_id" in top_df.columns else [f"C{i}" for i in range(len(top_df))]
        scores = top_df["total_score"].tolist()
        
        bars = ax1.barh(range(len(compounds)), scores, color=self.colors["good"], alpha=0.8)
        ax1.set_yticks(range(len(compounds)))
        ax1.set_yticklabels(compounds)
        ax1.set_xlabel("Total Score", fontsize=12)
        ax1.set_title("Top Compounds by Total Score", fontsize=12, fontweight="bold")
        ax1.invert_yaxis()
        
        # IC50 comparison
        ax2 = axes[1]
        x = range(len(compounds))
        width = 0.35
        
        cdk4_ic50 = top_df.get("cdk4_ic50_nm", top_df.get(f"{self.config.on_target}_IC50_pred", [0]*len(top_df)))
        cdk11_ic50 = top_df.get("cdk11_ic50_nm", top_df.get(f"{self.config.anti_target}_IC50_pred", [0]*len(top_df)))
        
        ax2.barh([i - width/2 for i in x], cdk4_ic50, width, label="CDK4", color=self.colors["cdk4"], alpha=0.8)
        ax2.barh([i + width/2 for i in x], cdk11_ic50, width, label="CDK11", color=self.colors["cdk11"], alpha=0.8)
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(compounds)
        ax2.set_xlabel("IC50 (nM)", fontsize=12)
        ax2.set_xscale("log")
        ax2.set_title("CDK4 vs CDK11 IC50", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_optimization_progress(
        self,
        history: List[Dict],
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot optimization progress over iterations.
        
        Args:
            history: List of dicts with iteration stats
            save_path: Optional path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure
        """
        if not history:
            print("No optimization history to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = [h.get("iteration", i) for i, h in enumerate(history)]
        
        # Best score over time
        ax1 = axes[0, 0]
        best_scores = [h.get("best_score", 0) for h in history]
        ax1.plot(iterations, best_scores, 'o-', color=self.colors["good"], linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Score")
        ax1.set_title("Best Score vs Iteration", fontweight="bold")
        
        # Mean score over time
        ax2 = axes[0, 1]
        mean_scores = [h.get("mean_score", 0) for h in history]
        ax2.plot(iterations, mean_scores, 's-', color=self.colors["neutral"], linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Mean Score")
        ax2.set_title("Mean Score vs Iteration", fontweight="bold")
        
        # Valid molecules ratio
        ax3 = axes[1, 0]
        valid_ratios = [h.get("valid_ratio", 1) for h in history]
        ax3.plot(iterations, valid_ratios, '^-', color=self.colors["warning"], linewidth=2)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Valid Molecule Ratio")
        ax3.set_title("Valid Molecules vs Iteration", fontweight="bold")
        ax3.set_ylim(0, 1.05)
        
        # Diversity
        ax4 = axes[1, 1]
        diversity = [h.get("diversity", 0) for h in history]
        ax4.plot(iterations, diversity, 'd-', color=self.colors["cdk11"], linewidth=2)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Diversity")
        ax4.set_title("Molecular Diversity vs Iteration", fontweight="bold")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_report(
        self,
        scores_df: pd.DataFrame,
        summary: Dict,
        output_dir: str = None
    ) -> str:
        """Generate HTML report with all visualizations.
        
        Args:
            scores_df: DataFrame with scores
            summary: Summary statistics dict
            output_dir: Output directory (default: config.output_dir)
            
        Returns:
            Path to generated HTML report
        """
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self.plot_affinity_scatter(scores_df, save_path=output_dir / "affinity_scatter.png", show=False)
        self.plot_score_distribution(scores_df, save_path=output_dir / "score_distribution.png", show=False)
        self.plot_top_compounds(scores_df, save_path=output_dir / "top_compounds.png", show=False)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CDK Inhibitor Design Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2ecc71; }}
        .metric-label {{ color: #7f8c8d; }}
        img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>🧬 CDK Inhibitor Design Report</h1>
    
    <h2>Summary</h2>
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary.get('total_compounds', 0)}</div>
            <div class="metric-label">Total Compounds</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('max_total_score', 0):.3f}</div>
            <div class="metric-label">Best Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('potent_cdk4_count', 0)}</div>
            <div class="metric-label">Potent CDK4 (&lt;100nM)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('selective_count', 0)}</div>
            <div class="metric-label">Selective (&gt;10x)</div>
        </div>
    </div>
    
    <h2>Affinity Analysis</h2>
    <img src="affinity_scatter.png" alt="Affinity Scatter">
    
    <h2>Score Distribution</h2>
    <img src="score_distribution.png" alt="Score Distribution">
    
    <h2>Top Compounds</h2>
    <img src="top_compounds.png" alt="Top Compounds">
    
    <h2>Top 10 Compounds Table</h2>
    {scores_df.nsmallest(10, 'rank')[['compound_id', 'smiles', 'cdk4_ic50_nm', 'cdk11_ic50_nm', 'selectivity_ratio', 'total_score']].to_html(index=False)}
    
</body>
</html>
        """
        
        report_path = output_dir / "design_report.html"
        with open(report_path, "w") as f:
            f.write(html)
        
        print(f"Report generated: {report_path}")
        return str(report_path)

