"""
Model evaluation and visualization module.

Generates:
- Predicted vs actual plots
- Residual plots
- Scenario comparison
- Model comparison
- Feature importance (if available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """Configure matplotlib style for professional plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_predicted_vs_actual(y_test, y_pred, output_path):
    """Create predicted vs actual plot."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=60, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual AER', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted AER', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs Actual AER Values', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_residuals(y_test, y_pred, output_path):
    """Create residual plot."""
    setup_plot_style()
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted AER', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_scenario_comparison(metrics_df, output_path):
    """Compare metrics across scenarios."""
    setup_plot_style()
    
    scenarios = metrics_df['scenario'].unique()
    metrics_list = ['r2', 'mae', 'rmse']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metric in enumerate(metrics_list):
        scenario_data = {}
        for scenario in scenarios:
            subset = metrics_df[metrics_df['scenario'] == scenario]
            scenario_data[scenario] = subset[metric].values
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, model in enumerate(metrics_df['model'].unique()):
            values = [metrics_df[(metrics_df['scenario'] == s) & (metrics_df['model'] == model)][metric].values[0]
                     for s in scenarios]
            axes[idx].bar(x + i*width, values, width, label=model.upper(), alpha=0.8)
        
        axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Scenario Comparison - {metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x + width)
        axes[idx].set_xticklabels([s.split()[0] for s in scenarios], rotation=15, ha='right')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_model_comparison(metrics_df, output_path):
    """Compare metrics across models."""
    setup_plot_style()
    
    models = metrics_df['model'].unique()
    metrics_list = ['r2', 'mae', 'rmse']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metric in enumerate(metrics_list):
        model_data = {}
        for model in models:
            subset = metrics_df[metrics_df['model'] == model]
            model_data[model] = subset[metric].mean()
        
        bars = axes[idx].bar(models, [model_data[m] for m in models], alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Average {metric.upper()} by Model', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels([m.upper() for m in models], rotation=0)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_feature_importance(model, feature_names, output_path, model_type='rf'):
    """Plot feature importance if available."""
    setup_plot_style()
    
    try:
        if model_type == 'rf':
            importances = model.named_steps['model'].feature_importances_
        elif model_type == 'gb':
            importances = model.named_steps['model'].feature_importances_
        else:
            return  # Linear regression doesn't have feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        transformed_names = preprocessor.get_feature_names_out()
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([transformed_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")


def generate_evaluation_plots(metrics_df, y_test, y_pred, best_model, best_model_type, output_dir):
    """Generate all evaluation plots."""
    print("\n" + "="*80)
    print("GENERATING EVALUATION PLOTS")
    print("="*80 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_predicted_vs_actual(y_test, y_pred, str(output_dir / 'predicted_vs_actual.png'))
    plot_residuals(y_test, y_pred, str(output_dir / 'residual_plot.png'))
    plot_scenario_comparison(metrics_df, str(output_dir / 'scenario_comparison.png'))
    plot_model_comparison(metrics_df, str(output_dir / 'model_comparison.png'))
    
    if best_model_type in ['rf', 'gb']:
        plot_feature_importance(best_model, [], str(output_dir / 'feature_importance.png'), best_model_type)
    
    print("\n" + "="*80 + "\n")
