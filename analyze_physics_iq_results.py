#!/usr/bin/env python3
"""
Analyze Physics-IQ benchmark results with detailed per-scenario metrics.

Usage:
    python analyze_physics_iq_results.py <csv_file_path> [--output_dir <output_directory>]

Example:
    python analyze_physics_iq_results.py physics_iq_results_stage2_long/results/.wan22_ti2v_stage2_lora.csv
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


VIEWS = ["perspective-left", "perspective-center", "perspective-right"]


def parse_list_of_floats(value):
    """Parse a string or list representing a list of floats."""
    if isinstance(value, str):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError("Invalid string format for list of floats.")
        try:
            parsed_floats = [round(float(x), 4) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)]
            if not parsed_floats:
                raise ValueError("No valid floats found in the input string.")
            return parsed_floats
        except ValueError:
            raise ValueError("Failed to parse floats from the input string.")
    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            parsed_floats = [round(float(x), 4) for x in value]
            if not parsed_floats:
                raise ValueError("Input list is empty or contains no valid numeric values.")
            return parsed_floats
        else:
            raise ValueError("List contains non-numeric values.")
    raise TypeError("Input must be a string representing a list of floats or a list of numbers.")


def load_and_parse_csv(csv_file_path):
    """Load CSV and parse list columns."""
    print(f"Loading CSV from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    # Identify list columns
    list_columns = [
        f"v1_mse_{view}" for view in VIEWS
    ] + [
        f"v2_mse_{view}" for view in VIEWS
    ] + [
        f"spatiotemporal_iou_v1_{view}" for view in VIEWS
    ] + [
        f"spatiotemporal_iou_v2_{view}" for view in VIEWS
    ] + [
        f"variance_mse_{view}" for view in VIEWS
    ] + [
        f"variance_spatiotemporal_iou_{view}" for view in VIEWS
    ]

    # Parse list columns
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_of_floats)

    print(f"Loaded {len(df)} scenarios")
    return df


def calculate_per_scenario_metrics(df):
    """Calculate aggregated metrics for each scenario."""
    results = []

    for idx, row in df.iterrows():
        scenario_name = row['scenario']
        metrics = {'scenario': scenario_name}

        # --- MSE Metrics ---
        # Average MSE for v1 (across all views and frames)
        v1_mse_all = np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])
        metrics['avg_v1_mse'] = round(np.mean(v1_mse_all), 4)
        metrics['std_v1_mse'] = round(np.std(v1_mse_all), 4)

        # Average MSE for v2 (across all views and frames)
        v2_mse_all = np.concatenate([row[f"v2_mse_{view}"] for view in VIEWS])
        metrics['avg_v2_mse'] = round(np.mean(v2_mse_all), 4)
        metrics['std_v2_mse'] = round(np.std(v2_mse_all), 4)

        # Variance MSE (physical variance between v1 and v2)
        variance_mse_all = np.concatenate([row[f"variance_mse_{view}"] for view in VIEWS])
        metrics['avg_variance_mse'] = round(np.mean(variance_mse_all), 4)

        # --- Spatiotemporal IOU Metrics ---
        # Average spatiotemporal IOU for v1
        v1_iou_all = np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])
        metrics['avg_v1_spatiotemporal_iou'] = round(np.mean(v1_iou_all), 4)
        metrics['std_v1_spatiotemporal_iou'] = round(np.std(v1_iou_all), 4)

        # Average spatiotemporal IOU for v2
        v2_iou_all = np.concatenate([row[f"spatiotemporal_iou_v2_{view}"] for view in VIEWS])
        metrics['avg_v2_spatiotemporal_iou'] = round(np.mean(v2_iou_all), 4)
        metrics['std_v2_spatiotemporal_iou'] = round(np.std(v2_iou_all), 4)

        # Variance spatiotemporal IOU
        variance_iou_all = np.concatenate([row[f"variance_spatiotemporal_iou_{view}"] for view in VIEWS])
        metrics['avg_variance_spatiotemporal_iou'] = round(np.mean(variance_iou_all), 4)

        # --- Spatial IOU Metrics ---
        # Average spatial IOU for v1 (across all views)
        v1_spatial_iou = np.mean([row[f"spatial_iou_v1_{view}"] for view in VIEWS])
        metrics['avg_v1_spatial_iou'] = round(v1_spatial_iou, 4)

        # Average spatial IOU for v2 (across all views)
        v2_spatial_iou = np.mean([row[f"spatial_iou_v2_{view}"] for view in VIEWS])
        metrics['avg_v2_spatial_iou'] = round(v2_spatial_iou, 4)

        # Variance spatial IOU
        metrics['avg_variance_spatial'] = round(np.mean([row[f"variance_spatial_{view}"] for view in VIEWS]), 4)

        # --- Weighted Spatial IOU Metrics ---
        # Average weighted spatial IOU for v1
        v1_weighted_spatial_iou = np.mean([row[f"weighted_spatial_iou_v1_{view}"] for view in VIEWS])
        metrics['avg_v1_weighted_spatial_iou'] = round(v1_weighted_spatial_iou, 4)

        # Average weighted spatial IOU for v2
        v2_weighted_spatial_iou = np.mean([row[f"weighted_spatial_iou_v2_{view}"] for view in VIEWS])
        metrics['avg_v2_weighted_spatial_iou'] = round(v2_weighted_spatial_iou, 4)

        # Variance weighted spatial IOU
        metrics['avg_variance_weighted_spatial'] = round(np.mean([row[f"variance_weighted_spatial_{view}"] for view in VIEWS]), 4)

        # --- Compute Per-Scenario Score (similar to overall Physics-IQ) ---
        # Using v1 metrics (you can also compute for v2 or average them)
        avg_spatiotemporal_iou = metrics['avg_v1_spatiotemporal_iou']
        avg_spatial_iou = metrics['avg_v1_spatial_iou']
        avg_weighted_spatial_iou = metrics['avg_v1_weighted_spatial_iou']
        avg_mse = metrics['avg_v1_mse']

        variance_spatiotemporal = metrics['avg_variance_spatiotemporal_iou']
        variance_spatial = metrics['avg_variance_spatial']
        variance_weighted_spatial = metrics['avg_variance_weighted_spatial']
        variance_mse = metrics['avg_variance_mse']

        # Compute score (following the formula from calculate_iq_score.py)
        try:
            scenario_score = (
                (
                    (avg_spatiotemporal_iou / variance_spatiotemporal) +
                    (avg_spatial_iou / variance_spatial) +
                    (avg_weighted_spatial_iou / variance_weighted_spatial)
                ) / 3
            ) - (avg_mse - variance_mse)

            scenario_score *= 100
            scenario_score = round(max(min(scenario_score, 100.0), 0.0), 2)
        except (ZeroDivisionError, RuntimeWarning):
            scenario_score = 0.0

        metrics['scenario_physics_iq_score'] = scenario_score

        results.append(metrics)

    return pd.DataFrame(results)


def generate_summary_statistics(metrics_df):
    """Generate summary statistics across all scenarios."""
    summary = [
        ('Total Scenarios', len(metrics_df)),
        ('Average Physics-IQ Score', round(metrics_df['scenario_physics_iq_score'].mean(), 2)),
        ('Std Physics-IQ Score', round(metrics_df['scenario_physics_iq_score'].std(), 2)),
        ('Min Physics-IQ Score', round(metrics_df['scenario_physics_iq_score'].min(), 2)),
        ('Max Physics-IQ Score', round(metrics_df['scenario_physics_iq_score'].max(), 2)),
        ('Median Physics-IQ Score', round(metrics_df['scenario_physics_iq_score'].median(), 2)),
        ('---', '---'),  # Separator
        ('Average MSE (v1)', round(metrics_df['avg_v1_mse'].mean(), 4)),
        ('Average Spatiotemporal IOU (v1)', round(metrics_df['avg_v1_spatiotemporal_iou'].mean(), 4)),
        ('Average Spatial IOU (v1)', round(metrics_df['avg_v1_spatial_iou'].mean(), 4)),
        ('Average Weighted Spatial IOU (v1)', round(metrics_df['avg_v1_weighted_spatial_iou'].mean(), 4)),
    ]
    return summary


def plot_scenario_scores(metrics_df, output_path):
    """Create a bar plot of per-scenario Physics-IQ scores."""
    plt.figure(figsize=(16, 8))

    # Sort by score
    sorted_df = metrics_df.sort_values('scenario_physics_iq_score', ascending=True)

    colors = ['#d32f2f' if score < 30 else '#ff9800' if score < 60 else '#4caf50'
              for score in sorted_df['scenario_physics_iq_score']]

    plt.barh(range(len(sorted_df)), sorted_df['scenario_physics_iq_score'], color=colors)
    plt.yticks(range(len(sorted_df)), sorted_df['scenario'], fontsize=7)
    plt.xlabel('Physics-IQ Score', fontsize=12)
    plt.ylabel('Scenario', fontsize=12)
    plt.title('Per-Scenario Physics-IQ Scores', fontsize=14, fontweight='bold')
    plt.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scenario scores plot to: {output_path}")
    plt.close()


def plot_metric_distributions(metrics_df, output_path):
    """Create distribution plots for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE distribution
    axes[0, 0].hist(metrics_df['avg_v1_mse'], bins=30, color='#e57373', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Average MSE (v1)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].axvline(metrics_df['avg_v1_mse'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].legend()

    # Spatiotemporal IOU distribution
    axes[0, 1].hist(metrics_df['avg_v1_spatiotemporal_iou'], bins=30, color='#81c784', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Average Spatiotemporal IOU (v1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Spatiotemporal IOU Distribution')
    axes[0, 1].axvline(metrics_df['avg_v1_spatiotemporal_iou'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].legend()

    # Spatial IOU distribution
    axes[1, 0].hist(metrics_df['avg_v1_spatial_iou'], bins=30, color='#64b5f6', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Average Spatial IOU (v1)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Spatial IOU Distribution')
    axes[1, 0].axvline(metrics_df['avg_v1_spatial_iou'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].legend()

    # Physics-IQ Score distribution
    axes[1, 1].hist(metrics_df['scenario_physics_iq_score'], bins=30, color='#ba68c8', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Scenario Physics-IQ Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Physics-IQ Score Distribution')
    axes[1, 1].axvline(metrics_df['scenario_physics_iq_score'].mean(), color='purple', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metric distributions plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Physics-IQ benchmark results with detailed per-scenario metrics.")
    parser.add_argument("csv_file", type=str, help="Path to the Physics-IQ results CSV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for analysis results (default: same as CSV directory)")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv_file)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get model name from CSV filename
    model_name = os.path.splitext(os.path.basename(args.csv_file))[0]

    print(f"\n{'='*70}")
    print(f"Physics-IQ Results Analysis: {model_name}")
    print(f"{'='*70}\n")

    # Load and parse CSV
    df = load_and_parse_csv(args.csv_file)

    # Calculate per-scenario metrics
    print("\nCalculating per-scenario metrics...")
    metrics_df = calculate_per_scenario_metrics(df)

    # Save detailed metrics CSV
    output_csv = os.path.join(args.output_dir, f"{model_name}_detailed_metrics.csv")
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed metrics to: {output_csv}")

    # Generate summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    summary = generate_summary_statistics(metrics_df)
    for key, value in summary:
        if key == '---':
            print()
        else:
            print(f"{key:.<50} {value}")

    # Save summary to text file
    summary_file = os.path.join(args.output_dir, f"{model_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Physics-IQ Results Analysis: {model_name}\n")
        f.write("="*70 + "\n\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*70 + "\n")
        for key, value in summary:
            if key == '---':
                f.write("\n")
            else:
                f.write(f"{key:.<50} {value}\n")
    print(f"\nSaved summary to: {summary_file}")

    # Show top and bottom scenarios
    print("\n" + "="*70)
    print("TOP 10 SCENARIOS (Highest Physics-IQ Scores)")
    print("="*70)
    top_10 = metrics_df.nlargest(10, 'scenario_physics_iq_score')[['scenario', 'scenario_physics_iq_score']]
    for idx, row in top_10.iterrows():
        print(f"{row['scenario']:.<55} {row['scenario_physics_iq_score']:>6.2f}")

    print("\n" + "="*70)
    print("BOTTOM 10 SCENARIOS (Lowest Physics-IQ Scores)")
    print("="*70)
    bottom_10 = metrics_df.nsmallest(10, 'scenario_physics_iq_score')[['scenario', 'scenario_physics_iq_score']]
    for idx, row in bottom_10.iterrows():
        print(f"{row['scenario']:.<55} {row['scenario_physics_iq_score']:>6.2f}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    scenario_plot = os.path.join(args.output_dir, f"{model_name}_scenario_scores.png")
    plot_scenario_scores(metrics_df, scenario_plot)

    distribution_plot = os.path.join(args.output_dir, f"{model_name}_metric_distributions.png")
    plot_metric_distributions(metrics_df, distribution_plot)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {args.output_dir}")
    print(f"  - Detailed metrics CSV: {model_name}_detailed_metrics.csv")
    print(f"  - Summary statistics: {model_name}_summary.txt")
    print(f"  - Scenario scores plot: {model_name}_scenario_scores.png")
    print(f"  - Metric distributions: {model_name}_metric_distributions.png")
    print()


if __name__ == "__main__":
    main()
