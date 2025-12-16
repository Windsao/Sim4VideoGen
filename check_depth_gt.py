#!/usr/bin/env python
"""
Script to check depth ground truth data in the dataset.
Checks for file existence, format, value ranges, NaN/Inf values, and visualizes samples.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import glob


def find_depth_directories(base_dir: str, depth_subdir: str = "distance_to_camera") -> List[str]:
    """Find all directories containing depth data."""
    depth_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if depth_subdir in dirs:
            depth_path = os.path.join(root, depth_subdir)
            depth_dirs.append(depth_path)
    return sorted(depth_dirs)


def load_depth_file(filepath: str) -> Optional[np.ndarray]:
    """Load a single depth file."""
    try:
        depth = np.load(filepath)
        return depth
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def analyze_depth_file(depth: np.ndarray, filepath: str) -> Dict:
    """Analyze a single depth array and return statistics."""
    stats = {
        "filepath": filepath,
        "shape": depth.shape,
        "dtype": str(depth.dtype),
        "min": np.nanmin(depth) if not np.all(np.isnan(depth)) else np.nan,
        "max": np.nanmax(depth) if not np.all(np.isnan(depth)) else np.nan,
        "mean": np.nanmean(depth) if not np.all(np.isnan(depth)) else np.nan,
        "std": np.nanstd(depth) if not np.all(np.isnan(depth)) else np.nan,
        "median": np.nanmedian(depth) if not np.all(np.isnan(depth)) else np.nan,
        "nan_count": np.sum(np.isnan(depth)),
        "inf_count": np.sum(np.isinf(depth)),
        "zero_count": np.sum(depth == 0),
        "negative_count": np.sum(depth < 0),
        "total_pixels": depth.size,
    }
    stats["nan_ratio"] = stats["nan_count"] / stats["total_pixels"]
    stats["inf_ratio"] = stats["inf_count"] / stats["total_pixels"]
    stats["zero_ratio"] = stats["zero_count"] / stats["total_pixels"]
    return stats


def check_depth_directory(depth_dir: str, pattern: str = "*.npy", verbose: bool = True) -> Dict:
    """Check all depth files in a directory."""
    depth_files = sorted(glob.glob(os.path.join(depth_dir, pattern)))

    if not depth_files:
        return {
            "directory": depth_dir,
            "num_files": 0,
            "error": "No depth files found"
        }

    all_stats = []
    shapes = set()
    dtypes = set()

    for filepath in depth_files:
        depth = load_depth_file(filepath)
        if depth is not None:
            stats = analyze_depth_file(depth, filepath)
            all_stats.append(stats)
            shapes.add(stats["shape"])
            dtypes.add(stats["dtype"])

    if not all_stats:
        return {
            "directory": depth_dir,
            "num_files": len(depth_files),
            "error": "Failed to load any depth files"
        }

    # Aggregate statistics
    mins = [s["min"] for s in all_stats if not np.isnan(s["min"])]
    maxs = [s["max"] for s in all_stats if not np.isnan(s["max"])]
    means = [s["mean"] for s in all_stats if not np.isnan(s["mean"])]
    nan_ratios = [s["nan_ratio"] for s in all_stats]
    inf_ratios = [s["inf_ratio"] for s in all_stats]

    summary = {
        "directory": depth_dir,
        "num_files": len(depth_files),
        "num_loaded": len(all_stats),
        "shapes": list(shapes),
        "dtypes": list(dtypes),
        "consistent_shape": len(shapes) == 1,
        "consistent_dtype": len(dtypes) == 1,
        "global_min": min(mins) if mins else np.nan,
        "global_max": max(maxs) if maxs else np.nan,
        "avg_mean": np.mean(means) if means else np.nan,
        "avg_nan_ratio": np.mean(nan_ratios),
        "avg_inf_ratio": np.mean(inf_ratios),
        "files_with_nan": sum(1 for s in all_stats if s["nan_count"] > 0),
        "files_with_inf": sum(1 for s in all_stats if s["inf_count"] > 0),
        "files_with_negative": sum(1 for s in all_stats if s["negative_count"] > 0),
        "per_file_stats": all_stats if verbose else None,
    }

    return summary


def visualize_depth_samples(depth_dir: str, num_samples: int = 4,
                           pattern: str = "*.npy", output_path: Optional[str] = None,
                           cmap: str = "viridis"):
    """Visualize sample depth maps from a directory."""
    depth_files = sorted(glob.glob(os.path.join(depth_dir, pattern)))

    if not depth_files:
        print(f"No depth files found in {depth_dir}")
        return

    # Select evenly spaced samples
    indices = np.linspace(0, len(depth_files) - 1, min(num_samples, len(depth_files)), dtype=int)
    sample_files = [depth_files[i] for i in indices]

    fig, axes = plt.subplots(1, len(sample_files), figsize=(4 * len(sample_files), 4))
    if len(sample_files) == 1:
        axes = [axes]

    for ax, filepath in zip(axes, sample_files):
        depth = load_depth_file(filepath)
        if depth is not None:
            # Handle NaN/Inf for visualization
            depth_viz = depth.copy()
            depth_viz[np.isnan(depth_viz)] = 0
            depth_viz[np.isinf(depth_viz)] = np.nanmax(depth_viz[~np.isinf(depth_viz)])

            im = ax.imshow(depth_viz, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(os.path.basename(filepath), fontsize=8)
            ax.axis('off')

    plt.suptitle(f"Depth Samples from {os.path.basename(depth_dir)}", fontsize=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_depth_histogram(depth_dir: str, pattern: str = "*.npy",
                              output_path: Optional[str] = None,
                              num_bins: int = 100):
    """Plot histogram of depth values across all files in directory."""
    depth_files = sorted(glob.glob(os.path.join(depth_dir, pattern)))

    if not depth_files:
        print(f"No depth files found in {depth_dir}")
        return

    all_values = []
    for filepath in depth_files:
        depth = load_depth_file(filepath)
        if depth is not None:
            # Filter out NaN and Inf
            valid = depth[~np.isnan(depth) & ~np.isinf(depth)]
            all_values.append(valid.flatten())

    if not all_values:
        print("No valid depth values found")
        return

    all_values = np.concatenate(all_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale histogram
    axes[0].hist(all_values, bins=num_bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Depth Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Depth Distribution (Linear Scale)')
    axes[0].axvline(np.mean(all_values), color='r', linestyle='--', label=f'Mean: {np.mean(all_values):.2f}')
    axes[0].axvline(np.median(all_values), color='g', linestyle='--', label=f'Median: {np.median(all_values):.2f}')
    axes[0].legend()

    # Log scale histogram (for depth values > 0)
    positive_values = all_values[all_values > 0]
    if len(positive_values) > 0:
        axes[1].hist(np.log10(positive_values), bins=num_bins, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Log10(Depth Value)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Depth Distribution (Log Scale)')

    plt.suptitle(f"Depth Value Distribution - {os.path.basename(depth_dir)}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(summary: Dict):
    """Print a formatted summary of depth directory analysis."""
    print("\n" + "=" * 60)
    print(f"Directory: {summary['directory']}")
    print("=" * 60)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"Number of files: {summary['num_files']}")
    print(f"Successfully loaded: {summary['num_loaded']}")
    print(f"\nShapes: {summary['shapes']}")
    print(f"Shape consistent: {'Yes' if summary['consistent_shape'] else 'NO - INCONSISTENT!'}")
    print(f"Data types: {summary['dtypes']}")
    print(f"Dtype consistent: {'Yes' if summary['consistent_dtype'] else 'NO - INCONSISTENT!'}")
    print(f"\nValue Statistics:")
    print(f"  Global min: {summary['global_min']:.6f}")
    print(f"  Global max: {summary['global_max']:.6f}")
    print(f"  Average mean: {summary['avg_mean']:.6f}")
    print(f"\nData Quality:")
    print(f"  Files with NaN: {summary['files_with_nan']} ({100*summary['files_with_nan']/summary['num_loaded']:.1f}%)")
    print(f"  Files with Inf: {summary['files_with_inf']} ({100*summary['files_with_inf']/summary['num_loaded']:.1f}%)")
    print(f"  Files with negative values: {summary['files_with_negative']} ({100*summary['files_with_negative']/summary['num_loaded']:.1f}%)")
    print(f"  Average NaN ratio: {100*summary['avg_nan_ratio']:.4f}%")
    print(f"  Average Inf ratio: {100*summary['avg_inf_ratio']:.4f}%")


def main():
    parser = argparse.ArgumentParser(description="Check depth ground truth data")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing dataset")
    parser.add_argument("--depth_subdir", type=str, default="distance_to_camera",
                       help="Name of depth subdirectory")
    parser.add_argument("--pattern", type=str, default="*.npy",
                       help="File pattern to match depth files")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of sample frames to visualize")
    parser.add_argument("--cmap", type=str, default="viridis",
                       help="Colormap for depth visualization")
    parser.add_argument("--single_dir", type=str, default=None,
                       help="Check a single depth directory instead of searching")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-file statistics")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Find or use specified depth directories
    if args.single_dir:
        depth_dirs = [args.single_dir]
    else:
        print(f"Searching for depth directories in {args.base_dir}...")
        depth_dirs = find_depth_directories(args.base_dir, args.depth_subdir)

    if not depth_dirs:
        print(f"No depth directories found with subdirectory name '{args.depth_subdir}'")
        sys.exit(1)

    print(f"Found {len(depth_dirs)} depth directories")

    # Analyze each directory
    all_summaries = []
    for depth_dir in depth_dirs:
        summary = check_depth_directory(depth_dir, args.pattern, verbose=args.verbose)
        all_summaries.append(summary)
        print_summary(summary)

        if args.verbose and summary.get("per_file_stats"):
            print("\nPer-file statistics:")
            for stats in summary["per_file_stats"][:5]:  # Show first 5
                print(f"  {os.path.basename(stats['filepath'])}: "
                      f"shape={stats['shape']}, "
                      f"range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                      f"mean={stats['mean']:.2f}")
            if len(summary["per_file_stats"]) > 5:
                print(f"  ... and {len(summary['per_file_stats']) - 5} more files")

        if args.visualize:
            # Create visualizations
            safe_name = depth_dir.replace("/", "_").replace("\\", "_")

            if args.output_dir:
                sample_path = os.path.join(args.output_dir, f"depth_samples_{safe_name[-50:]}.png")
                hist_path = os.path.join(args.output_dir, f"depth_hist_{safe_name[-50:]}.png")
            else:
                sample_path = None
                hist_path = None

            visualize_depth_samples(depth_dir, args.num_samples, args.pattern,
                                   sample_path, args.cmap)
            visualize_depth_histogram(depth_dir, args.pattern, hist_path)

    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total directories checked: {len(all_summaries)}")

    valid_summaries = [s for s in all_summaries if "error" not in s]
    if valid_summaries:
        total_files = sum(s["num_files"] for s in valid_summaries)
        total_loaded = sum(s["num_loaded"] for s in valid_summaries)
        dirs_with_nan = sum(1 for s in valid_summaries if s["files_with_nan"] > 0)
        dirs_with_inf = sum(1 for s in valid_summaries if s["files_with_inf"] > 0)
        dirs_with_neg = sum(1 for s in valid_summaries if s["files_with_negative"] > 0)

        all_shapes = set()
        for s in valid_summaries:
            all_shapes.update(s["shapes"])

        print(f"Total depth files: {total_files}")
        print(f"Successfully loaded: {total_loaded}")
        print(f"Unique shapes across all dirs: {all_shapes}")
        print(f"Directories with NaN values: {dirs_with_nan}")
        print(f"Directories with Inf values: {dirs_with_inf}")
        print(f"Directories with negative values: {dirs_with_neg}")

        global_min = min(s["global_min"] for s in valid_summaries if not np.isnan(s["global_min"]))
        global_max = max(s["global_max"] for s in valid_summaries if not np.isnan(s["global_max"]))
        print(f"Global depth range: [{global_min:.6f}, {global_max:.6f}]")

    error_summaries = [s for s in all_summaries if "error" in s]
    if error_summaries:
        print(f"\nDirectories with errors: {len(error_summaries)}")
        for s in error_summaries:
            print(f"  {s['directory']}: {s['error']}")


if __name__ == "__main__":
    main()
