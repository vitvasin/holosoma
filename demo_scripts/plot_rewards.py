import argparse
import os
import warnings
# Suppress Matplotlib Axes3D warning
warnings.filterwarnings("ignore", message=".*Unable to import Axes3D.*")

import matplotlib.pyplot as plt
from packaging import version
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

def main():
    parser = argparse.ArgumentParser(description="Plot key Training Metrics from TensorBoard logs")
    parser.add_argument("logdir", type=str, help="Path to run directory or tfevents file (e.g. logs/WholeBodyTracking/<run-dir>)")
    parser.add_argument("--save", type=str, default="training_metrics.png", help="Path to save the plot")
    args = parser.parse_args()

    print(f"Loading tensorboard events from: {args.logdir} ...\n")
    ea = event_accumulator.EventAccumulator(args.logdir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print("No scalar data found in these logs.")
        return

    # 4 metrics we want to visualize
    metrics_to_plot = {
        "Mean Reward": [t for t in tags if 'mean_reward' in t.lower() or t.lower() == 'reward'],
        "Episode Length": [t for t in tags if 'average_episode_length' in t.lower()],
        "Joint Pos Error (rad)": [t for t in tags if 'error_joint_pos' in t.lower()],
        "Curriculum Entropy": [t for t in tags if 'adaptive_timesteps_sampler_entropy' in t.lower()]
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()

    data_found = False
    
    for i, (title, possible_tags) in enumerate(metrics_to_plot.items()):
        ax = axs[i]
        tag_to_use = possible_tags[0] if possible_tags else None
        
        if tag_to_use:
            events = ea.Scalars(tag_to_use)
            iterations = [e.step for e in events]
            values = [e.value for e in events]
            
            if iterations:
                data_found = True
                ax.plot(iterations, values, marker='o', linestyle='-', markersize=1.5, alpha=0.8, color=f"C{i}")
                ax.set_title(title)
                ax.set_xlabel('Learning Iterations')
                ax.set_ylabel('Value')
                ax.grid(True, linestyle='--', alpha=0.6)
                print(f"Found {title} -> {tag_to_use} ({len(iterations)} points)")
        else:
            ax.set_title(f"{title} (Data offline/not found)")
            ax.axis('off')

    if not data_found:
        print("\nCould not extract any of the requested metrics.")
        return

    plt.tight_layout()
    plt.savefig(args.save, dpi=120)
    print(f"\nPlot saved successfully to {args.save}")

if __name__ == "__main__":
    main()
