"""
Script for collecting and summarizing statistics recorded from the exp_run.py script.
Any results existing in the outputs folder will be gathered and summarized, as long
as they exist in the right arrangement (i.e., that expected after running exp_run.py
scripts in appropriate settings).
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import csv
import uuid
import logging
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import tqdm
import hydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
TAB = "\t"


OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../python/itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dirs_with_results = []
    outputs_root_dir = os.path.join(cfg.paths.root_dir, "outputs")
    for group_dir in os.listdir(outputs_root_dir):
        if not os.path.isdir(os.path.join(outputs_root_dir, group_dir)): continue
        if group_dir.startswith("_"): continue      # Ignore groups prefixed with "_"

        for run_dir in os.listdir(os.path.join(outputs_root_dir, group_dir)):
            full_out_dir = os.path.join(outputs_root_dir, group_dir, run_dir)

            if not os.path.isdir(full_out_dir):
                continue
            
            if "results" in os.listdir(full_out_dir):
                dirs_with_results.append((group_dir, run_dir))

    results = defaultdict(dict)
    collected_configs = defaultdict(set)

    # Collect results
    for res_dir in tqdm.tqdm(dirs_with_results, total=len(dirs_with_results)):
        group_dir, run_dir = res_dir
        res_dir = os.path.join(outputs_root_dir, *res_dir, "results")
        dir_contents = list(os.listdir(res_dir))

        for data in dir_contents:
            name_parse = data.split(".")[0].split("_")
            task, player_type, seed = name_parse

            collected_configs[seed].add(player_type)

            # Collect data
            with open(os.path.join(res_dir, data)) as data_f:
                reader = csv.reader(data_f)

                # Header column, containing field names:
                # 1) episode
                # 2) num_search_failure
                # 3) num_invalid_pickup
                # 4) num_invalid_join
                # 5) num_planning_forfeiture
                # 6) num_distractor_pickup
                # 7) episode_discarded
                # 8) num_planning_attempts
                # 9) num_collision_queries
                # 10) episode_length
                # 11) mean_f1
                fields = next(reader)
                data_of_interests = [
                    ([1, 2, 3, 4, 5], True),
                    ([1], False), ([2], False), ([3], False), ([4], False), ([5], False),
                    ([7], False), ([8], True), ([9], False), ([10], False)
                ]
                data_names = ["cumulative_regret"] + [
                    fields[i] for i in [1, 2, 3, 4, 5, 7, 8, 9, 10]
                ]

                row = [row_data for row_data in reader]
                for (col_inds, cumsum), d_name in zip(data_of_interests, data_names):
                    curve = np.array([
                        [int(entry[0]), sum(float(entry[i]) for i in col_inds)]
                        for entry in row
                    ])
                    if cumsum:
                        curve[:,1] = np.cumsum(curve[:,1])
                        curve = np.concatenate([[[0, 0]], curve])

                    if player_type in results[d_name]:
                        stats_agg = results[d_name][player_type]
                        for ep_num, val in curve:
                            if ep_num in stats_agg:
                                stats_agg[ep_num].append(val)
                            else:
                                stats_agg[ep_num] = [val]
                    else:
                        results[d_name][player_type] = {
                            ep_num: [val] for ep_num, val in curve
                        }

    all_player_types = set.union(*collected_configs.values())
    for seed, player_types in collected_configs.items():
        if len(player_types) < 4:
            missing = all_player_types - player_types
            missing = ",".join(missing)
            logger.info(f"Missing data from: seed={seed}, player_type=[{missing}]")

    # Pre-defined ordering for listing legends
    config_ord = ["bool", "demo", "label", "full"]
    config_aliases = {
        "bool": "Languageless-Minimal",
        "demo": "Languageless-Helpful",
        "label": "Languageful-Labeling",
        "full": "Languageful-Semantic"
    }   # To be actually displayed in legend
    config_colors = {
        "bool": "tab:red",
        "demo": "tab:orange",
        "label": "tab:green",
        "full": "tab:blue"
    }
    data_titles = {
        "cumulative_regret": "Cumulative regrets",
        "num_search_failure": "# Search Failure",
        "num_invalid_pickup": "# Pickups of Incompatible Instance Pairs",
        "num_invalid_join": "# Joins at Incorrect Pose",
        "num_planning_forfeiture": "# Planning Forfeitures",
        "num_distractor_pickup": "# Pickups of Distractor Instances",
        "episode_discarded": "# Aborted Episodes",
        "num_planning_attempts": "# Planning Attempts",
        "num_collision_queries": "# Queries to Collision Checker",
        "episode_length": "Episode Lengths",
        "mean_f1": "Mean F1 Scores"
    }

    # Aggregate and visualize: cumulative regret curve
    for d_name, collected_data in results.items():
        _, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ymax = 0

        for player_type, data in collected_data.items():
            stats = [
                (i, np.mean(rgs), 1.96 * np.std(rgs)/np.sqrt(len(rgs)))
                for i, rgs in data.items()
            ]
            ymax = max(ymax, max(mrg+cl for _, mrg, cl in stats))

            # Plot mean curve
            ax.plot(
                [i for i, _, _ in stats],
                [mrg for _, mrg, _ in stats],
                label=player_type,
                color=config_colors[player_type]
            )
            # Plot confidence intervals
            ax.fill_between(
                [i for i, _, _ in stats],
                [mrg-cl for _, mrg, cl in stats],
                [mrg+cl for _, mrg, cl in stats],
                color=config_colors[player_type], alpha=0.2
            )

        # Plot curve
        ax.set_xlabel("# training episodes")
        ax.set_xticks([10, 20, 30, 40])
        yrange = (0, 1) if d_name == "mean_f1" else (0, ymax * 1.1)
        ax.set_xlim(0, 40)
        ax.set_ylim(yrange[0], yrange[1])
        ax.grid()
        if task == "subtype":
            ax.set_xticks([5, 10, 20, 30, 40])
            ax.vlines(
                5, yrange[0], yrange[1],
                color="grey", linestyle="--", linewidth=1.1
            )

        # Ordering legends according to the prespecified ordering above
        handles, labels = ax.get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [
            hl[0] for hl in hls_sorted
        ] + [mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.1)]
        labels = [
            config_aliases.get(hl[1], hl[1]) for hl in hls_sorted
        ] + ["Novel subtypes introduced"]
        ax.legend(handles, labels)

        ax.set_title(f"{data_titles[d_name]} ({len(collected_configs)} datasets)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"{d_name}.png"), bbox_inches="tight")

    # print("")
    # print(f"Endpoint cumulative regret CIs:")
    # # Report final values on stdout
    #         print(f"{TAB}{player_type}: {stats[-1][1]:.2f} \xB1 {stats[-1][2]:.2f}")

if __name__ == "__main__":
    main()
