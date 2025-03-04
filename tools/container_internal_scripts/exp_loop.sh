#!/bin/bash
export PYTHONHASHSEED=42
player_types=( bool demo label full )

for ((seed=$1;seed<=$2;seed++))
do

    for pl_type in "${player_types[@]}"
    do
        python ~/semantic-assembler/tools/exp_run.py paths.assets_dir=$3 exp.task=$4 exp.player_type=${pl_type} vision.calibrate_camera=false seed=$seed
    done
done