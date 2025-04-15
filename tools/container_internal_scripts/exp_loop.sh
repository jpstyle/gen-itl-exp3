#!/bin/bash
export PYTHONHASHSEED=42
player_types=( bool demo label full )

for ((seed=$1;seed<=$2;seed++))
do

    for pl_type in "${player_types[@]}"
    do
        if [ "$3" == build_truck_supertype ]
        then
            # Base-level experiment of building generic truck supertype
            python ~/semantic-assembler/tools/exp_run.py exp.task=$3 exp.player_type=${pl_type} vision.calibrate_camera=false seed=$seed
        else
            # Advanced-level experiment of building truck subtypes with
            # novel part subtypes introduced
            python ~/semantic-assembler/tools/exp_run.py +exp.agent_model_path=/home/jayp/git/semantic-assembler/assets/agent_models/color_pretrained.ckpt exp.task=$3 exp.player_type=${pl_type} vision.calibrate_camera=false seed=$seed
            # Comment above and uncomment below before pushing to docker hub
            # python ~/semantic-assembler/tools/exp_run.py +exp.agent_model_path=/home/nonroot/semantic-assembler/assets/agent_models/color_pretrained.ckpt exp.task=$3 exp.player_type=${pl_type} vision.calibrate_camera=false seed=$seed
        fi
    done
done