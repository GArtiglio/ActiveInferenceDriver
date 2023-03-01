#! /bin/bash
python test.py \
--exp_name "02-16-2023 15-47-03"\
--max_steps 130 \
--save True \
--seed 0

# batch test exp names
exp_names=(
    # "02-14-2023 22-23-43"
    # "02-14-2023 23-26-46"
    # "02-15-2023 00-29-27"
    # "02-15-2023 01-33-57"
    # "02-15-2023 02-36-35"
    # "02-15-2023 03-39-20"
    # "02-15-2023 04-39-34"
    # "02-15-2023 05-34-15"
    # "02-15-2023 06-28-38"
    # "02-15-2023 07-23-06"
    # "02-15-2023 08-17-52"
    # "02-15-2023 09-13-30"
    "02-16-2023 15-47-03"
)

for exp_name in "${exp_names[@]}";
    do
    echo $exp_name
    python test.py \
    --exp_name "$exp_name"\
    --max_steps 130 \
    --save True \
    --seed 0
    done