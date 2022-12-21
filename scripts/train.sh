#! /bin/bash
for seed in 0;
    do
        for prior_penalty in 0.001;
            do 
                python train.py \
                --state_dim 2 \
                --horizon 35 \
                --obs_dist norm \
                --a_cum False \
                --prior_cov full \
                --bc_penalty 0.5 \
                --obs_penalty 1. \
                --prior_penalty $prior_penalty \
                --cp_path "none" \
                --t_add 3 \
                --lr 0.01 \
                --decay 3e-4 \
                --epochs 100 \
                --cp_every 100 \
                --verbose 1 \
                --save True \
                --seed $seed
            done
    done