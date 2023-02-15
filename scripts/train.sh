#! /bin/bash
python train.py \
--state_dim 2 \
--horizon 30 \
--prior_cov full \
--bc_penalty 0. \
--obs_penalty 0. \
--kl_penalty 0.005 \
--cp_path "none" \
--lr 0.01 \
--decay 1e-5 \
--cp_path "none" \
--epochs 15000 \
--cp_every 500 \
--verbose 10 \
--save True \
--seed 0

# grid search
# for bc_penalty in 1. 0.5 0.001;
#     do 
#     for obs_penalty in 4. 1. 0.5 0.001;
#         do 
#             python train.py \
#             --state_dim 2 \
#             --horizon 30 \
#             --prior_cov full \
#             --bc_penalty $bc_penalty \
#             --obs_penalty $obs_penalty \
#             --kl_penalty 0.007 \
#             --cp_path "none" \
#             --lr 0.01 \
#             --decay 1e-5 \
#             --cp_path "none" \
#             --epochs 15000 \
#             --cp_every 500 \
#             --verbose 10 \
#             --save True \
#             --seed 0
#         done
#     done
