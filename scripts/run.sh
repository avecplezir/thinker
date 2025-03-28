#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=None
#SBATCH --mail-type=FAIL
#SBATCH --job-name=thinker
#SBATCH --no-requeue

seeds=${SLURM_ARRAY_TASK_ID}

source activate /home/mila/i/ivan.anokhin/anaconda3/envs/thinker


#python thinker/train.py --xpid thinkerv2 --model_warm_up_n 500 --rec_t 10 --imagination_loss true --use_wandb false

#"thinker/train.py --xpid thinker_r10_decdepth4_v4
#--model_warm_up_n 500 --detach_features false --img_fea_cos false
#--model_decoder_depth 4 --imagination_loss true --use_dones_im true --rec_t 10 --use_wandb true"

#model_policy_loss_cost: 0.5 # cost for training model's policy
#model_vs_loss_cost: 0.25 # cost for training model's values
#model_rs_loss_cost: 1.0 # cost for training model's reward

export WANDB_USER=irina-rish
python thinker/train.py \
              --xpid r10_imcost5_decd4__v4 \
              --model_warm_up_n 10000 \
              --im_separate_head false \
              --im_loss_cost 5 \
              --priority_alpha 0 \
              --vp_loss false \
              --num_im_iterations 6 \
              --model_policy_loss_cost 0. \
              --model_vs_loss_cost 0. \
              --detach_features false \
              --img_fea_cos false \
              --model_decoder_depth 4 \
              --imagination_loss true \
              --rec_t 10 \
              --use_wandb true

#python thinker/train.py \
#              --xpid r10_imcost5_decd4_v2 \
#              --model_warm_up_n 10000 \
#              --im_separate_head false \
#              --im_loss_cost 5 \
#              --vp_loss false \
#              --num_im_iterations 6 \
#              --model_policy_loss_cost 0. \
#              --model_vs_loss_cost 0. \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid r10_imcost4_decd4_vs01pl01_v2 \
#              --model_warm_up_n 10000 \
#              --im_loss_cost 5 \
#              --vp_loss true \
#              --num_im_iterations 6 \
#              --model_policy_loss_cost 0.1 \
#              --model_vs_loss_cost 0.1 \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid r10_im_v7 \
#              --model_warm_up_n 500 \
#              --im_separate_head true \
#              --im_loss_cost 5 \
#              --num_im_iterations 6 \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid r10_v1 \
#              --model_warm_up_n 500 \
#              --im_separate_head false \
#              --im_loss_cost 1 \
#              --num_im_iterations 6 \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss false \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid r10_imcost4it6_decd4_vs005pl005_v1 \
#              --model_warm_up_n 10000 \
#              --im_loss_cost 1 \
#              --vp_loss true \
#              --num_im_iterations 6 \
#              --model_policy_loss_cost 0.05 \
#              --model_vs_loss_cost 0.05 \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid thinker_r10_im_decdepth4_vsdistill01_v6 \
#              --model_warm_up_n 10000 \
#              --vp_loss true \
#              --model_policy_loss_cost 0. \
#              --model_vs_loss_cost 0.1 \
#              --priority_alpha 0 \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --use_dones_im true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid thinker_r10_im_decdepth4_nocos_v5 \
#              --model_warm_up_n 10000 \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 4 \
#              --imagination_loss true \
#              --use_dones_im true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid thinker_r10_decdepth4_noim_nocos_v4 \
#              --model_warm_up_n 10000 \
#              --detach_features false \
#              --img_fea_cos false \
#              --model_decoder_depth 3 \
#              --imagination_loss false \
#              --use_dones_im true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid thinker_r10_imloss_v5 \
#              --model_warm_up_n 500 \
#              --model_img_loss_cost 0 \
#              --model_fea_loss_cost 10 \
#              --detach_features true \
#              --img_fea_cos false \
#              --imagination_loss true \
#              --use_dones_im true \
#              --rec_t 10 \
#              --use_wandb true




#python thinker/train.py \
#              --xpid thinker_r10_dones_detachsr_v4 \
#              --model_warm_up_n 500 \
#              --detach_features false \
#              --imagination_loss true \
#              --use_dones_im true \
#              --rec_t 10 \
#              --use_wandb true

#python thinker/train.py \
#              --xpid thinker_onegpu_v4 \
#              --imagination_loss false \
#              --rec_t 10 \
#              --use_wandb true



#python thinker/train.py --xpid thinker_r10_v3 --model_warm_up_n 10000 --imagination_loss false --rec_t 10 --use_wandb true

#python thinker/train.py --drc true \
#                --xpid drc_gamma098_check \
#                --tran_t 1 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.98 \
#                --use_wandb true


#python thinker/train.py --drc true \
#                --xpid drc_gamma098_trant1 \
#                --tran_t 1 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.98 \
#                --use_wandb true

#python thinker/train.py --drc true \
#                --xpid drc_gamma098_lambda098_trant1 \
#                --tran_t 1 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.98 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.98 \
#                --use_wandb true

#python thinker/train.py --drc true \
#                --xpid drc_gamma098_trant1_hd64 \
#                --tran_t 1 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.98 \
#                --use_wandb true

#python thinker/train.py --drc true \
#                --xpid drc_gamma097_trant1 \
#                --tran_t 1 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.97 \
#                --use_wandb true

#python thinker/train.py --drc true \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --use_wandb true


#python thinker/train.py --drc true \
#                --xpid drc_gamma099 \
#                --actor_unroll_len 20 \
#                --reg_cost 0.01 \
#                --actor_learning_rate 4e-4 \
#                --entropy_cost 1e-2 \
#                --v_trace_lamb 0.97 \
#                --actor_adam_eps 1e-4 \
#                --has_model false \
#                --discounting 0.99 \
#                --use_wandb true
