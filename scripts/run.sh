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

export WANDB_USER=irina-rish
python thinker/train.py --drc true \
                --xpid drc_gamma098_check \
                --tran_t 1 \
                --actor_unroll_len 20 \
                --reg_cost 0.01 \
                --actor_learning_rate 4e-4 \
                --entropy_cost 1e-2 \
                --v_trace_lamb 0.97 \
                --actor_adam_eps 1e-4 \
                --has_model false \
                --discounting 0.98 \
                --use_wandb true


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

