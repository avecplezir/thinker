salloc -c 4 --gres=gpu:1 --mem=48G --time=7:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --time=1:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --time=4:00:00
salloc -c 4  --mem=15G --time=4:00:00
salloc -c 4  --mem=48G
source activate /home/mila/i/ivan.anokhin/anaconda3/envs/alm310

salloc --partition=short-unkillable  --cpus-per-task=4 --ntasks-per-node=1 --mem=15G -n 1 --time=03:00:00 --gpus-per-task=h100:1

salloc -c 4 --gres=gpu:h100:1 --mem=15G --time=4:00:00 --partition=short-unkillable

salloc -c 4 --gres=gpu:rtx8000:1 --mem=15G --time=4:00:00

module load cudatoolkit/12.6.0
salloc -c 4 --gres=gpu:a100:1 --mem=15G --time=4:00:00 --partition=main
module load cudatoolkit/12.1.1
source activate /home/mila/i/ivan.anokhin/anaconda3/envs/openrlhf

MAX_JOBS=4 pip install flash-attn==2.7.0.post2 --no-cache-dir --no-build-isolation


salloc -c 4 --gres=gpu:a100:1 --mem=15G --time=2:00:00 --partition=long

salloc -c 16 --gres=gpu:a100:1 --mem=40G --time=4:00:00 --partition=main


jupyter notebook --no-browser --ip=* --port=8081
ssh -L 8082:cn-c007:8081 -fN mila

python sac_wm.py --exp_name sac --learning_starts 0 --frame_skip 1 --num_envs 1 --batch_size 256 --backward None --N_hidden_layers 1 --agent default --env_id HalfCheetah-v4

python sac_wm.py --exp_name sac --learning_starts 0 \
                 --frame_skip 1 --num_envs 1 --batch_size 256 --backward None \
                 --N_hidden_layers 1 --agent default --env_id HalfCheetah-v4

python train.py --env_id HalfCheetah-v4 --exp_name slowagent --trainer delayed_action --agent slow --num_envs 8 --policy_frequency 3 --learning_starts 0  --track

python train.py --env_id HalfCheetah-v4 --exp_name slowskip --trainer delayed_action --agent slowskip --num_envs 8 --policy_frequency 3 --learning_starts 0  --track

salloc -c 4 --gres=gpu:1 --mem=15G --account=rrg-ebrahimi --time=2:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --account=def-ebrahimi --time=2:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --account=def-ebrahimi --time=7:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --account=def-ebrahimi --time=7:00:00
salloc -c 4 --gres=gpu:1 --mem=15G --account=rrg-ebrahimi --time=1:00:00
salloc -c 4 --gres=gpu:2 --mem=15G --account=rrg-ebrahimi --time=7:00:00
salloc -c 4  --mem=15G --account=rrg-ebrahimi --time=24:00:00
cd /home/anokhin/projects/rrg-ebrahimi/anokhin
cd /home/anokhin/projects/def-ebrahimi/anokhin/slow_agent
source /home/anokhin/projects/rrg-ebrahimi/anokhin/envs/rl/bin/activate
source /home/anokhin/projects/rrg-ebrahimi/anokhin/envs/rbhm/bin/activate
source /home/anokhin/projects/def-ebrahimi/anokhin/envs/slowagent/bin/activate
source /home/anokhin/projects/def-ebrahimi/anokhin/envs/mujoco/bin/activate
git@github.com:avecplezir/slow_agent.git
module load httpproxy

gym==0.26.2+computecanada
gymnasium==0.29.1
numpy==1.23.0+computecanada
stable_baselines3==2.3.0
tyro==0.7.3
