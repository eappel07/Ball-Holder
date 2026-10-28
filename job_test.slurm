#!/bin/bash
# Quick check that everything works on a compute node (~1 min, uses the fast test queue)
#   sbatch job_test.slurm
#SBATCH --job-name=ball-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

module purge
module load anaconda3/2026.7
conda activate ball-env

cd /scratch/network/$USER/ball_holder

python -u -c "
from BallEnv import BallEnv
from agent import Agent
env = BallEnv(seed=0)
agent = Agent(alr=1e-3, clr=1e-3, tau=0.005, input_dims=(10,), env=env)
obs = env.reset()
for i in range(1500):
    action = agent.choose_action(obs)
    new_obs, reward, terminate, truncate = env.step(action)
    agent.remember(obs, action, reward, new_obs, terminate)
    agent.learn()
    obs = env.reset() if (terminate or truncate) else new_obs
print('TEST OK: steps', agent.time_step, 'learn updates', agent.learn_cnt, 'device', agent.actor.device)
"
