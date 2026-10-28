import time
import os
import numpy as np
import mujoco
from torch.utils.tensorboard import SummaryWriter
from BallEnv import BallEnv
from agent import Agent


if __name__ == "__main__":

    if not os.path.exists("td3/ball_holder"):
        os.makedirs("td3/ball_holder")

    # one log folder per training run, named by date + time
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    env = BallEnv(render=False)
    agent = Agent(alr=1e-3, clr=1e-3, tau=0.005, input_dims=(10,), env=env, warmup=0)
    agent.load_models()
    score_history = []
    best_score = -np.inf
    total_steps = 0

    for episode in range(500):
        # reset every episode
        obs = env.reset()
        reward = 0
        trial_done = False

        # running totals of each reward piece for this episode
        parts_total = {"alive": 0.0, "centre": 0.0, "jitter": 0.0}

        while not trial_done:
            action = agent.choose_action(obs)
            new_obs, reward_step , terminate, truncate = env.step(action)

            agent.remember(obs, action, reward_step, new_obs, terminate)

            learn_before = agent.learn_cnt
            agent.learn()
            total_steps += 1

            # only log losses on steps where learning actually happened
            if agent.learn_cnt > learn_before:
                writer.add_scalar("loss/critic", agent.critic_loss, agent.learn_cnt)
                writer.add_scalar("q/mean_q", agent.q_mean, agent.learn_cnt)
                if agent.learn_cnt % agent.updat_actor_interval == 0:
                    writer.add_scalar("loss/actor", agent.actor_loss, agent.learn_cnt)

            for name, value in env.reward_parts.items():
                parts_total[name] += value

            reward += reward_step
            obs = new_obs

            trial_done = bool(terminate or truncate)

        score_history.append(reward)

        # avg score to avoid noise
        avg_score = np.mean(score_history[-50:])

        # per-episode graphs
        writer.add_scalar("episode/score", reward, episode)
        writer.add_scalar("episode/avg_score_50", avg_score, episode)
        writer.add_scalar("episode/length_steps", env.steps, episode)
        writer.add_scalar("episode/fell_off", int(terminate), episode)
        for name, value in parts_total.items():
            writer.add_scalar(f"reward_parts/{name}", value, episode)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"Episode {episode}: reward = {reward:.2f}, avg = {avg_score:.2f}, "
              f"steps = {env.steps}, ended: {'Terminated' if terminate else 'Time'}")

    writer.close()
    env.close()
