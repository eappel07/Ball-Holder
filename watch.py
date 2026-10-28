from agent import Agent
from BallEnv import BallEnv
import numpy as np
import time

if __name__ == "__main__": 

    env = BallEnv(render=True)

    agent = Agent(alr=1e-3, clr=1e-3, tau=0.005, input_dims=(10,), env=env)
    agent.load_models()

    for _ in range(3):

        obs = env.reset()
        trial_done = False 

        while not trial_done: 

            action = agent.choose_action(obs, validation=True)
            
            new_obs, _, terminate, truncate = env.step(action)
            
            obs = new_obs
            trial_done = bool(truncate or terminate)

    env.close()





