import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering; use "osmesa" if egl is missing

import argparse
import numpy as np
import mujoco
import imageio
from agent import Agent
from BallEnv import BallEnv

# records the trained policy balancing the ball to an mp4 (no window needed)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="media/ball_holder_sim.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = BallEnv(render=False, seed=args.seed)
    agent = Agent(alr=1e-3, clr=1e-3, tau=0.005, input_dims=(10,), env=env)
    agent.load_models()

    renderer = mujoco.Renderer(env.model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.08]
    cam.distance = 0.45
    cam.elevation = -30
    cam.azimuth = 135

    # each env step is 20 sim steps of 1 ms -> 50 frames per second of sim time
    fps = int(round(1 / (20 * env.model.opt.timestep)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8)

    for ep in range(args.episodes):
        obs = env.reset()
        trial_done = False
        while not trial_done:
            action = agent.choose_action(obs, validation=True)
            obs, _, terminate, truncate = env.step(action)
            cam.azimuth += 0.2      # slow orbit
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())
            trial_done = bool(terminate or truncate)
        print(f"Episode {ep}: {env.steps} steps, ended: {'fell off' if terminate else 'time limit'}")

    writer.close()
    env.close()
    print("saved", args.out)
