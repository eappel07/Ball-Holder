import time
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces

class BallEnv:

    def __init__(self, render=False, seed=None):
        self.model = mujoco.MjModel.from_xml_path("ball_balancer.xml")
        self.data = mujoco.MjData(self.model)
        self.rand = np.random.default_rng(seed)
        self.max_tilt = np.radians(8)      
        self.max_steps = 250  
        self.max_kick = 0.05
        self.steps = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._set_up()

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # intiallizes all values needed for joint location + rotation
    def _set_up(self):

        m = self.model
        d = self.data

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self.plate_center = np.array([0, 0, 0.1021])
        self.plate_radius = m.geom("glass").size[0]

        # legs array with dict lookup for each leg
        self.legs = []
        for s in "abc": 

            motor_joint, elbow_joint = m.joint(f"motor_{s}"), m.joint(f"elbow_{s}")
            motor_pos = d.xanchor[motor_joint.id].copy()
            elbow_pos = d.xanchor[elbow_joint.id].copy()
            motor_axis = d.xaxis[motor_joint.id].copy()

            ball_pos = m.equality(f"balljoint_{s}").data[0:3].copy()

            upper_arm_l = np.linalg.norm(elbow_pos - motor_pos)
            forearm_l = np.linalg.norm(ball_pos - elbow_pos)

            # reference joint for when plate is flat + 2d representation of unit vectors
            ref = d.joint(f"motor_{s}").qpos[0]
            plane_x = (elbow_pos - motor_pos) / upper_arm_l
            plane_y = np.cross(motor_axis, plane_x)

            self.legs.append(dict(motor_pos=motor_pos, elbow_pos=elbow_pos,
                                forearm_l=forearm_l, ball_local_pos=ball_pos, 
                                ref=ref, plane_x=plane_x, plane_y=plane_y, 
                                upper_arm_l=upper_arm_l))

        # establish position and velocition
        self.ball_q = m.jnt_qposadr[m.joint("ball").id]
        self.ball_v = m.jnt_dofadr[m.joint("ball").id]

    def set_servo_angles(self, x_roll, y_pitch, dz): 

        # Rotation logic
        Rotation_x = np.array([[1, 0, 0], 
                              [0, np.cos(x_roll), -np.sin(x_roll)], 
                              [0, np.sin(x_roll), np.cos(x_roll)]
                              ])
        Rotation_y = np.array([[np.cos(y_pitch), 0, np.sin(y_pitch)], 
                              [0, 1, 0], 
                              [-np.sin(y_pitch), 0, np.cos(y_pitch)]
                              ])
        Rotation = Rotation_y @ Rotation_x

        servo_angles = []
        for leg in self.legs: 
            offset = leg["ball_local_pos"] - self.plate_center

            # new point location
            target_location = self.plate_center + Rotation @ offset + np.array([0, 0, dz])

            # create chadow onto plane of arm
            rel = target_location - leg["motor_pos"]
            x = rel @ leg["plane_x"]
            y = rel @ leg["plane_y"]

            # find hypotnuse + use it in law of cosines
            dist = np.hypot(x, y)
            ul = leg["upper_arm_l"]
            fl = leg["forearm_l"]

            angle = np.clip((ul**2 + dist**2 - fl**2)/(2 * ul * dist), -1, 1)
            angle = np.arccos(angle)
            
            # find angle to move by taking angle across from forearm and adding to direct vector
            angle_to_target = np.arctan2(y, x)

            option_1 = angle_to_target + angle
            option_2 = angle_to_target - angle
            angle = option_1 if abs(option_1) < abs(option_2) else option_2

            servo_angles.append(leg["ref"] + angle)

        return np.clip(servo_angles, 0, 2.0944) 


    def reset(self, max_tilt_degrees=8, min_plate_h=-10, max_plate_h=10, radius_ball_pot_mm=50):
        mujoco.mj_resetData(self.model, self.data)


        x_roll, y_pitch = np.radians(self.rand.uniform(-max_tilt_degrees, max_tilt_degrees, size=2))
        p_height = self.rand.uniform(min_plate_h, max_plate_h) / 1000     # mm -> m, change from flat height

        self.data.ctrl[:] = self.set_servo_angles(x_roll, y_pitch, p_height)
        self.data.qpos[self.ball_q: self.ball_q + 3] = [0.5, 0, .02]

        for i in range(300): 
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0

        # where plate ended
        plate_rot = self.plate_rot()
        plate_top = self.data.site_xpos[self.model.site("platform_center").id]

        # use polar cords to get ball placement
        rand_radius = radius_ball_pot_mm / 1000 * np.sqrt(self.rand.uniform())
        rand_angle = self.rand.uniform(0, 2 * np.pi)

        radius_x = rand_radius * np.cos(rand_angle)
        radius_y = rand_radius * np.sin(rand_angle)
        radius_ball = 0.02
        ball_placement = [radius_x, radius_y, radius_ball]

        # take location and rotate to match plate
        self.data.qpos[self.ball_q : self.ball_q + 3] = plate_top + plate_rot @ ball_placement

        # random start kick
        self.kick()
    
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.last_ctrl = self.data.ctrl.copy() 

        return self._get_obs()

    def _get_obs(self):
        # get ball position + distance from center of plate + plate
        ball_q = self.model.jnt_qposadr[self.model.joint("ball").id]
        ball_pos= self.data.qpos[ball_q : ball_q + 3]
        _, _, ball_on_plate_z = self.data.sensor("ball_pos").data


        # get ball vlocity + put it in axes of plate for standardization
        ball_v = self.model.jnt_dofadr[self.model.joint("ball").id]
        ball_velo= self.data.qvel[ball_v: ball_v + 3]

        motor_rotate_pos = self.data.ctrl[:]

        return np.concatenate([ball_pos, ball_velo, [ball_on_plate_z],
                        motor_rotate_pos])

    def step(self, action):

        reward = 0
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        roll = action[0] * self.max_tilt
        pitch = action[1] * self.max_tilt
        self.data.ctrl[:] = self.set_servo_angles(roll, pitch, 0)

        self.random_ball_movment() 

        # each epoch is 10 seconds
        for _ in range(20): 
            mujoco.mj_step(self.model, self.data)
        
        # progress step
        self.steps += 1
        
        new_objects = self._get_obs()

        # distance from center + height of ball
        x, y, ball_height = self.data.sensor("ball_pos").data
        dist = np.hypot(x, y)

        # check ball fell off
        terminate = bool(dist > self.plate_radius or ball_height <= 0.01) 

        # motors changing pos
        change = self.data.ctrl - self.last_ctrl

        # create reward (with bonus for just surviving)
        alive = 0.2
        centre = 0.7 * ( 1 - dist/self.plate_radius)
        jitter = -0.1 * np.sum(change**2)
        reward += alive + centre + jitter

        # each piece kept separately so main.py can graph them
        self.reward_parts = {"alive": alive, "centre": centre, "jitter": jitter}

        # check truncation
        truncate = bool(self.steps >= self.max_steps)

        self.last_ctrl = self.data.ctrl.copy()
        self._draw()

        return new_objects, reward, terminate, truncate

    def random_ball_movment(self): 

        # random ball movement every 2 seconds
        if self.rand.random() < 1/100:
            self.kick()

    def kick(self):
        kick = self.rand.uniform(0, self.max_kick)
        kick_dir = self.rand.uniform(0, 2 * np.pi)

        xvelo_kick = kick * np.cos(kick_dir)
        yvelo_kick = kick * np.sin(kick_dir)

        # in plate cords 
        plate_rot = self.plate_rot()

        ball_kick = plate_rot @ [xvelo_kick, yvelo_kick, 0]

        self.data.qvel[self.ball_v : self.ball_v + 3] += ball_kick

    def plate_rot(self): 
        return self.data.body("platform").xmat.reshape(3, 3)

    def _draw(self):
        if self.viewer is not None:
            self.viewer.sync()      # push current state to the window
            time.sleep(0.04)        # slow to roughly real time

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
