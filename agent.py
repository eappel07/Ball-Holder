from networks import ActorNetwork, CriticNetwork
import torch as T
import torch.nn.functional as F
import numpy as np
from replaybuffer import ReplayBuffer

class Agent(): 

    def __init__(self, alr, clr, input_dims, tau, env, gamma=0.99,
                 update_actor_interval=2, warmup=1000, n_actions=2, 
                 max_size=1000000, layer1_size=64, layer2_size=50,
                 noise=0.1, batch_size = 100
                 ): 

        self.alr = alr
        self.clr = clr
        self.tau = tau
        self.gamma = gamma
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.warmup = warmup 
        self.updat_actor_interval = update_actor_interval 
        self.time_step = 0
        self.learn_cnt = 0

        # latest training numbers, read by main.py for graphs
        self.critic_loss = None
        self.actor_loss = None
        self.q_mean = None
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.batch_size = batch_size
        self.max_size = max_size
        self.memory =ReplayBuffer(self.max_size, self.input_dims, self.n_actions)
        self.noise = noise

        self.actor = ActorNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                  n_actions=self.n_actions, name='actor', learning_rate=self.alr)

        self.critic_1 = CriticNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                          n_actions=self.n_actions, name='critic_1', learning_rate=self.clr)

        self.critic_2 = CriticNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                                  n_actions=self.n_actions, name='critic_2', learning_rate=self.clr)


        self.target_actor = ActorNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                          n_actions=self.n_actions, name='target_actor', learning_rate=self.alr)
        
        self.target_critic_1 = CriticNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                            n_actions=self.n_actions, name='target_critic_1', learning_rate=self.clr)

        self.target_critic_2 = CriticNetwork(input_dims=self.input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                                    n_actions=self.n_actions, name='target_critic_2', learning_rate=self.clr)

        self.update_network_parameters(tau=1) 


    def choose_action(self, observation, validation=False):

        if self.time_step < self.warmup and validation is False: 
            mu = T.tensor(np.random.uniform(-1, 1, size=(self.n_actions,))).to(self.actor.device)

        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        if not validation:
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else : mu_prime = mu
        
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def learn(self):

        if self.memory.memory_counter <  10 * self.batch_size: 
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        with T.no_grad():
            target_actions = self.target_actor.forward(next_state)
            target_noise = T.tensor(np.random.normal(scale=0.2, size=target_actions.shape), dtype=T.float).to(self.critic_1.device)
            target_actions = target_actions + T.clamp(target_noise, -0.5, 0.5)
            target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

            next_q1 = self.target_critic_1.forward(next_state, target_actions)
            next_q2 = self.target_critic_2.forward(next_state, target_actions)

            next_q1[done] = 0.0
            next_q2[done] = 0.0

            next_q1 =  next_q1.view(-1)
            next_q2 =  next_q2.view(-1)

            next_critic_value = T.min(next_q1, next_q2)

            target = reward + self.gamma*next_critic_value
            target = target.view(self.batch_size, 1)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_cnt += 1

        # save for logging (.item() = tensor -> plain number)
        self.critic_loss = critic_loss.item()
        self.q_mean = q1.mean().item()

        if self.learn_cnt % self.updat_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor_loss = actor_loss.item()

        self.actor.optimizer.step()
        self.update_network_parameters()


    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_network_parameters(self, tau=None):

        if tau == None: 
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_dict = dict(actor_params)
        critic_1_dict = dict(critic_1_params)
        critic_2_dict = dict(critic_2_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_1_dict = dict(target_critic_1_params)
        target_critic_2_dict = dict(target_critic_2_params)

        for name in actor_dict: 
            actor_dict[name] = tau*actor_dict[name].clone() + (1 - tau) * target_actor_dict[name]

        for name in critic_1_dict: 
            critic_1_dict[name] = tau*critic_1_dict[name].clone() + (1 - tau) * target_critic_1_dict[name]

        for name in critic_2_dict: 
            critic_2_dict[name] = tau*critic_2_dict[name].clone() + (1 - tau) * target_critic_2_dict[name]

        self.target_actor.load_state_dict(actor_dict)
        self.target_critic_1.load_state_dict(critic_1_dict)
        self.target_critic_2.load_state_dict(critic_2_dict)

    def save_models(self): 
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()


    def load_models(self):
        try: 
            self.actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
        except: 
            print("Failed to load, starting from scratch.")