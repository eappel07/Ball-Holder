import numpy as np

class ReplayBuffer():

    def __init__(self, max_size, input_shape, n_actions): 

        self.memory_size = max_size
        self.memory_counter = 0 

        self.state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.finished_mem = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done): 

        ticker = self.memory_counter % self.memory_size

        self.state_memory[ticker] = state
        self.action_memory[ticker] = action
        self.reward_memory[ticker] = reward
        self.new_state_memory[ticker] = new_state 
        self.finished_mem[ticker] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size=16):

        mem = min(self.memory_size, self.memory_counter)

        batch = np.random.choice(mem, batch_size, replace=False)

        rand_state, rand_action = self.state_memory[batch], self.action_memory[batch]
        rand_new_state, rand_reward = self.new_state_memory[batch], self.reward_memory[batch]
        rand_done = self.finished_mem[batch]

        return rand_state, rand_action, rand_reward, rand_new_state, rand_done

