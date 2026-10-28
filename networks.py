import os 
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module): 

    def __init__(self, input_dims, n_actions, fc1_dims=128, fc2_dims=64, name='critic', 
                 checkpoint_dir='td3/ball_holder', learning_rate=1e-3): 
        
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.model  = nn.Sequential(nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims), 
                               nn.LayerNorm(self.fc1_dims), nn.LeakyReLU(),
                               nn.Linear(self.fc1_dims, self.fc2_dims), 
                               nn.LayerNorm(self.fc2_dims), nn.LeakyReLU(),
                               nn.Linear(self.fc2_dims, 1)
                               )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, actions): 
        return self.model(T.cat([state, actions], dim=1))

    def save_checkpoint(self): 
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module): 

    def __init__(self, input_dims, n_actions, fc1_dims=128, fc2_dims=64, name='actor', 
                 checkpoint_dir='td3/ball_holder', learning_rate=1e-3):    

        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.model  = nn.Sequential(nn.Linear(self.input_dims[0], self.fc1_dims), 
                                       nn.LayerNorm(self.fc1_dims), nn.LeakyReLU(),
                                       nn.Linear(self.fc1_dims, self.fc2_dims), 
                                       nn.LayerNorm(self.fc2_dims), nn.LeakyReLU(),
                                       nn.Linear(self.fc2_dims, n_actions), 
                                       nn.Tanh()
                                       )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state): 
        return self.model(state)

    def save_checkpoint(self): 
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    