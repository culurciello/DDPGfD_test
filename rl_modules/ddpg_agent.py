# E. Culurciello
# November 2020

# DDPGfd test from: https://github.com/MrSyee/pg-is-all-you-need/blob/master/06.DDPGfD.ipynb

import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_modules.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl_modules.utils import OUNoise
from rl_modules.models import Actor, Critic

np.set_printoptions(precision=2)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

class DDPGfDAgent:
    """DDPGfDAgent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        demo (list): demonstration
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        pretrain_step (int): the number of step for pre-training
        n_step (int): the number of multi step
        use_n_step (bool): whether to use n_step memory
        prior_eps (float): guarantees every transitions can be sampled
        lambda1 (float): n-step return weight
        lambda2 (float): l2 regularization weight
        lambda3 (float): actor loss contribution of prior weight
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        self,
        args: list,
        env: gym.Env,
        # memory_size: int,
        # batch_size: int,
        # ou_noise_theta: float,
        # ou_noise_sigma: float,
        # demo: list,
        # pretrain_step: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        # initial_random_steps: int = 1e4,
        # # PER parameters
        alpha: float = 0.3,
        beta: float = 1.0,
        prior_eps: float = 1e-6,
        # # N-step Learning
        # n_step: int = 3,
        # # loss parameters
        lambda1: float = 1.0, # N-step return weight
        lambda2: float = 1e-4, # l2 regularization weight
        lambda3: float = 1.0, # actor loss contribution of prior weight
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.args = args
        self.env = env
        self.batch_size = self.args.batch_size
        self.pretrain_step = self.args.pretrain_step
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = self.args.initial_random_steps
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        
        self.demo = self.args.demo
        demos_1_step, demos_n_step = [], []
        if self.demo:
            demos_1_step, demos_n_step = self._get_n_step_info_from_demo(
                self.demo, self.args.n_step
            )
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, self.args.memory_size, self.args.batch_size, gamma, alpha, demo=demos_1_step
        )
        
        # memory for N-step Learning
        self.use_n_step = True if self.args.n_step > 1 else False
        if self.use_n_step:
            self.n_step = self.args.n_step
            self.memory_n = ReplayBuffer(
                obs_dim, 
                self.args.memory_size, 
                self.args.batch_size, 
                gamma, 
                demos_n_step, 
                self.n_step
            )
                
        # noise
        self.noise = OUNoise(
            action_dim,
            theta=self.args.ou_noise_theta,
            sigma=self.args.ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=lambda2,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=1e-3,
            weight_decay=lambda2,
        )
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
        
        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            transition = self.transition
            if self.use_n_step:
                transition = self.memory_n.store(*self.transition)

            # add a single step transition
            if transition:
                self.memory.store(*transition)
    
        return next_state, reward, done
    
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.memory.sample_batch(self.beta)        
        state = torch.FloatTensor(samples["obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(device)
        epsilon_d = samples["epsilon_d"]
        indices = samples["indices"]
        
        # train critic
        # 1-step loss
        critic_loss_element_wise = self._get_critic_loss(samples, self.gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)
        
        # n-step loss
        if self.use_n_step:
            samples_n = self.memory_n.sample_batch(indices)
            n_gamma = self.gamma ** self.n_step
            critic_loss_n_element_wise = self._get_critic_loss(
                samples_n, n_gamma
            )
            
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.lambda1
            )
            critic_loss = torch.mean(critic_loss_element_wise * weights) 
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
                
        # train actor
        actor_loss_element_wise = -self.critic(state, self.actor(state))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # target update
        self._target_soft_update()
        
        # PER: update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.prior_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += epsilon_d
        self.memory.update_priorities(indices, new_priorities)
        
        # check the number of sampling demos
        demo_idxs = np.where(epsilon_d != 0.0)
        n_demo = demo_idxs[0].size
        
        return actor_loss.data, critic_loss.data, n_demo
    
    def _pretrain(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pretraining steps."""
        actor_losses = []
        critic_losses = []
        print("Pre-Train %d step." % self.pretrain_step)
        for _ in range(1, self.pretrain_step + 1):
            actor_loss, critic_loss, _ = self.update_model()
            actor_losses.append(actor_loss.data)
            critic_losses.append(critic_loss.data)
        print("Pre-Train Complete!\n")
        return actor_losses, critic_losses
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        actor_losses, critic_losses, n_demo_list, scores = [], [], [], []
        score = 0
        
        if self.demo:
            output = self._pretrain()
            actor_losses.extend(output[0])
            critic_losses.extend(output[1])
        
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # PER: increase beta
            fraction = min(self.total_step / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:         
                state = self.env.reset()
                scores.append(score)
                last_score = score
                score = 0

            # if training is ready
            if (
                len(self.memory) >= self.batch_size 
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss, n_demo = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                n_demo_list.append(n_demo)
            
                # reporting
                if self.total_step % plotting_interval == 0:
                    print('Steps:', self.total_step, 'scores:', last_score,  
                        'actor, critic losses:', actor_loss.item(),  critic_loss.item(), 
                        'n_demo_list:', n_demo)
                
                # saving model
                if self.total_step % 10000 == 0:
                    #save model:
                    torch.save([self.actor.state_dict()], 
                        self.model_path+'/model_'+str(self.total_step)+'.pth')

            # plotting:
            # if self.total_step % plotting_interval == 0:
            #     self._plot(
            #         self.total_step, 
            #         scores, 
            #         actor_losses, 
            #         critic_losses,
            #         n_demo_list,
            #     )
                
        self.env.close()
        
    def test(self):
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames
    

    def _get_critic_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        device = self.device  # for shortening the following lines
        
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + gamma * next_value * masks
        curr_return = curr_return.to(device).detach()

        # train critic
        values = self.critic(state, action)
        critic_loss_element_wise = (values - curr_return).pow(2)

        return critic_loss_element_wise
    

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
    

    def _get_n_step_info_from_demo(
        self, demo: List, n_step: int
    ) -> Tuple[List, List]:
        """Return 1 step and n step demos."""
        demos_1_step = list()
        demos_n_step = list()
        n_step_buffer: Deque = deque(maxlen=n_step)

        for transition in demo:
            n_step_buffer.append(transition)

            if len(n_step_buffer) == n_step:
                # add a single step transition
                demos_1_step.append(n_step_buffer[0])

                # add a multi step transition
                curr_state, action = n_step_buffer[0][:2]
                
                # get n-step info
                reward, next_state, done = n_step_buffer[-1][-3:]
                for transition in reversed(list(n_step_buffer)[:-1]):
                    r, n_o, d = transition[-3:]

                    reward = r + self.gamma * reward * (1 - d)
                    next_state, done = (n_o, d) if d else (next_state, done)
                
                transition = (curr_state, action, reward, next_state, done)
                demos_n_step.append(transition)

        return demos_1_step, demos_n_step
    

    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
        n_demo: List[int],
    ):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)
                   
        subplot_params = [
            (141, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (142, "actor_loss", actor_losses),
            (143, "critic_loss", critic_losses),
            (144, "the number of sampling demos", n_demo),
        ]
        
        # clear_output(True)
        plt.figure(figsize=(30, 5))            
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()
        plt.pause(0.001)
