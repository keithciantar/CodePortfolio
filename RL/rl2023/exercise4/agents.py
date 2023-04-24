import os
import gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from rl2023.exercise3.agents import Agent
from rl2023.exercise3.networks import FCNetwork
from rl2023.exercise3.replay import Transition


class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps


class DDPG(Agent):
    """ DDPG
        ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # self.actor = Actor(STATE_SIZE, policy_hidden_size, ACTION_SIZE)
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )

        self.actor_target.hard_update(self.actor)
        # self.critic = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)
        # self.critic_target = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)

        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        self.noise = DiagGaussian(mean, std)

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )


    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path


    def restore(self, filename: str, dir_path: str = None):
        """Restores PyTorch models from models file given by path

        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """

        if dir_path is None:
            dir_path, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dir_path, filename)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        action = self.actor(torch.FloatTensor(obs))
        if explore:
            action += self.noise.sample()
        action = torch.clamp(action, min=self.lower_action_bound, max=self.upper_action_bound).detach().numpy()
        return action


    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critic and actor networks, target networks with soft
        updates, and return the q_loss and the policy_loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        target_action = self.actor_target(batch.next_states)
        target_critic_value = self.critic_target(torch.cat((batch.next_states, target_action), dim=1))
        target_networks_q_values = batch.rewards + ((1-batch.done) * self.gamma * target_critic_value)

        taken_actions_critic_values = self.critic(torch.cat((batch.states, batch.actions), dim=1))

        q_loss = torch.nn.functional.mse_loss(target_networks_q_values, taken_actions_critic_values)

        self.critic.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        actions_to_take = self.actor(batch.states)
        p_loss = torch.mean(-self.critic(torch.cat((batch.states, actions_to_take), dim=1)))

        self.actor.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        self.actor_target.soft_update(source=self.actor,tau=self.tau)
        self.critic_target.soft_update(source=self.critic, tau=self.tau)

        # q_loss = 0.0
        # p_loss = 0.0
        return {
            "q_loss": q_loss.item(),
            "p_loss": p_loss.item(),
        }
