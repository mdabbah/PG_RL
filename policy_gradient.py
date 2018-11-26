import sys
import gym
import numpy as np
from itertools import count
import taxi_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PGAgent(nn.Module):
    def __init__(self, number_of_actions, num_of_states, HL_size, gamma, exploration, device, clip_grads):
        """
        this class represnts an agient which learns with policy gradient methods
        in the simple case the policy is represented by a shallow NN with the following proprities
        :param number_of_actions: the number of actions available to the agent, which is also the output dimension
                                    of the network that represents the policy
        :param num_of_states: if state is represented by a vector then this would be the size of the vector
                            which is also the size of the NN input layer
        :param HL_size: hidden layer size
        :param gamma: the discount factor
        """
        super(PGAgent, self).__init__()

        self.num_of_states, self.number_of_actions = num_of_states, number_of_actions
        self.device = device
        # self.affine1 = nn.Linear(num_of_states, HL_size)
        # self.affine2 = nn.Linear(HL_size, number_of_actions)

        # another arch i played with
        # self.affine1 = nn.Linear(num_of_states, 2048)
        # self.affine2 = nn.Linear(2048, 1024)
        # self.affine3 = nn.Linear(1024, number_of_actions)

        # architecture that also predicts the value function
        self.affine1 = nn.Linear(num_of_states, HL_size)
        self.affine2 = nn.Linear(HL_size, number_of_actions)
        self.affine3 = nn.Linear(HL_size, 1)  # value function is just one number

        self.clip_grads = clip_grads
        self.optimizer = None
        self.gamma = gamma
        self.explor_factor = exploration
        assert 0 < gamma <= 1, 'gamma i.e. the discount factor should be between 0 and 1'
        assert 0 < exploration <= 1, 'epsilon i.e. the exploration factor should be between 0 and 1'

        self.saved_log_probs = []   # saves chosen actions log probability during an episode
        self.rewards = []           # saves the chosen actions rewards during an episode
        self.save_value_function = []    # saves the states during an episode

    def forward(self, state):
        """
        defines the forward pass in the shallow NN we defined
        note: this is a private method and not a part of the api
        :param state:
        :return: softmax on actions
        """
        x = F.relu(self.affine1(state))
        V_s = self.affine3(x)
        x = F.relu(self.affine2(x))
        action_scores = x
        return F.softmax(action_scores, dim=1), V_s

    def set_optimizer(self, optimizer):
        """

        :param optimizer: optimerzer object from the torch.optim library
        :return: None
        """
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def select_action(self, state):
        """
        when the agent is given a state of the world use this method to make the agent chose an action acording
        to the policy
        :param state: state of the world, with the same diminsions that the agient knows
        :return: sampled action according to the policy the agent learned
        """
        one_hot = np.zeros([1, self.num_of_states])  # encode state in oneHot vector
        one_hot[0, state] = 1
        state = torch.tensor(one_hot, device=self.device).float()
        probs, V_s = self.forward(state)
        pai_s = Categorical(probs)

        if self.should_explore():
            action = torch.tensor(taxi_policy.optimal_human_policy(int(state.argmax())), device=self.device)
        else:
            action = pai_s.sample()

        self.saved_log_probs.append(pai_s.log_prob(action))
        self.save_value_function.append(V_s)
        return action.item()

    def should_explore(self):
        """
        applies greedy exploration policy epsilon greedy
        with diminishing epsilon as the agent gets better
        :return: returns true if the agent should explore
        """
        explore = Categorical(torch.tensor([1 - self.explor_factor, self.explor_factor])).sample()
        return False  # explore == 1

    def random_action(self):
        """
        retunrs a random action polled from discrete uniform distribution over the number of actions
        :return: a random action
        """
        uniform_sampler = Categorical(torch.tensor([1/self.number_of_actions]*self.number_of_actions))
        return uniform_sampler.sample()

    def update(self, episode):
        """
        after an episode is finished use this method to update the policy the agent has learned so far acording
        to the monte carlo samples the agent have seen during the episode
        episode parameter is for "episode normalization" code which is not used
        :return: policy loss
        """
        if self.optimizer is None:
            raise ValueError('optimizer not set!'
                             'please use agent.set_optimizer method to set an optimizer')

        R = 0
        policy_loss = []
        value_losses = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, device=self.device)
        # code for trying baseline reduction
        # V_s = torch.tensor(np.zeros(self.number_of_actions), device=self.device)
        # for state in range(self.num_of_states):
        #     V_s[state] = rewards[self.visited_states == state].mean()

        # code for reward normalization trick i found online .. helps with convergence

        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward, v_s in zip(self.saved_log_probs, rewards, self.save_value_function):
            advantage = reward - v_s.item()
            policy_loss.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(v_s.squeeze(), reward).expand(policy_loss[0].shape))

        total_loss = torch.cat(policy_loss).sum().to(self.device) + torch.cat(value_losses).sum().to(self.device)
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grads:
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.save_value_function[:]
        return float(total_loss)


def eval_agent(agent, env, num_of_eps=1, render=False):
    """
    evalueates the given agent with the env for
    '  num_of_eps '
    :param agent: agent
    :param env: env
    :param num_of_eps: number of episodes
    :param render: ef we should render each episode
    :return:
    """
    agent.explor_factor = 0.0001
    for episode in range(num_of_eps):
        state = env.reset()
        episode_reward = []
        for t in range(201):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            if render:
                env.render()
            episode_reward.append(reward)
            if done:
                break
        print('Episode {0}\tLast length: {1}\tepisode reward: {2} '.format(episode, t, np.sum(episode_reward)))


def main():

    seed = 0
    env = gym.make('Taxi-v2')
    env.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = PGAgent(env.action_space.n, env.observation_space.n,
                    HL_size=1024, gamma=0.99, exploration=0.0001, device=device, clip_grads=False)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
    agent.set_optimizer(optimizer)
    agent.to(device)
    last_100_ep_reward = []

    for i_episode in count(1):
        state = env.reset()
        # agent.visited_states.append(state)
        for t in range(1000):  # Don't infinite loop while learning

            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)

            if '-render' in sys.argv:  # option to render states in "video" form
                env.render()
            if done:
                break

        episode_reward = np.sum(agent.rewards)
        policy_loss = agent.update(i_episode)
        last_100_ep_reward.insert(0, episode_reward)
        # printing stats
        log_interval = 2
        if i_episode % log_interval == 0:
            print('Episode {} \tstarting state: {:3d} \tLast length: {:5d}\tepisode reward: {:.3f} \tpolicy loss: {:5f} '.format(
                i_episode, state, t, episode_reward, policy_loss))
        if i_episode > 100:
            last_100_ep_reward.pop()
            # if np.mean(last_100_ep_reward) > env.spec.reward_threshold:
            #     print("Solved! Running reward is now {} and "
            #           "the last episode runs to {} time steps!".format(episode_reward, t))
            #     break

        if i_episode == 10000:  # after 700 the policy seems pretty stable
            break

    torch.save(agent.state_dict(), r'PG_taxi_agent.pt')
    eval_agent(agent, env, 20)
    state = env.reset()
    for t in range(1000):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        screen = env.render(mode='rgb_array')

        if done:
            break


if __name__ == '__main__':
    main()
