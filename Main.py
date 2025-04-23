import os

import pandas as pd
import torch
import numpy as np
import argparse
import time

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tools.normalization import Normalization, RewardScaling
from tools.replay_buffer import ReplayBuffer
from tools.mappo_mpe import MAPPO_MPE
from environment.make_env import make_env


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = make_env(env_name)  # Discrete action space
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in
                               range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space[i].n for i in
                                  range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(
            self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

        self.reward_array = []
        self.observation = [[], [], [], [], [], []]

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps, info = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps
            self.reward_array.append(_)

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        evaluate_global = 0
        evaluate_local = 0
        self.observation[2] = []
        self.observation[3] = []
        self.observation[4] = []

        start_time = time.time()
        for _ in range(self.args.evaluate_times):
            episode_reward, _, info = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
            evaluate_global += info[0]
            evaluate_local += info[1]
            self.observation[2] = self.observation[2] + info[2]
            self.observation[3] = self.observation[3] + info[3]
            self.observation[4] = self.observation[4] + info[4]
        self.observation[0].append(evaluate_global / self.args.evaluate_times)
        self.observation[1].append(evaluate_local / self.args.evaluate_times)
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        time_count = (time.time() - start_time) * 1000 / self.args.evaluate_times

        print("\t \t total_steps:{} \t evaluate_reward:{} \t time:{}".format(self.total_steps, evaluate_reward, time_count))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
                               global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        # print(f"CPU utilization: {psutil.cpu_percent()}%")
        # print(f"Memory utilization: {psutil.virtual_memory().percent}%")

        # Output the results in evaluation
        if self.total_steps > 0:
            self.result_output()

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        episode_global = []
        episode_local = []
        episode_state = []
        episode_action = []
        info = []

        obs_n = self.env.reset(evaluate)

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

            # print("In total ", self.args.episode_limit,
            #       " episode steps, start to run the step of episode: ", episode_step)
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)

            episode_reward += r_n[0]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            else:
                episode_global.append(_[0])
                episode_local.append(_[1])
                episode_state.append(_[2])
                episode_action.append(_[3])

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
        else:
            info.append(sum(episode_global) / self.args.episode_limit)
            info.append(sum(episode_local) / self.args.episode_limit)
            info.append(episode_state)
            info.append(episode_action)
            info.append([['global'] + episode_global, ['local'] + episode_local])

        return episode_reward, episode_step + 1, info

    def result_output(self):
        # Show the figure
        ax = plt.subplot()
        ax.plot(self.reward_array)
        path_a = os.path.join(os.getcwd(), 'log/output/reward.png')
        plt.savefig(path_a, dpi=300, bbox_inches='tight')
        plt.close()
        # Store data
        path_data = os.path.join(os.getcwd(), 'log/output/reward_data.csv')
        data = pd.DataFrame(columns=['metric1', 'metric2'])
        data['metric1'] = self.observation[0]
        data['metric2'] = self.observation[1]
        data.to_csv(path_data, index=False)
        # Store the state
        path_state = os.path.join(os.getcwd(), 'log/output/state.csv')
        data = pd.DataFrame(self.observation[2])
        data.to_csv(path_state, index=False, header=False)
        # Store the action
        path_action = os.path.join(os.getcwd(), 'log/output/action.csv')
        data = pd.DataFrame(self.observation[3])
        data.to_csv(path_action, index=False, header=False)
        # Store the global and local changes per epoch
        path_values_epoch = os.path.join(os.getcwd(), 'log/output/values_epoch.csv')
        data = pd.DataFrame(self.observation[4])
        data.to_csv(path_values_epoch, index=False, header=False)


if __name__ == '__main__':
    # Environment Settings: synthetic, energy, UAV
    env_name = "synthetic"

    # MAPPO Hyperparameters
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")
    parser.add_argument("--max_train_steps", type=int, default=400000, help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=32, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=640,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    args = parser.parse_args()

    runner = Runner_MAPPO_MPE(args, env_name=env_name, number=1, seed=0)
    print("Start to train the model")
    runner.run()

