import os
import subprocess
import csv
import random
import numpy as np
import pandas as pd
import gym
from gym import spaces

from environment.DataExtract import DataExtract

class PlanEnv(gym.Env):

    def __init__(self, env_config):
        # INPUT
        # Number of the agents
        self.n = env_config["agents"]
        # Number of steps in an episode
        self.steps = env_config["steps"]
        # Number of dimension in the plan
        self.dimension = env_config["dimension"]
        # Total number of plans per agent
        self.plansNum = env_config["plansNum"]
        # The global cost function
        self.global_cost_function = env_config["global_cost_func"]
        # Tradeoff weight between global and local cost
        self.sigma = env_config["sigma"]
        # The name of datasets scenario
        self.dataset = env_config["datasets"]

        # The high-level strategies
        self.isGrouping = env_config['grouping']
        self.isBehavior = env_config['behavior']
        # Number of action dimensions
        self.a_dim_g = env_config["a_dim_g"]
        self.a_dim_b = env_config["a_dim_b"]
        # Set the action dim without high-level strategies, i.e., the baseline MAPPO
        if not self.isGrouping and not self.isBehavior:
            self.a_dim = self.plansNum
        else:
            self.a_dim = int(self.a_dim_g * self.a_dim_b)

        # cross validation, 70% of targets for training and 30% for testing
        data_num = env_config["dataNum"]
        self.data_idx_test = random.sample(range(data_num), int(data_num * 0.3))
        self.data_idx_train = [num for num in range(data_num) if num not in self.data_idx_test]

        # IMPORTANT PARAMS
        # The index of the step/period in an episode
        self.phase = None
        # The target at the current step
        self.target = None
        self.response_total = None
        self.total_target = None
        # Read the generated plans of agents
        data_extract = DataExtract(self.dataset, self.n, self.plansNum,
                                   self.a_dim_g, self.isGrouping, self.isBehavior)
        self.generated_plans = data_extract.generated_plans

        # ACTION AND OBSERVATION
        self.action_space = []
        self.observation_space = []
        for a in range(self.n):
            # Dimension space of action (D x D matrix) and observation (D vector)
            self.action_space.append(spaces.Discrete(self.a_dim))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf,
                                                     shape=(12,), dtype=np.float32))

    def reset(self, evaluate=False):
        self.phase = 1
        self.response_total = np.zeros(self.dimension)

        # Set the target game
        data_path = os.path.join(os.getcwd(), f'datasets/Map/{self.dataset}.csv')
        targets_arr = np.array(pd.read_csv(data_path, header=None))

        data_id = random.choice(self.data_idx_test) if evaluate else random.choice(self.data_idx_train)
        self.total_target = targets_arr[data_id]
        self.target = self.total_target

        obs_n = []
        for agent_id in range(self.n):
            reward = 120
            cost_idx = agent_id % 16
            plans = self.generated_plans[agent_id]['plans']
            selected_plan = np.mean(plans, axis=0)
            local_global_cost = self.cost_function(selected_plan, self.target)
            other_global_cost = self.cost_function(self.target - selected_plan, self.target)

            observation = [agent_id, reward, 0, 0, cost_idx, reward - cost_idx,
                           local_global_cost, other_global_cost, local_global_cost, other_global_cost,
                           reward, self.phase]
            obs_agent = np.array(observation)
            obs_n.append(obs_agent)

        return obs_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        # If no high-level strategies, then the baseline MAPPO
        if not self.isBehavior and not self.isGrouping:
            selected_plan_indexes = np.zeros(shape=self.n, dtype=np.int32)
            discomfort_cost = 0
            response = np.zeros(self.dimension)
            for agent_id in range(self.n):
                selected_plan_idx = action_n[agent_id]
                generated_plans_agent = self.generated_plans[agent_id]
                cost = generated_plans_agent['costs'][selected_plan_idx]
                plan = generated_plans_agent['plans'][selected_plan_idx]
                selected_plan_indexes[agent_id] = selected_plan_idx
                discomfort_cost += cost
                response += plan
            discomfort_cost = discomfort_cost / self.n

        else:
            # Write the target
            target_path = os.path.join(os.getcwd(), f'datasets/{self.dataset}/{self.dataset}.target')
            with open(target_path, 'w', newline='', encoding='utf-8-sig') as targetFile:
                writer = csv.writer(targetFile)
                writer.writerow(self.target)
            targetFile.close()

            if self.isBehavior and self.isGrouping:
                # Action is behavior x grouping
                action_behavior = np.array(action_n / self.a_dim_g, dtype=np.int32)
                action_grouping = np.array(action_n % self.a_dim_g, dtype=np.int32)
            else:
                action_behavior = action_n
                action_grouping = action_n

            if self.isBehavior:
                beta_action = np.vectorize(self.action_to_beta)(action_behavior)
                path_behavior = os.path.join(os.getcwd(), f'datasets/{self.dataset}/behaviours.csv')
                behavior_df = pd.DataFrame(columns=['idx', 'alpha', 'beta'])
                behavior_df['idx'] = np.array(range(self.n))
                behavior_df['alpha'] = np.zeros(self.n, dtype=np.int32)
                behavior_df['beta'] = beta_action
                behavior_df.to_csv(path_behavior, index=False, header=False)

            if self.isGrouping:
                # The plans number per agent is plansNum / actions
                plans_format_input = [str(a) for a in action_grouping]
                plans_format_input = ",".join(plans_format_input)
                command = ['java', '-jar', 'IEPOS_input.jar',
                           f'{self.phase}', 'True', f'{plans_format_input}']
            else:
                command = ['java', '-jar', 'IEPOS_input.jar', f'{self.phase}', 'False']

            epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
            # 3. Get the results from EPOS: global cost, local cost, and global response
            epos_output = epos_output.replace('\n', '').split(',')
            while not epos_output[0].isdigit():
                epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                epos_output = epos_output.replace('\n', '').split(',')
            selected_plan_indexes = np.array(
                [int(x) for x in epos_output[-self.dimension - 2 - self.n:-self.dimension - 2]])
            discomfort_cost = float(epos_output[-self.dimension - 2])
            response = np.array([float(x) for x in epos_output[-self.dimension:]])

        # Calculate system-wide cost
        self.response_total = self.response_total + response
        inefficiency_cost = self.cost_function(self.response_total, self.total_target)
        global_epos = self.cost_function(response, self.target)

        # For each agent, update the individual observation and calculate the reward
        for agent_id in range(self.n):

            reward = self.reward_calc(inefficiency_cost, discomfort_cost)
            reward_n.append(reward)

            generated_plans_agent = self.generated_plans[agent_id]
            costs = generated_plans_agent['costs']
            plans = generated_plans_agent['plans']
            select_col = selected_plan_indexes[agent_id]
            if self.isGrouping:
                select_col = select_col + int(action_grouping[agent_id] * self.plansNum / self.a_dim_g)
            local_cost_agent = costs[select_col]
            selected_plan = plans[select_col]

            # Set the observation of the agent
            other_local_cost = discomfort_cost * self.n - local_cost_agent
            local_global_cost = self.cost_function(selected_plan, self.total_target)
            other_global_cost = self.cost_function(self.response_total - selected_plan, self.total_target)
            local_global_epos = self.cost_function(selected_plan, self.target)
            other_global_epos = self.cost_function(response - selected_plan, self.target)
            observation = [agent_id, discomfort_cost, inefficiency_cost, global_epos, local_cost_agent, other_local_cost,
                           local_global_cost, other_global_cost, local_global_epos, other_global_epos,
                           reward, self.phase + 1]
            obs_agent = np.array(observation)
            obs_n.append(obs_agent)

            done_n.append(False)

        # Update the target
        if self.phase < self.steps:
            self.phase += 1
            # target_new = self.targets_arr[self.phase - 1]
            # self.target = self.target - response + target_new
            self.target = self.target - response

        print(f"Step: {self.phase-1}; Reward: {sum(reward_n)/self.n}")
        state_info = self.response_total.tolist()
        action_info = action_n.tolist()
        info_n.append(inefficiency_cost)
        info_n.append(discomfort_cost)
        info_n.append(state_info)
        info_n.append(action_info)

        return obs_n, reward_n, done_n, info_n

    def cost_function(self, response, target=None):
        if self.global_cost_function == 'VAR':
            return np.var(response, ddof=1) / self.n

        if self.global_cost_function == 'RMSE':
            return np.sqrt(np.mean((target - response) ** 2))

        if self.global_cost_function == 'RSS':
            # Normalize vectors to unit length
            y_true_scale = np.linalg.norm(target) + 1
            y_pred_scale = np.linalg.norm(response) + 1
            y_true_normalized = target / y_true_scale
            y_pred_normalized = response / y_pred_scale
            # Calculate residual sum of squares
            rss = np.sum((y_true_normalized - y_pred_normalized) ** 2)
            return rss

    def action_to_beta(self, action):
        beta_disparity = 1 / self.a_dim_b

        return round(action * beta_disparity, 2)

    def reward_calc(self, inefficiency_cost, discomfort_cost):
        sigmoid_inefficiency = 1 / (1 + np.exp(-inefficiency_cost))
        sigmoid_discomfort = 1 / (1 + np.exp(-discomfort_cost))

        return - self.sigma * sigmoid_discomfort - (1 - self.sigma) * sigmoid_inefficiency



