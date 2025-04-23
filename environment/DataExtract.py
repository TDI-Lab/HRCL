import os
import numpy as np
import pandas as pd
from pathlib import Path


class DataExtract:

    def __init__(self, dataset, agents, plans_num,
                 plan_groups_num, is_grouping, is_behavior):

        self.dataset = dataset
        self.agents = agents
        self.plansNum = plans_num
        self.plan_groups_num = plan_groups_num
        self.isGrouping = is_grouping
        self.isBehavior = is_behavior
        self.generated_plans = None

        if self.dataset == 'gaussian':
            self.generated_plans = self.read_generated_plans_gaussian()

        if self.dataset == 'energy':
            self.generated_plans = self.read_generated_data_energy()

    def read_generated_plans_gaussian(self):
        generated_plans = {}
        print("To generate the plans...")
        plans_num_stored = 16
        dataset_path_from = os.path.join(os.getcwd(), f'datasets/{self.dataset}_origin/')
        dataset_path_to = os.path.join(os.getcwd(), f'datasets/{self.dataset}/')
        if not Path(dataset_path_to).exists():
            Path(dataset_path_to).mkdir()

        for a_id in range(self.agents):
            files_num_to_find = int(self.plansNum / plans_num_stored)
            costs = []
            plans = []
            plans_format = ['' for i in range(self.plan_groups_num)]
            for f_id in range(files_num_to_find):
                plan_id = a_id * files_num_to_find + f_id
                plan_path = dataset_path_from + f'agent_{plan_id}.plans'
                plan_data_read = pd.read_csv(plan_path, sep=':', header=None, names=['cost', 'plan'])

                costs_read = [i / plans_num_stored for i in range(plans_num_stored)]
                plans_read = plan_data_read['plan'].tolist()
                plans_read = [list(map(float, plans_read[i].split(','))) for i in range(plans_num_stored)]
                costs = costs + costs_read
                plans = plans + plans_read

                # store the plans in str format for epos
                for a in range(self.plan_groups_num):
                    b = int(plans_num_stored / self.plan_groups_num)
                    for j in range(b):
                        i = a * b + j
                        plan_format = f'{costs_read[i]}:' + ','.join(map(str, plans_read[i])) + '\n'
                        plans_format[a] = plans_format[a] + plan_format

            plans_for_agent = {'costs': np.array(costs), 'plans': np.array(plans)}
            generated_plans[a_id] = plans_for_agent

            # Store the plans into the input dataset
            if self.isGrouping:
                agent_directory_path = dataset_path_to + f'agent_{a_id}/'
                if not Path(agent_directory_path).exists():
                    Path(agent_directory_path).mkdir()
                for a in range(self.plan_groups_num):
                    agent_plan_path = agent_directory_path + f'{a}.plans'
                    with open(agent_plan_path, 'w', newline='', encoding='utf-8') as outFile:
                        outFile.write(plans_format[a])
                    outFile.close()

            if self.isBehavior:
                plans_format_all = ''
                for i in range(self.plansNum):
                    plans_format_all = plans_format_all + f'{costs[i]}:' + ','.join(map(str, plans[i])) + '\n'
                agent_plan_path = dataset_path_to + f'agent_{a_id}.plans'
                with open(agent_plan_path, 'w', newline='', encoding='utf-8') as outFile:
                    outFile.write(plans_format_all)
                outFile.close()

        return generated_plans

    def read_generated_data_energy(self):
        print("To generate the plans...")
        generated_plans = {}
        dataset_path_from = os.path.join(os.getcwd(), f'datasets/{self.dataset}_origin/')
        dataset_path_to = os.path.join(os.getcwd(), f'datasets/{self.dataset}/')
        if not Path(dataset_path_to).exists():
            Path(dataset_path_to).mkdir()

        # Store the plans in each group
        for a_idx in range(self.agents):
            plan_path = dataset_path_from + f'agent_{a_idx}.plans'
            plans_in_agent = pd.read_csv(plan_path, sep=':', header=None, names=['cost', 'plan'])

            plans_data_sorted = plans_in_agent.sort_values(by='cost').reset_index(drop=True)
            group_size = len(plans_data_sorted) // self.plan_groups_num
            if not self.isGrouping:
                group_size = len(plans_data_sorted) // self.plansNum
            groups_num = int(len(plans_data_sorted) / group_size)
            plans_data_sorted['group'] = (plans_data_sorted.index // group_size)

            costs = []
            plans = []
            plans_format = ['' for i in range(groups_num)]
            for a in range(groups_num):
                plan_data = plans_data_sorted[plans_data_sorted['group'] == a]
                costs_read = (1 - plan_data['cost']).tolist()  # preference score in energy datasets
                plans_read = plan_data['plan'].tolist()
                plans_read = [list(map(float, plans_read[i].split(','))) for i in range(len(plans_read))]
                costs = costs + costs_read
                plans = plans + plans_read

                for p_id in range(len(costs_read)):
                    plans_format[a] = plans_format[a] + f'{costs_read[p_id]}:' + ','.join(
                        map(str, plans_read[p_id])) + '\n'

            generated_plans_agent = {'costs': np.array(costs), 'plans': np.array(plans)}
            generated_plans[a_idx] = generated_plans_agent

            if self.isGrouping:
                agent_directory_path = dataset_path_to + f'agent_{a_idx}/'
                if not Path(agent_directory_path).exists():
                    Path(agent_directory_path).mkdir()
                for a in range(groups_num):
                    agent_plan_path = agent_directory_path + f'{a}.plans'
                    with open(agent_plan_path, 'w', newline='', encoding='utf-8') as outFile:
                        outFile.write(plans_format[a])
                    outFile.close()

            if self.isBehavior:
                plans_format_all = ''
                for i in range(self.plansNum):
                    plans_format_all = plans_format_all + f'{costs[i]}:' + ','.join(map(str, plans[i])) + '\n'
                agent_plan_path = dataset_path_to + f'agent_{a_idx}.plans'
                with open(agent_plan_path, 'w', newline='', encoding='utf-8') as outFile:
                    outFile.write(plans_format_all)
                outFile.close()

        return generated_plans
