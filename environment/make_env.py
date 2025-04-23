"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""


def make_env(scenario_name):

    from environment.PlanEnv import PlanEnv
    import configparser
    import os

    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), f'conf/env.properties')
    config.read(config_path)

    agents_num = int(config.get('env', 'numAgents'))
    plan_dim = int(config.get('env', 'planDim'))
    plans_group_num = int(config.get('env', 'numPlanGroups'))
    behavior_ranges_num = int(config.get('env', 'numBehaviorRanges'))
    steps_num = int(config.get('env', 'numSteps'))
    plans_num = int(config.get('env', 'numPlans'))
    data_num = int(config.get('env', 'numTargets'))
    sigma = float(config.get('env', 'weightForBalance'))
    global_cost_func = config.get('env', 'globalCostFunction')
    is_group_plan = (plans_group_num != 1)
    is_group_behavior = (behavior_ranges_num != 1)

    dataset_name = None
    if scenario_name == 'synthetic':
        dataset_name = 'gaussian'
    if scenario_name == 'energy':
        dataset_name = 'energy'

    env_config = {
        "datasets": dataset_name,
        "agents": agents_num,
        "grouping": is_group_plan,
        "behavior": is_group_behavior,
        "a_dim_g": plans_group_num,
        "a_dim_b": behavior_ranges_num,
        "steps": steps_num,
        "dimension": plan_dim,
        "plansNum": plans_num,
        "global_cost_func": global_cost_func,
        "dataNum": data_num,
        "sigma": sigma,
    }
    env = PlanEnv(env_config)

    return env


