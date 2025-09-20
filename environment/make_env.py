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


def make_env():

    from environment.PlanEnv import PlanEnv
    import configparser
    import os

    config_env = configparser.ConfigParser()
    config_env.read(os.path.join(os.getcwd(), f'conf/hrcl.properties'))
    is_group_plan = bool(config_env.get('env', 'isGroupPlans'))
    is_group_behavior = bool(config_env.get('env', 'isGroupBehaviorRanges'))
    plans_group_num = int(config_env.get('env', 'numPlanGroups'))
    behavior_ranges_num = int(config_env.get('env', 'numBehaviorRanges'))
    steps_num = int(config_env.get('env', 'numSteps'))
    data_num = int(config_env.get('env', 'numTargets'))
    sigma = float(config_env.get('env', 'weightForBalance'))

    config_system = configparser.ConfigParser()
    config_system.read(os.path.join(os.getcwd(), f'conf/epos.properties'))
    dataset_name = config_system.get('system', 'numAgents')
    agents_num = int(config_system.get('system', 'numAgents'))
    plan_dim = int(config_system.get('system', 'planDim'))
    plans_num = int(config_system.get('system', 'numPlans'))
    global_cost_func = config_system.get('system', 'globalCostFunction')

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


