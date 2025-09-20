# Hierarchical-Collective-MARL

## Introduction

Hierarchical multi-agent Reinforcement and Collective Learning (HRCL) is a powerful approach to solve decentralized 
combinatorial optimization problems in evolving multi-agent systems. It combines multi-agent reinforcement learning 
(MAPPO) and multi-agent collective learning (I-EPOS). 

## Setup

From builds upon Python 3.7 to 3.9

### 1. Clone this repo:
```
git clone git@github.com:TDI-Lab/Hierarchical-Collective-MARL.git
```

### 2. Install Prerequisites:
```
pip install -r requirements.txt
```

### 3. Modify parameters

- Modify the properties of algorithms in `conf/hrcl.properties` and `conf/epos.properties`.

- Modify the environments in `environment/make_env.py`.

- Modify the hyperparameters of reinforcement learning in `Main.py`.

### 4. Run the code:
```
python Main.py
```

## Code structure

```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── requirements.txt                <- The python environment for developers using this project.
├── IEPOS.jar                       <- The jar file to run EPOS in the HRCL approach
├── conf
│   ├── hrcl.properties             <- The parameters of the HRCL approach
│   ├── epos.properties             <- The parameters of the EPOS approach
│   ├── log4j.properties             
│   ├── measurement.conf             
│   ├── protopeer.conf            
├── datasets
│   ├── gaussian_origin             <- The orginal dataset of synthetic scenario
│   ├── energy_origin               <- The orginal dataset of energy management scenario
│   ├── gaussian.csv                <- The targets of synthetic scenario
├── environment
│   ├── make_env.py                 <- Create the environent of the scenarios
│   ├── PlanEnv.py                  <- Basic environment for MARL model
│   ├── DataExtract.py              <- Extract the plan data from the original dataset
├── tool
│   ├── mappo_mpe.py                <- Actor-critic networks and proximal policy optimization
│   ├── normalization.py            
│   ├── replay_buffer.py  
├── model
├── runs
├── log                             <- Logging and results output
└────── Main.py                     <- Case study and hyperparameter settings
```


## Documents

More details of I-EPOS can be found [here](https://github.com/epournaras/EPOS).

The benchmark datasets, including synthetic scenario (gaussian) and energy management, can be found in our 
[Figshare](https://doi.org/10.6084/m9.figshare.7806548.v6).


## Citation

If you use HRCL in any of your work, please cite our paper:
~~~

~~~