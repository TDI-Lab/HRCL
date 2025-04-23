## INTRODUCTION

This dataset was obtained by disaggregation of the simulated zonal power transmission in the Pacific Northwest dataset. The resulting dataset contains loads of 5600 users, recorded every 5 minutes in a 12 hours span of a day. For the disaggregation algorithm, please refer to the Algorithm 1 in the following publication:

Pournaras, E., Yao, M. and Helbing, D., 2017. Self-regulating supply–demand systems. Future Generation Computer Systems, 76, pp.73-91.


## AGENT INFO

Directory 'energy' contains 5600 files, each corresponding to one and only one agent. The naming scheme is as follows:

	agent_XXXX.plans

where XXXX stands for the agent identification which belongs to the range [0, 5599].



## PLANS INFO

Each agent file consist of exactly 10 possible plans. Each line represents one possible plan. The format of the line is as follows:

(score):(value1,value2,value3, ....,value98)

where (score) indicates the preference score (explained below). There are 144 comma-separated values after the colon sign (':'), each representing power consumption recorded every 5 minutes in a 12-hour time span during one day.

The first possible plan has preference score of 1, and it is the plan that was the output of the Algorithm 1 from the paper mentioned above. The next 3 possible plans are obtained via the SHUFFLE generation scheme that randomly permutes values from the plan 1. Since at most 144 values can change their position in the vector, the preference score of these plans is 1/144, and is the lowest one. They are least preferred since they are displaced the most from the original power consumption. Then, the next 3 possible plans are obtained via SWAP-15 generation scheme which randomly picks 15 positions in plan 1 and swaps the values on those positions. The preference score of these plans is accordingly 1/15. Finally, the last 3 possible plans are obtained via SWAP-30 generation scheme which picks 30 positions in plan 1 to swap the values on those positions, and consequently the preference score is 1/30. Note that preference score of possible plans obtained via SWAP-15 scheme is higher than those obtained via SWAP-30 scheme since a lower number of swaps is performed.

All possible plans have preference scores that belong to the range [0,1]. Moreover, since all possible plans of one agent contain the same set of values, only permuted, it is guaranteed that all possible plans of one agent have the same mean and standard deviation of their values.

## CONTACT

For inquiries, please contact Evangelos Pournaras (mail@epos-net.org)
