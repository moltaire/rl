#!usr/bin/python
# %% Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rl

matplotlib = rl.plot_utils.set_mpl_defaults(matplotlib)
np.random.seed(1)

# %% 1.1 Set up Reversal-Learning-Task (rv) from Kahnt, Park et al. (2008)
# ------------------------------------------------------------------------
print("1.1 Set up Reversal-Learning-Task (rv) from Kahnt, Park et al. (2008)")
print("---------------------------------------------------------------------")

# Set up the three different states (we changed the rewards from 1 to 0.5, for parameter values to better work with both tasks)
rv_states = {
    # 20:80
    0: {"p_r": [0.2, 0.8], "a_correct": [1], "rewards": [0.5, 0]},
    # 80:20
    1: {"p_r": [0.8, 0.2], "a_correct": [0], "rewards": [0.5, 0]},
    # 50:50
    2: {"p_r": [0.5, 0.5], "a_correct": [0, 1], "rewards": [0.5, 0]},
}
rv_task_vars = rl.task.TaskVars(
    n_trials=100,
    n_blocks=2,
    n_options=2,
    states=rv_states,
    n_trials_reversal_min=10,  # minimum number of trials before reversal
    n_trials_reversal_max=16,  # maximum number of trials without reversal
    p_correct_reversal_min=0.7,  # minimum accuracy before reversal before `n_trials_reversal_max`
)

rv_task = rl.task.ReversalLearningTask(task_vars=rv_task_vars)
print(rv_task)


# %% 1.2 Set up Multiple-State-Task (ms) from Lefebvre et al. (2017) Exp. 2
# -------------------------------------------------------------------------
print("1.2 Set up Multiple-State-Task (ms) from Lefebvre et al. (2017) Exp. 2")
print("----------------------------------------------------------------------")

# States (Conditions from Lefebvre et al. 2017, Experiment 2)
ms_states = {
    # 75/25
    0: {"p_r": [0.75, 0.25], "a_correct": [0], "rewards": [0.5, -0.5],},
    # 25/25
    1: {"p_r": [0.25, 0.25], "a_correct": [0, 1], "rewards": [0.5, -0.5],},
    # 25/75
    2: {"p_r": [0.25, 0.75], "a_correct": [1], "rewards": [0.5, -0.5],},
    # 75/75
    3: {"p_r": [0.75, 0.75], "a_correct": [0, 1], "rewards": [0.5, -0.5],},
}
n_states = len(ms_states)

n_blocks = 1

# Build state sequence: Each condition (pair) was shown 24 times
n_repeats = 24
state_sequence = np.repeat(np.arange(n_states), n_repeats)
# Randomize the order of conditions
np.random.shuffle(state_sequence.ravel())
state_sequence = state_sequence.reshape((n_blocks, -1))

# Number of blocks and trials are automatically read from `state_sequence.shape`
ms_task_vars = rl.task.TaskVars(
    states=ms_states, state_sequence=state_sequence, n_options=2
)

ms_task = rl.task.MultipleStateTask(task_vars=ms_task_vars)
print(ms_task)


# %% 2. Set up Dual-Learning-Rate Agent
# -------------------------------------
print("2. Set up Dual-Learning-Rate Agent")
print("----------------------------------")


# Use mean parameters from Lefebvre et al. (2017)
agent_vars = rl.agent.AgentVars(alpha_pos=0.36, alpha_neg=0.22, beta=(1 / 0.13))
agent = rl.agent.DualLearningRateAgent(
    agent_vars=agent_vars, n_options=rv_task.task_vars.n_options
)
print(agent)

# %% 3.1 Let the agent perform the RV-task
# ----------------------------------------
print("3.1 Let the agent perform the RV-task")
print("-------------------------------------")

rv_data = rl.interaction.agent_task_interaction(rv_task, agent)
rl.plots.plot_data(rv_data)
plt.savefig("rv_example-data.pdf")
rv_data.to_csv("rv_example-data.csv")

# %% 3.2 Let the agent perform the MS-task
# ----------------------------------------
print("3.2 Let the agent perform the MS-task")
print("-------------------------------------")

ms_data = rl.interaction.agent_task_interaction(ms_task, agent)
fig, axs = rl.plots.plot_data(ms_data.sort_values("state").reset_index())
axs[-1].set_xlabel("Trial (re-sorted by state for visualization)")
plt.savefig("ms_example-data.pdf")
ms_data.to_csv("ms_example-data.csv")
