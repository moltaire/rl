import numpy as np
import pandas as pd

from rl.agent import AgentVars, DualLearningRateAgent
from rl.task import ReversalLearningTask, TaskVars


def agent_task_interaction(task, agent):

    """ This function simulates the interaction of the task- and agent-object
    """

    # Extract task variables
    n_trials = task.task_vars.n_trials
    n_blocks = task.task_vars.n_blocks
    n_options = task.task_vars.n_options
    T = n_trials * n_blocks
    # Initialize data frame for recorded variables
    df = pd.DataFrame(index=range(0, T), dtype="float")

    # Initialize variables
    ## Task variables
    trial = np.full(T, np.nan)  # trial
    block = np.full(T, np.nan)  # block
    state = np.full(T, np.nan)  # state (rule / condition)
    p_r = np.full((T, n_options), np.nan)  # reward probability for actions
    r = np.full(T, np.nan)  # reward

    ## Agent variables
    a = np.full(T, np.nan)  # decision
    corr = np.full(T, np.nan)  # correct decision
    p_a = np.full((T, n_options), np.nan)  # probability of actions
    v_a = np.full((T, n_options), np.nan)  # action values
    ll = np.full(T, np.nan)  # log choice probability

    t = 0
    # Cycle over blocks b = 1, ..., n_blocks
    # --------------------------------------
    for b in range(0, n_blocks):

        # Reset values (assume new stimuli in each block)
        agent.v_a_t = np.zeros(n_options)
        agent.p_a_t = np.zeros(n_options)

        # Cycle over trials t = 1,...T
        # ----------------------------
        for t_b in range(0, n_trials):

            # Task-agent interaction
            # ----------------------
            # Make a choice
            agent.decide()

            # Return reward
            task.sample_reward(agent.a_t)

            # Learning
            agent.learn(task.r_t)

            # Prepare the next trial (check for reversals / choose next state)
            task.prepare_next_trial()

            # Record task variables
            trial[t] = t
            block[t] = b
            state[t] = task.state_t
            r[t] = task.r_t
            for i in range(n_options):
                p_r[t, i] = task.task_vars.states[task.state_t]["p_r"][i]
            # Record agent variables
            a[t] = agent.a_t
            corr[t] = task.correct_t
            for i in range(n_options):
                p_a[t, i] = agent.p_a_t[i]
                v_a[t, i] = agent.v_a_t[i]
            ll[t] = np.log(agent.p_a_t[np.int(a[t])])

            t += 1  # increment trial counter

    # Attach model variables to data frame
    df["trial"] = trial
    df["block"] = block
    df["state"] = state
    for i in range(n_options):
        df[f"p_r_{i}"] = p_r[:, i]
    df["r"] = r
    df["a"] = a
    df["corr"] = corr
    for i in range(n_options):
        df[f"p_a_{i}"] = p_a[:, i]
        df[f"v_a_{i}"] = v_a[:, i]
    df["ll"] = ll

    return df


if __name__ == "__main__":

    np.random.seed(2)
    OUTPUT_FILE = "example-data.csv"

    # Task
    task_vars = TaskVars()
    task_vars.n_trials = 100
    task_vars.n_blocks = 2

    states = {
        "0": {"p_r": np.array([0.2, 0.8]), "a_correct": [1]},
        "1": {"p_r": np.array([0.8, 0.2]), "a_correct": [0]},
        "2": {"p_r": np.array([0.5, 0.5]), "a_correct": [0, 1]},
    }

    task_vars.states = states
    task_vars.n_trials_reversal_min = 10
    task_vars.n_trials_reversal_max = 16
    task_vars.p_correct_reversal_min = 0.7
    task_vars.reward = 1
    task_vars.noreward = 0

    task = ReversalLearningTask(task_vars=task_vars)

    # Agent
    agent_vars = AgentVars(alpha_win=0.2, alpha_loss=0.1, beta=4)
    agent = DualLearningRateAgent(agent_vars=agent_vars)

    df = agent_task_interaction(task, agent)
    df.to_csv(OUTPUT_FILE)
    print(f"Created example data file at '{OUTPUT_FILE}'")
