import numpy as np
import pandas as pd

from rl.agent import Agent, AgentVars
from rl.task import Task, TaskVars


def agent_task_interaction(task, agent):

    """ This function simulates the interaction of the task- and agent-object
    """

    # Extract task variables
    n_trials = task.task_vars.n_trials
    n_blocks = task.task_vars.n_blocks
    T = n_trials * n_blocks
    # Initialize data frame for recorded variables
    df = pd.DataFrame(index=range(0, T), dtype="float")

    # Initialize variables
    ## Task variables
    trial = np.full(T, np.nan)  # trial
    block = np.full(T, np.nan)  # block
    rule = np.full(T, np.nan)  # rule
    p_r_0 = np.full(T, np.nan)  # reward probability for action 0
    p_r_1 = np.full(T, np.nan)  # reward probability for action 1
    r = np.full(T, np.nan)  # reward

    ## Agent variables
    a = np.full(T, np.nan)  # decision
    corr = np.full(T, np.nan)  # correct decision
    p_a0 = np.full(T, np.nan)  # choice probability a = 0
    p_a1 = np.full(T, np.nan)  # choice probability a = 1
    v_a_0 = np.full(T, np.nan)  # value a = 0
    v_a_1 = np.full(T, np.nan)  # value a = 1
    ll = np.full(T, np.nan)  # log choice probability

    T = 0
    # Cycle over blocks b = 1, ..., n_blocks
    # --------------------------------------
    for b in range(0, n_blocks):

        # Cycle over trials t = 1,...T
        # ----------------------------
        for t in range(0, n_trials):

            # Task-agent interaction
            # ----------------------
            # Make a choice
            agent.decide()

            # Return reward
            task.sample_reward(agent.a_t)

            # Learning
            agent.learn(task.r_t)

            # Check for reversals
            task.check_reversal()

            # Record task variables
            trial[t] = t
            block[t] = b
            rule[t] = task.rule_t
            p_r_0[t] = task.task_vars.task_rules[task.rule_t]["p_r_0"]
            p_r_1[t] = task.task_vars.task_rules[task.rule_t]["p_r_1"]
            r[t] = task.r_t

            # Record agent variables
            a[t] = agent.a_t
            corr[t] = task.correct_t
            p_a0[t] = agent.p_a_t[0]
            p_a1[t] = agent.p_a_t[1]
            v_a_0[t] = agent.v_a_t[0]
            v_a_1[t] = agent.v_a_t[1]
            ll[t] = np.log(agent.p_a_t[np.int(a[t])])

            T += 1  # increment trial counter

    # Attach model variables to data frame
    df["trial"] = trial
    df["block"] = block
    df["rule"] = rule
    df["p_r_0"] = p_r_0
    df["p_r_1"] = p_r_1
    df["r"] = r
    df["a"] = a
    df["corr"] = corr
    df["p_a0"] = p_a0
    df["p_a1"] = p_a1
    df["v_a_0"] = v_a_0
    df["v_a_1"] = v_a_1
    df["ll"] = ll

    return df


if __name__ == "__main__":

    np.random.seed(2)
    OUTPUT_FILE = "example-data.csv"

    # Task
    task_vars = TaskVars()
    task_vars.n_trials = 100
    task_vars.n_blocks = 1
    task = Task(task_vars=task_vars)

    # Agent
    agent_vars = AgentVars()
    agent = Agent(agent_vars=agent_vars)

    df = agent_task_interaction(task, agent)
    df.to_csv(OUTPUT_FILE)
    print(f"Created example data file at '{OUTPUT_FILE}'")
