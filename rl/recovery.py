import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from rl.agent import AgentVars, DualLearningRateAgent
from rl.estimation import Estimation, EstimationVars
from rl.interaction import agent_task_interaction
from rl.plot_utils import set_mpl_defaults
from rl.plots import plot_recovery_results, plot_data
from rl.task import ReversalLearningTask, TaskVars


def run_systematic_recovery(task, agent, est, parameter_values):
    """This function runs a systematic parameter recovery.
    Each parameter is treated as `variable` once, and varied over combinations of three levels (`low`, `medium`, `high`) of the other parameters.
    Parameter levels and values are defined in the `parameter_values` dictionary.

    Args:
        task (rl.task.Task): Task instance
        agent (rl.agent.Agent): Agent instance
        est (rl.estimation.estimation): Estimation instance
        parameter_values (dict): Parameter values and levels. Each key is a parameter, and value is a dict with keys `variable` (array like), `low` (float), `medium` (float) and `high` (float).

    Returns:
        pandas.DataFrame: Combined result of all recovery runs.
    """

    # Calculate total number of recoveries to perform
    n_recovs = (
        3 * 3 * len(parameter_values["alpha_pos"]["variable"])
        + 3 * 3 * len(parameter_values["alpha_neg"]["variable"])
        + 3 * 3 * len(parameter_values["beta"]["variable"])
    )

    # Initialize progress bar
    pbar = tqdm(total=n_recovs)

    # Initialize storage for results
    recovery_results = []

    # Create running index
    i = 0

    # Make a handy list of parameter names
    parameter_names = list(parameter_values.keys())

    # Initialize dictionary of parameteter values, to be filled in each run
    parameters = {parameter: np.nan for parameter in parameter_names}

    # Each parameter is the `variable` parameter ones
    for var in parameter_names:

        # Determine the `other`, nonvariable parameters
        others = [p for p in parameter_names if p != var]

        # Set the first other parameter to one of the three levels
        for other0_val in ["low", "medium", "high"]:

            # And the second...
            for other1_val in ["low", "medium", "high"]:

                # Fill the parameter dictionary with the corresponding values
                parameters[others[0]] = parameter_values[others[0]][other0_val]
                parameters[others[1]] = parameter_values[others[1]][other1_val]

                # Loop over values of the `variable` parameter
                for v in parameter_values[var]["variable"]:
                    parameters[var] = v

                    # Assign the parameters to the agent
                    agent.agent_vars.alpha_pos = parameters["alpha_pos"]
                    agent.agent_vars.alpha_neg = parameters["alpha_neg"]
                    agent.agent_vars.beta = parameters["beta"]

                    # Simulate data, by letting the agent interact with the task
                    data = agent_task_interaction(task, agent)

                    # Estimate parameters from the simulated data, using the estimation instance
                    nll, bic, (alpha_pos_hat, alpha_neg_hat, beta_hat) = est.estimate(
                        data=data, seed=i
                    )

                    # Write results to a DataFrame and add it to the results list
                    result = pd.DataFrame(
                        dict(
                            variable=var,
                            alpha_pos=parameters["alpha_pos"],
                            alpha_neg=parameters["alpha_neg"],
                            beta=parameters["beta"],
                            alpha_pos_hat=alpha_pos_hat,
                            alpha_neg_hat=alpha_neg_hat,
                            beta_hat=beta_hat,
                            n_trials=task.task_vars.n_trials,
                            n_blocks=task.task_vars.n_blocks,
                            n_sp=est.est_vars.n_sp,
                        ),
                        index=[i],
                    )
                    result[f"{var}_level"] = "variable"
                    result[f"{others[0]}_level"] = other0_val
                    result[f"{others[1]}_level"] = other1_val

                    recovery_results.append(result)

                    # Update index and progressbar
                    i += 1
                    pbar.update()

    # Combine results into one DataFrame and return it
    recovery_results = pd.concat(recovery_results)

    return recovery_results


def run_estimate_recovery(task, agent, est, parameter_values):
    """This function runs a parameter recovery using a set of parameter estimates.
    Estimates are usually obtained from fitting the model to empirical data.

    Args:
        task (rl.task.Task): Task instance
        agent (rl.agent.Agent): Agent instance
        est (rl.estimation.estimation): Estimation instance
        parameter_values (dict): Dictionary with parameter as keys and arrays of parameters as values. For example `dict(alpha_pos=np.array([0.3, 0.5, 0.3]), alpha_neg=np.array([0.2, 0.3, 0.3]), beta=np.array([2, 3, 4]))`

    Returns:
        pandas.DataFrame: Combined result of all recovery runs.
    """

    # Make a handy list of parameter names
    parameter_names = list(parameter_values.keys())

    # Calculate total number of recoveries to perform
    N = len(parameter_values[parameter_names[0]])

    # Initialize storage for results
    recovery_results = []

    # Cycle over parameter sets (usually these belong to individual subjects)
    for i in tqdm(range(N)):

        # Make a dictionary of parameter values
        parameters = {
            parameter: parameter_values[parameter][i] for parameter in parameter_names
        }

        # Assign the parameters to the agent
        agent.agent_vars.update(**parameters)

        # Simulate data, by letting the agent interact with the task
        data = agent_task_interaction(task, agent)

        # Estimate parameters from the simulated data, using the estimation instance
        nll, bic, parameter_estimates = est.estimate(
            data=data, agent_vars=agent.agent_vars, seed=i
        )

        # Write results to a DataFrame and add it to the results list
        result = pd.DataFrame(
            dict(
                idx=i,
                n_trials=task.task_vars.n_trials,
                n_blocks=task.task_vars.n_blocks,
                n_sp=est.est_vars.n_sp,
                nll=nll,
                bic=bic,
            ),
            index=[i],
        )
        for parameter, estimate in zip(parameter_names, parameter_estimates):
            result[parameter] = parameter_values[parameter][i]
            result[f"{parameter}_hat"] = estimate

        recovery_results.append(result)

    # Combine results into one DataFrame and return it
    recovery_results = pd.concat(recovery_results)

    return recovery_results


# Running this script standalone does not seem to work because of the imports. It should work, once the rl module is installed in the python path, though.
if __name__ == "__main__":

    # Set up task
    task_vars = TaskVars(n_trials=20, n_blocks=2, n_options=2)

    states = {
        "0": {
            "p_r": np.array([0.2, 0.8]),
            "a_correct": [1],
            "rewards": np.array([1, 0]),
        },
        "1": {
            "p_r": np.array([0.8, 0.2]),
            "a_correct": [0],
            "rewards": np.array([1, 0]),
        },
        "2": {
            "p_r": np.array([0.5, 0.5]),
            "a_correct": [0, 1],
            "rewards": np.array([1, 0]),
        },
    }

    task_vars.states = states
    task_vars.n_trials_reversal_min = 10
    task_vars.n_trials_reversal_max = 16
    task_vars.p_correct_reversal_min = 0.7

    task = ReversalLearningTask(task_vars=task_vars)
    print(task)

    # Set up agent
    agent_vars = AgentVars(alpha_pos=np.nan, alpha_neg=np.nan, beta=np.nan)
    agent = DualLearningRateAgent(agent_vars=agent_vars)
    print(agent)

    # Set up parameter ranges
    alpha_pos_values = {
        "variable": np.linspace(0, 1.0, 10),
        "low": 0.1,
        "medium": 0.3,
        "high": 0.6,
    }
    alpha_neg_values = {
        "variable": np.linspace(0, 1.0, 10),
        "low": 0.1,
        "medium": 0.3,
        "high": 0.6,
    }
    beta_values = {
        "variable": np.linspace(1, 10, 10),
        "low": 3,
        "medium": 5,
        "high": 10,
    }
    parameter_values = {
        "alpha_pos": alpha_pos_values,
        "alpha_neg": alpha_neg_values,
        "beta": beta_values,
    }

    # Set up estimation
    est_vars = EstimationVars(task_vars=task_vars)
    est_vars.n_sp = 1
    est = Estimation(est_vars=est_vars)

    # Run recovery
    recovery_results = run_recovery(task, agent, est, parameter_values)
    recovery_results.to_csv("example_recovery-results.csv")

    # Make a plot
    matplotlib = set_mpl_defaults(matplotlib)
    fig, axs = plot_recovery_results(
        recovery_results, variable_parameter="alpha_pos", limits=[0, 1], ticks=[0, 1]
    )
    plt.savefig("recovery_alpha-win.pdf")
