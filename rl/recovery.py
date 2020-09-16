import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from rl.agent import AgentVars, DualLearningRateAgent
from rl.estimation import Estimation, EstimationVars
from rl.interaction import agent_task_interaction
from rl.plot_utils import set_mpl_defaults
from rl.plots import plot_recovery_results
from rl.task import ReversalLearningTask, TaskVarsKahntPark2008


def run_recovery(task, agent, est, parameter_values):
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
        3 * 3 * len(parameter_values["alpha_win"]["variable"])
        + 3 * 3 * len(parameter_values["alpha_loss"]["variable"])
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
                    agent.agent_vars.alpha_win = parameters["alpha_win"]
                    agent.agent_vars.alpha_loss = parameters["alpha_loss"]
                    agent.agent_vars.beta = parameters["beta"]

                    # Simulate data, by letting the agent interact with the task
                    data = agent_task_interaction(task, agent)

                    # Estimate parameters from the simulated data, using the estimation instance
                    nll, bic, (alpha_win_hat, alpha_loss_hat, beta_hat) = est.estimate(
                        data=data, seed=i
                    )

                    # Write results to a DataFrame and add it to the results list
                    result = pd.DataFrame(
                        dict(
                            variable=var,
                            alpha_win=parameters["alpha_win"],
                            alpha_loss=parameters["alpha_loss"],
                            beta=parameters["beta"],
                            alpha_win_hat=alpha_win_hat,
                            alpha_loss_hat=alpha_loss_hat,
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


# Running this script standalone does not seem to work because of the imports. It should work, once the rl module is installed in the python path, though.
if __name__ == "__main__":

    # Set up task
    task_vars = TaskVars()
    task_vars.n_trials = 100
    task_vars.n_blocks = 2
    task = Task(task_vars=task_vars)

    # Set up agent
    agent_vars = AgentVars()
    agent_vars.alpha_win = np.nan
    agent_vars.alpha_loss = np.nan
    agent_vars.beta = np.nan
    agent = Agent(agent_vars=agent_vars)

    # Set up parameter ranges
    alpha_win_values = {
        "variable": np.linspace(0, 1.0, 10),
        "low": 0.1,
        "medium": 0.3,
        "high": 0.6,
    }
    alpha_loss_values = {
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
        "alpha_win": alpha_win_values,
        "alpha_loss": alpha_loss_values,
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
        recovery_results, variable_parameter="alpha_win", limits=[0, 1], ticks=[0, 1]
    )
    plt.savefig("recovery_alpha-win.pdf")
