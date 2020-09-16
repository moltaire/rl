import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

from rl.plot_utils import cm2inch, make_pstring


def plot_data(data):
    """Plot learning data from a single subject.

    Args:
        data (pd.DataFrame): Dataframe containing the data.
                             Must have the following columns:
                             `trial`, `a`, `r`, `state` and
                             `Q_{i}`, `p_r_{i}` for each option i.
    
    Returns:
        (matplotlib.figure, matplotlib.axes)
    """

    # Get number of options in the data
    n_options = np.sum(data.columns.str.startswith("Q_"))
    options = range(n_options)

    # Set up the figure
    n_panels = 3 + n_options  # choice, reward, Q_i / p_r_i, state
    fig, axs = plt.subplots(
        n_panels, 1, figsize=cm2inch(n_panels * 3, 7.5), sharex=True
    )

    # Choice (a_t)
    axs[0].plot(data["a"], "ok")
    axs[0].set_ylabel("Choice\n$(a_t)$")

    # Reward
    axs[1].plot(data["r"], "ok")
    axs[1].set_ylabel("Reward\n$(r_t)$")

    # The agent's action values and the tasks true reward probabilities
    for ax, option in zip(axs[2:-1], options):
        ax.plot(data[f"Q_{option}"], color=f"C{option}")
        ax.plot(data[f"ev_{option}"], "--", color="gray", alpha=0.5, zorder=-1)
        ax.axhline(0, color="black", linewidth=0.5, zorder=-2)
        ax.set_ylabel(f"Value {option}\n$(Q_{{a_t}}$)")

    # Task state
    axs[-1].plot(data["state"], color="black")
    axs[-1].set_yticks(data["state"].unique())
    axs[-1].set_ylim(data["state"].min() - 0.5, data["state"].max() + 0.5)
    axs[-1].set_ylabel("State\n$(s_t)$")

    axs[-1].set_xlabel("Trial")

    return fig, axs


def plot_recovery_results(
    recovery_results, variable_parameter, limits=None, ticks=None
):
    """This function plots the results of a parameter recovery for a single variable parameter.

    Args:
        recovery_results (pandas.DataFrame): Recovery results DataFrame, generated by `rl.recovery.run_recovery`
        variable_parameter (str): Name of the `variable` parameter
        limits (tupl, list, optional): x and y limits of the individual axes. Defaults to None.
        ticks (list, optional): x and y ticks for the individual axes. Defaults to None.

    Returns:
        [(matplotlib.figure, matplotlib.axes)]: Plot figure and axes
    """

    fig, axs = plt.subplots(3, 3, figsize=cm2inch(9, 9), sharex=True, sharey=True)

    parameter_names = ["alpha_pos", "alpha_neg", "beta"]
    parameter_labels = {
        "alpha_pos": r"$\alpha_{+}$",
        "alpha_neg": r"$\alpha_{-}$",
        "beta": r"$\beta$",
    }

    others = [p for p in parameter_names if p != variable_parameter]

    df = recovery_results.loc[recovery_results["variable"] == variable_parameter]

    for i, other0_val in enumerate(["low", "medium", "high"]):
        for j, other1_val in enumerate(["low", "medium", "high"]):
            label = (
                parameter_labels[others[0]]
                + f": {other0_val}"
                + "\n"
                + parameter_labels[others[1]]
                + f": {other1_val}"
            )

            ax = axs[i, j]
            ax.set_title(label)

            df_ij = df.loc[
                (df[f"{others[0]}_level"] == other0_val)
                & (df[f"{others[1]}_level"] == other1_val)
            ]

            # Scatter plot of generating and recovered parameters
            ax.plot(
                df_ij[f"{variable_parameter}"],
                df_ij[f"{variable_parameter}_hat"],
                "o",
                markeredgewidth=0.25,
            )

            # Set ticks and boundaries
            if ticks is not None:
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

            if limits is None:
                limits = [
                    df_ij[
                        [variable_parameter, f"{variable_parameter}_hat"]
                    ].values.min(),
                    df_ij[
                        [variable_parameter, f"{variable_parameter}_hat"]
                    ].values.max(),
                ]
            ax.set_xlim(limits)
            ax.set_ylim(limits)

            # Run robust linear regression and Spearman correlation
            gen = df_ij[variable_parameter].values
            rec = df_ij[f"{variable_parameter}_hat"].values
            r, p = spearmanr(gen, rec)
            endog = rec
            exog = sm.add_constant(gen)
            model = sm.RLM(endog=endog, exog=exog)
            results = model.fit()

            # Plot regression line
            intercept, beta = results.params
            x = np.linspace(*ax.get_xlim(), 100)
            ax.plot(x, intercept + beta * x, color="C0", alpha=0.7, zorder=0)

            # Annotate regression statistics
            pstring = make_pstring(p)
            annotation = (
                f"r = {r:.2f}, "
                + pstring
                + f"\nIntercept = {intercept:.2f}\nSlope = {beta:.2f}"
            )
            ax.annotate(
                annotation,
                (0.95, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=3,
            )

            # Plot diagonal line
            ax.plot(
                ax.get_xlim(),
                ax.get_ylim(),
                "-",
                linewidth=0.5,
                color="lightgray",
                zorder=-1,
            )

    for ij in range(3):
        axs[-1, ij].set_xlabel("Gen.")
        axs[ij, 0].set_ylabel("Rec.")

    fig.tight_layout()
    fig.suptitle(
        r"$\bf{" + parameter_labels[variable_parameter][1:-1] + "}$",
        y=1.05,
        fontweight="bold",
    )

    return fig, axs
