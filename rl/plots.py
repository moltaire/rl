import matplotlib.pyplot as plt
from rl.plot_utils import cm2inch


def plot_data(data):
    """Plot learning data from a single subject.

    Args:
        data (pd.DataFrame): Dataframe containing the data.
                             Must have the following columns:
                             `trial`, `a`, `r`, `v_a_0`, `v_a_1`, `p_r_0`, `p_r_1`, `rule`
    
    Returns:
        (matplotlib.figure, matplotlib.axes)
    """
    fig, axs = plt.subplots(5, 1, figsize=cm2inch(15, 7.5), sharex=True)

    # Choice (a_t)
    axs[0].plot(data["a"], "ok")
    axs[0].set_ylabel("Choice\n$(a_t)$")
    axs[0].set_ylim(-0.5, 1.5)

    # Reward
    axs[1].plot(data["r"], "ok")
    axs[1].set_ylabel("Reward\n$(r_t)$")
    axs[1].set_ylim(-0.5, 1.5)

    # The agent's action values and the tasks true reward probabilities
    for ax, action in zip([axs[2], axs[3]], [0, 1]):
        ax.plot(data[f"v_a_{action}"], color=f"C{action}")
        ax.plot(data[f"p_r_{action}"], "--", color="gray", alpha=0.5, zorder=-1)
        ax.set_ylabel(f"Value {action}\n$(v_{{a_t}}$)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])

    # Task rule
    axs[-1].plot(data["rule"], color="black")
    axs[-1].set_yticks(data["rule"].unique())
    axs[-1].set_ylim(data["rule"].min() - 0.5, data["rule"].max() + 0.5)
    axs[-1].set_ylabel("Rule\n$(s_t)$")

    axs[-1].set_xlabel("Trial")

    return fig, axs
