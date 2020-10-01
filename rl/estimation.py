import numpy as np
from scipy.optimize import minimize

from rl.agent import AgentVars, DualLearningRateAgent


class EstimationVars:
    def __init__(
        self,
        task_vars,
        agent_class,
        parameters,
        bounds,
        fixed_sp=None,
        n_sp=1,
        rand_sp=True,
    ):
        """This function initializes variables for estimation.

        Args:
            task_vars (rl.task.TaskVars): Task variables, needed to set number of trials and blocks.
            agent_class (class): The agent class to estimate parameters for. For example `rl.agent.DualLearningRateAgent`
            parameters (list): List of parameter names to include in estimation.
            bounds (dict): Dictionary of tuples, indicating bounds for each parameter. Bounds are used to determine random starting points, and to constrain optimization procedure.
            fixed_sp (dict, optional): Dictionary of floats, indicating fixed starting points for each parameter. Used if `rand_sp` is True.
            n_sp (int, optional): Number of starting points. Defaults to 1.
            rand_sp (bool, optional): Toggle use of random starting points. If False, fixed starting points are used. Defaults to True.
        """
        self.n_trials = task_vars.n_trials
        self.n_blocks = task_vars.n_blocks
        self.n_options = task_vars.n_options
        self.n_states = task_vars.n_states
        self.agent_class = agent_class
        self.n_sp = n_sp
        self.parameters = parameters
        self.n_params = len(parameters)
        self.bounds = bounds
        self.fixed_sp = fixed_sp
        self.rand_sp = rand_sp


class Estimation:
    def __init__(self, est_vars):
        self.est_vars = est_vars

    def nll(self, x, data, agent_vars):
        """This function computes the negative log-likelihood of choices

        Args:
            x (np.array): Free parameters
            data (pd.DataFrame): DataFrame containing choice (`a`) and reward (`r`) data.
            agent (rl.agent.AgentVars): Agent variables.
        """

        # Name parameter inputs
        parameters = {self.est_vars.parameters[i]: x[i] for i, v in enumerate(x)}

        # Initialize likelihood array
        nll_a = np.full([self.est_vars.n_trials, self.est_vars.n_blocks], np.nan)

        # Cycle over blocks
        for b in range(self.est_vars.n_blocks):

            # Extract required data
            a = data[data["block"] == b]["a"].values
            r = data[data["block"] == b]["r"].values
            s = data[data["block"] == b]["s"].values

            # Assign current parameters
            agent_vars.update(**parameters)

            # Agent initialization
            agent = self.est_vars.agent_class(
                agent_vars=agent_vars,
                n_options=self.est_vars.n_options,
                n_states=self.est_vars.n_states,
            )

            # Initialize block specific variables
            cp = np.full(self.est_vars.n_trials, np.nan)  # choice probability

            # Cycle over trials
            for t in range(0, self.est_vars.n_trials):

                # Skip the whole trial if no action was made
                if np.isnan(a[t]):
                    continue

                # Set observed state
                agent.s_t = np.int(s[t])

                # Evaluate probability of economic decisions
                agent.decide()

                # Set decision
                agent.a_t = np.int(a[t])

                # Extract probability of current decision
                cp[t] = agent.p_a_t[np.int(a[t])]

                # Compute log likelihood of economic decision
                nll_a[t, b] = np.log(agent.p_a_t[np.int(a[t])])

                # Agent contingency parameter update
                agent.learn(r[t])

        # Sum negative log likelihoods
        nll = -1 * np.nansum(nll_a)

        return nll

    def estimate(self, data, agent_vars=None, seed=None):

        # Set RNG seed
        np.random.seed(seed)

        if agent_vars is None:
            agent_vars = AgentVars()

        # Initialize values
        min_nll = np.inf  # "Best" initial nll, overwritten by first estimation run
        min_x = np.nan  # And no parameter estimate

        # Cycle over starting points
        for r in range(self.est_vars.n_sp):

            # Estimate parameters
            # -------------------

            # Make a list of parameter bound tuples
            bounds = [
                self.est_vars.bounds[parameter]
                for parameter in self.est_vars.parameters
            ]

            # Set starting points
            if self.est_vars.rand_sp:
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            else:
                x0 = [
                    self.est_vars.fixed_sp[parameter]
                    for parameter in self.est_vars.parameters
                ]

            # Run the estimation
            result = minimize(
                self.nll, x0, args=(data, agent_vars), method="L-BFGS-B", bounds=bounds
            )

            # Extract maximum likelihood parameter estimate
            x = result.x

            # Extract minimized negative log likelihood
            nll = result.fun

            # Check if cumulated negative log likelihood is lower than the previous
            # one and select the lowest
            if nll < min_nll:
                min_nll = nll
                min_x = x

        # Compute BIC for economic decision
        bic = self.compute_bic(min_nll, self.est_vars.n_params)

        return min_nll, bic, min_x.tolist()

    def compute_bic(self, nll, n_params):
        """ This function compute the Bayesian information criterion (BIC)
            
        Args:
            nll (float): Minimized negative log likelihood
            n_params (int): Number of free parameters
        
        Returns:
            float: BIC
        """
        N = self.est_vars.n_trials * self.est_vars.n_blocks
        LL = -nll
        BIC = -2 * LL + n_params * np.log(N)
        return BIC
