import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm


def generate_normal_correlated_data(
    N,
    r=0.5,
    mu_a=0,
    mu_b=0,
    sig_a=1,
    sig_b=1,
    lim_a=None,
    lim_b=None,
    tolerance=0.05,
    seed=None,
    verbose=False,
):
    """This function simulates data from two normally distributed variables A and B with given correlation.
    
    Args:
        N (int): Number of data points
        r (float): Desired Pearson correlation
        mu_a, mu_b (float, optional): Mean values of A and B
        sig_a, sig_b (float, optional): Standard deviations of A and B
        lim_a, lim_b (tuple): Limits of A and B
        tolerance (float, optional): Tolerated difference between desired and generated correlation
        seed (int, optional): Random seed
        verbose (bool, optional): Toggle verbosity
    
    Returns:
        numpy.array: N by 2 array with simulated values
    """
    np.random.seed(seed)

    # Construct mean vector
    mu = np.array([mu_a, mu_b])

    # Construct covariance matrix
    sig = np.array([[sig_a ** 2, r * sig_a * sig_b], [r * sig_a * sig_b, sig_b ** 2]])

    # Simulate data
    delta = np.inf  # initialize deviance to target correlation

    i = 1  # count tries

    while delta > tolerance:

        # Draw data from multivariate normal
        data = np.random.multivariate_normal(mean=mu, cov=sig, size=N)

        # Truncate data to limits
        if lim_a is not None:
            data[:, 0] = np.clip(data[:, 0], *lim_a)
        if lim_b is not None:
            data[:, 1] = np.clip(data[:, 1], *lim_b)

        # Compute generated correlation and deviance
        r_gen, p = pearsonr(data[:, 0], data[:, 1])
        delta = np.abs(r - r_gen)

        i += 1

    if verbose:
        print(f"Pearson r = {r_gen:.2f}, p = {p:.4f}")
        print(f"  (It took {i} tries to achieve the desired tolerance.)")

    return data


def generate_correlated_array(source, target_r, seed=None):
    """ This function generates an array of data
    that correlates with a source array to a desired degree.
    
    Args:
        source (numpy.array): The source array that is already known.
        target_r (float): The desired Pearson correlation between the source and target.
        seed (int): Numpy random seed.
        
    Returns:
        target (numpy.array): The target array.

    Source: https://stats.stackexchange.com/a/313138
    """
    np.random.seed(seed)

    # 1. Generate an initial target vector
    _target = np.random.normal(size=source.size)

    # 2. Run OLS regression of _target on source
    endog = _target
    exog = sm.add_constant(source)
    model = sm.OLS(endog, exog)
    results = model.fit()
    resid = results.resid  # residuals

    # 3. Add back a multiple of source to the residuals of the regression
    target = (
        target_r * np.std(resid) * source
        + np.sqrt(1 - target_r ** 2) * np.std(source) * resid
    )

    # 4. Verify the correlation
    r, p = pearsonr(source, target)
    print(f"r = {r:.2f}, p = {p:.4f}")

    return target