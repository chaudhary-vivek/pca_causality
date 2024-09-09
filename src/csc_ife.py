import numpy as np
import scipy.linalg as sla

# matrix left/right division (following MATLAB's convention)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: sla.lstsq(np.array(denom).T, np.array(numer).T)[0].T

class CSC_IFE(object):
    def __init__(self) -> None:
        pass

    def fit(self, df, id, time, outcome, treated, covariates, K, MaxIter=100, Mintol=1e-6, verbose=False):
        """
        df: pd.DataFrame
        id: str, column name of unit id
        time: str, column name of time period
        outcome: str, column name of outcome variable
        treated: str, column name of treatment indicator
        covariates: list of str, column names of covariates
        K: int, number of latent factors
        """
        # fit the model with the given data and parameters
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K

        # gen tr_group and post_period
        df['tr_group'] = df.groupby(id)[treated].transform('max')
        df['post_period'] = df.groupby(time)[treated].transform('max')

        # step 1: estimate F and beta using control units
        # gen X and Y to estimate F, beta and lambda
        Y0, X0 = _prepare_matrix(df.query("tr_group==0"), covariates, id, time, outcome)
        
        # Initialize parameters
        N, T, L = X0.shape
        F0 = np.random.randn(T, K)
        lambda0 = np.random.randn(N, K)
        beta0 = np.zeros(L)

        # estimate F, beta and lambda by ALS
        prev_obj, tol, iter = float('inf'), float('inf'), 0
        while iter < MaxIter and tol > Mintol:
            beta1, lambda1, F1 = als_est(Y0, X0, F0, lambda0, K)

            # compute objective function value
            obj_fun = np.linalg.norm(Y0 - (X0 @ beta1 + lambda1 @ F1.T))**2
            tol = abs(prev_obj - obj_fun)
            # update parameters
            beta0, lambda0, F0 = beta1, lambda1, F1
            prev_obj = obj_fun
            if verbose:
                print(f"Iteration {iter}: tol = {tol}")
            iter += 1

        # step 2: estimate lambda using treated units before treatment
        # gen X and Y to estimate lambda
        Y10, X10 = _prepare_matrix(df.query("tr_group==1 & post_period==0"), covariates, id, time, outcome)
        N_co, T0, L = X10.shape
        numer = F1[:T0, :].T@(Y10 - X10@beta0).T
        denom = F1[:T0, :].T@F1[:T0, :]
        lambda1 = _mldivide(denom, numer).T

        # store the estimated parameters
        self.beta, self.lambda1, self.F = beta1, lambda1, F1

    # step 3: compute Y_syn
    def predict(self):
        """
        Compute the synthetic potential outcomes for treated units.
        """
        # gen X for treated units after treatment
        Y1, X1 = _prepare_matrix(self.df.query("tr_group==1"), self.covariates, self.id, self.time, self.outcome)
        N_tr, T, L = X1.shape
        # compute synthetic potential outcomes
        Y_syn = X1 @ self.beta + self.lambda1 @ self.F.T
        return Y_syn

# define a function to prepare matrices X and Y
def _prepare_matrix(df, covariates, id, time, outcome):
    Y = df.pivot(index=id, columns=time, values=outcome).astype(float).values
    X = np.array([df.pivot(index=id, columns=time, values=x).astype(float).values for x in covariates]).transpose(1, 2, 0)
    return Y, X

# define a function to conduct ALS estimation
def als_est(Y, X, F0, lambda0, K):
    N, T, L = X.shape
    # flatten X and Y
    y = Y.flatten()
    x = X.reshape(N*T, L)

    # compute beta1
    denom = x.T @ x
    numer = x.T @ (y - (lambda0 @ F0.T).flatten())
    beta1 = _mldivide(denom, numer)

    # compute lambda1 and F1
    residual = (y - x @ beta1).reshape(N, T)
    M = (residual.T @ residual) / (N*T)
    s, v, d = sla.svd(M) # singular value decomposition
    F1 = s[:, :K]
    lambda1 = residual @ F1 / T
    return beta1, lambda1, F1

