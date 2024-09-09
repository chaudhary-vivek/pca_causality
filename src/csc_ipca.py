import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
from joblib import Parallel, delayed


# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class CSC_IPCA(object):
    def __init__(self) -> None:
        pass

    def fit(self, df, id, time, outcome, treated, covariates, K, MaxIter=100, MinTol=1e-6, verbose=False):
        """
        df: pd.DataFrame
        id: str, column name for unit id
        time: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors
        """
        # Initializes the model with given parameters and data.
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

        # gen Y and X to estimate F and Gama
        Y0, X0 = _prepare_matrix(df.query("tr_group==0"), covariates, id, time, outcome)
        _, _, L = X0.shape

        # initial guess for F0 and Gama0
        svU, svS, svV = ssla.svds(Y0, k=K)
        # reverse the order of singular values and vectors
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV)
        # initial guess for F0
        F0 = np.diag(svS) @ svV
        # initial guess for Gama0
        Gama0 = np.zeros((L, K))

        # estimate F1 and Gama1 by ALS algorithm
        iter, tol = 0, float('inf')
        while iter < MaxIter and tol > MinTol:
            Gama1, F1 = als_est(Y0, X0, K, F0)
            tol_Gama = abs(Gama1 - Gama0).max()
            tol_F = abs(F1 - F0).max()
            tol = max(tol_Gama, tol_F)

            if verbose:
                print('iter {}: tol_Gama: {}, tol_F: {}'.format(iter, tol_Gama, tol_F))
            F0, Gama0 = F1, Gama1
            iter += 1

        # store the estimated F and Gama
        self.F1 = F1
        self.Gama1 = Gama1
    
    def predict(self):
        """
        Predict the counterfactual outcome for treated units
        df: dataframe, should be the treated data
        """

        # gen Y and X from treated data before treatment to estimate Gama_tr
        Y, X = _prepare_matrix(self.df.query("tr_group==1 & post_period==0"), self.covariates, self.id, self.time, self.outcome)

        # estimate Gama_tr for treated units
        Gama_tr = estimate_gama(Y, X, self.F1, self.K)

        # normalize the estimated parameters
        R1 = sla.cholesky(Gama_tr.T @ Gama_tr)
        R2, _, _ = sla.svd(R1 @ self.F1 @ self.F1.T @ R1.T)
        # matrix division
        Gama_tr_norm = _mrdivide(Gama_tr, R1) @ R2
        F1_norm = _mldivide(R2, R1 @ self.F1)

        # compute counterfactual for treated units all time periods
        _, X = _prepare_matrix(self.df.query("tr_group==1"), self.covariates, self.id, self.time, self.outcome)
        Y_syn = compute_syn(X, Gama_tr_norm, F1_norm)
        self.Gama = Gama_tr_norm
        self.F = F1_norm
                
        return Y_syn
    
    def inference(self, nulls, alpha, n_jobs=-1, verbose=False):
        """
        Conduct conformal inference.
        n_jobs: Number of jobs to run in parallel. Default is -1, using all CPUs.
        """
        def estimate_ci_for_period(period):
            ci = confidence_interval_period(
                df=self.df,
                id=self.id,
                time=self.time, 
                outcome=self.outcome, 
                treated=self.treated, 
                covariates=self.covariates, 
                nulls=nulls, 
                K=self.K,
                period=period,
                alpha=alpha)

            if verbose:
                print(f'Estimation for period {period} is completed!')
            return ci

        # Determine the range of periods for inference.
        treatment_periods = self.df[self.df[self.treated] == 1][self.time].unique()
        start_period, end_period = treatment_periods.min(), treatment_periods.max()

        # Use parallel computation for each period.
        confidence_intervals = Parallel(n_jobs=n_jobs)(
            delayed(estimate_ci_for_period)(period)
            for period in range(start_period, end_period + 1)
        )

        # Combine results into a single DataFrame.
        ci_df = pd.concat(confidence_intervals, axis=0)

        return ci_df

##############################################
# define a function to prepare matrix
def _prepare_matrix(df, covariates, id, time, outcome):
    Y = df.pivot(index=id, columns=time, values=outcome).astype(float).values
    X = np.array([df.pivot(index=id, columns=time, values=x).astype(float).values for x in covariates]).transpose(1, 2, 0)    
    return Y, X

# define a function to conduct ALS estimation
def als_est(Y, X, K, F0):
    N, T, L = X.shape
    # with F fixed, estimate Gama
    vec_len = L*K
    numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
    for t in range(T):
        for i in range(N):
            # slice X and F
            X_slice = X[i, t, :]
            F_slice = F0[:, t]
            # compute kronecker product
            kron_prod = np.kron(X_slice, F_slice)
            # update numer and denom
            numer += kron_prod * Y[i, t]
            denom += np.outer(kron_prod, kron_prod)
    # solve for Gama
    Gama1 = _mldivide(denom, numer).reshape(L, K)

    # with Gama fixed, estimate F
    F1 = np.zeros((K, T))
    for t in range(T):
        denom = Gama1.T@X[:,t,:].T@X[:,t,:]@Gama1
        numer = Gama1.T@X[:,t,:].T@Y[:,t]
        F1[:, t] = _mldivide(denom, numer)
    return Gama1, F1

# define a function to compute Gama for treated units
def estimate_gama(Y, X, F1, K):
    N, T, L = X.shape
    # with F fixed, estimate Gama
    vec_len = L*K
    numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
    for t in range(T):
        for i in range(N):
            X_slice = X[i, t, :]
            F_slice = F1[:, t]
            kron_prod = np.kron(X_slice, F_slice)
            # update numer and denom
            numer += kron_prod * Y[i, t]
            denom += np.outer(kron_prod, kron_prod)
    # solve for Gama
    Gama1 = _mldivide(denom, numer).reshape(L, K)
    return Gama1

# build a function to predict the synthetic control
def compute_syn(X, Gama, F):
    N, T, L = X.shape
    Y_hat = np.zeros((N, T))
    for t in range (T):
        for i in range(N):
            Y_hat[i, t] = X[i, t, :] @ Gama @ F[:, t]
    return Y_hat

# define a function to generate the null
def under_null(df, null, treated, outcome):
    data = df.copy()
    y = np.where(data[treated]==1, data[outcome] - null, data[outcome])
    return data.assign(**{outcome: y})

# define a function to compute F and Gama
def update_parameter(Y, X, K, MaxIter=100, MinTol=1e-6, verbose=False):
    _, _, L = X.shape
    # initial guess
    svU, svS, svV = ssla.svds(Y, K)
    svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV)
    # initial guess for F
    F0 = np.diag(svS) @ svV
    # initial guess for Gama
    Gama0 = np.zeros((L, K))

    # iteratively update F and Gama
    tol, iter = float('inf'), 0
    while tol > MinTol and iter < MaxIter:
        Gama1, F1 = als_est(Y, X, K, F0)
        tol_Gama = abs(Gama1 - Gama0).max()
        tol_F = abs(F1 - F0).max() 
        tol = max(tol_Gama, tol_F)
        if verbose:
            print(f'iter: {iter}, tol_Gama: {tol_Gama}, tol_F: {tol_F}')
        F0, Gama0 = F1, Gama1
        iter += 1
    return F1, Gama1

# define a function to compute test statistic
def test_statistic(u_hat, q=1, axis=0):
    return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)

# build a function to compute p value
def compute_pvalue(y, yhat, window):
    residual = y - yhat
    block_permutations = np.stack([np.roll(residual, permutation, axis=0)[-window:] for permutation in range(len(residual))])

    test_stat = test_statistic(block_permutations, q=1, axis=1)
    p_val = np.mean(test_stat >= test_stat[0])
    return p_val

# grid search the p value under different null hypothesis
def pval_grid(df, id, time, outcome, treated, covariates, nulls, K):
    # build a dic to store p value
    p_vals = {}
    for null in nulls:
        # assign the null
        null_df = under_null(df, null, treated, outcome)

        # prepare the matrix for control units
        Y0, X0 = _prepare_matrix(null_df[null_df['tr_group']==0], covariates, id, time, outcome)
        # estimate F
        F, _ = update_parameter(Y0, X0, K)

        # prepare the matrix for treated units
        Y1, X1 = _prepare_matrix(null_df[null_df['tr_group']==1], covariates, id, time, outcome)
        # estimate Gama
        Gama_tr = estimate_gama(Y1, X1, F, K)

        # compute synthetic control
        Y_hat = compute_syn(X1, Gama_tr, F).mean(axis=0)

        # compute residual
        Y_mean = null_df[null_df['tr_group']==1].groupby(time)[outcome].mean()
        # compute p value
        p_val = compute_pvalue(Y_mean, Y_hat, window=1) # window = 1 for period by period estimation
        p_vals[null] = p_val
    return p_vals

# define a function to compute confidence interval
def confidence_interval(p_vals, alpha):
    big_p_vals = p_vals[p_vals.values >= alpha]
    return pd.DataFrame({
        f"{int(100-alpha*100)}_ci_lower": big_p_vals.index.min(),
        f"{int(100-alpha*100)}_ci_upper": big_p_vals.index.max()
    }, index=[p_vals.columns[0]])

# define a function to compute the confidence interval period by period
def confidence_interval_period(df, id, time, outcome, treated, covariates, nulls, K, period, alpha):
    # append the targeting period to the pre-treatment period
    df_aug = df[(df['post_period']==0) | (df[time]==period)]
    
    # grid search p values under different null hypothesis
    p_vals = pval_grid(df_aug, id, time, outcome, treated, covariates, nulls, K)
    # covert into dataframe
    p_vals = pd.DataFrame(p_vals, index=[period]).T

    # compute confidence interval
    ci = confidence_interval(p_vals, alpha=alpha)
    return ci
