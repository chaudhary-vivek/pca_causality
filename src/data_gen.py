import pandas as pd
import numpy as np

def gen_stationary_AR1(n):
    """
    Generate a random n x n matrix A1 for a VAR(1) process
    that satisfies the stationarity condition (all eigenvalues lie inside the unit circle).
    """
    while True:
        # Generate a random matrix
        A1 = np.random.rand(n, n) - 0.5  # Shift to have negative and positive values
        # Check if the generated matrix satisfies the stationarity condition
        eigenvalues = np.linalg.eigvals(A1)
        if np.all(np.abs(eigenvalues) < 1):
            return A1

def data_gen(T0, T1, N_co, N_tr, L, K, drift):  
    """
    Generate data for simulation.
    T0: int, number of time periods before treatment
    T1: int, number of time periods after treatment
    N_co: int, number of control units
    N_tr: int, number of treated units
    L: int, number of covariates
    K: int, number of factors
    drift: int, drift for the treated units in VAR(1) process
    """
    T = T0 + T1
    N = N_co + N_tr

    # gen factors for VAR(1) process
    # assuming a simple structure where each variable depends on its own and the other variable's past value
    # gen variance-covariance matrix
    A1 = gen_stationary_AR1(K)
    # initial values for the first period are drawn from a uniform distribution
    F = np.zeros((K, T))
    F[:, 0] = np.random.uniform(-1, 1, size=K)
    # gen the time series
    for t in range(1, T):
        F[:, t] = A1 @ F[:, t-1] + np.random.normal(0, 1, K)

    # covariates for VAR(1) process
    # assuming a simple structure where each variable depends on its own and the other variable's past value
    # gen variance-covariance matrix
    A2 = np.zeros((N, L, L))
    for i in range(N):
        A2[i] = gen_stationary_AR1(L)
    # initial values for the first period are drawn from a uniform distribution
    X = np.zeros((N, T, L))
    X[:, 0, :] = np.random.uniform(-1, 1, size=(N, L))

    # gen the time series for control units
    for i in range(N_co):
        for t in range(1, T):
             X[i, t, :] = A2[i] @ X[i, t-1, :] + np.random.normal(0, 1, L)
    # gen the time series for treated units with a drift
    for i in range(N_co, N):
        for t in range(1, T):
             X[i, t, :] = A2[i] @ X[i, t-1, :] + np.random.normal(drift, 1, size=L)

    # gen Gama
    Gama = np.random.uniform(-0.1, 0.1, size=(L, K))

    # gen coefficient beta, unit fixed effect alpha, time fixed effect xi
    beta = np.random.uniform(0, 1, L)
    alpha = np.random.uniform(0, 1, N)
    xi = np.random.uniform(0, 1, T)

    # Treatment effects and assignment
    d = np.concatenate([np.zeros(N_co), np.ones(N_tr)])
    delta = np.concatenate([np.zeros(T0), np.arange(1, T1+1) + np.random.normal(0, 1, T1)]) 

    # gen outcome variable
    Y = np.zeros((N, T))
    for t in range(T):
        for i in range(N):
            Y[i, t] += X[i, t, :] @ Gama @ F[:, t] # factors and instrumented factor loadings
            Y[i, t] += X[i, t, :] @ beta # linear effects from covariates
            Y[i, t] += alpha[i] + xi[t] # unit and time fixed effects
            Y[i, t] += d[i] * delta[t] # treatment effect
            Y[i, t] += np.random.normal(0, 1) # noise

    # Construct DataFrame
    df = pd.DataFrame({
        'id': np.repeat(np.arange(101, N + 101), T),
        'time': np.tile(np.arange(1981, T + 1981), N),
        'y': Y.flatten(),
        'tr_group': np.repeat(d, T),
        'post_period': np.tile(np.arange(1, T0+T1+1) > T0, N),
        'eff': np.tile(delta, N)
        })
    # treatement indicator
    df['treated'] = df['tr_group'] * df['post_period']
    # covariates
    for i in range(L):
        df['x' + str(i+1)] = X[:, :, i].flatten()
    
    return df

# we consider two covariates and two factors
def data_gen_xu(T0, T1, N_co, N_tr, L, K, w):
    '''
    N_co: number of control units
    N_tr: number of treated units
    T0: number of pre-treatment periods
    T1: number of post-treatment periods
    L: number of covariates
    K: number of factors
    w: similarity between treated and control units, 1 means identical
    '''
    # Constants
    N = N_co + N_tr
    T = T0 + T1
    ss = np.sqrt(3)

    # gen beta
    beta = np.random.uniform(0, 1, L)
    # gen factor f
    f = np.random.normal(0, 1, size=(K, T))
    # gen time fixed effects xi
    xi = np.random.normal(0, 1, T)

    # gen factor loadings for control and treated units
    lambda_co = np.random.uniform(-ss, ss, size=(K, N_co))
    lambda_tr = np.random.uniform(ss-2*w*ss, 3*ss-2*w*ss, size=(K, N_tr))

    # gen unit fixed effects for control and treated units
    alpha_co = np.random.uniform(-ss, ss, size=N_co)
    alpha_tr = np.random.uniform(ss-2*w*ss, 3*ss-2*w*ss, size=N_tr)

    # combine factor loadings and unit fixed effects
    lambda_ = np.concatenate([lambda_co, lambda_tr], axis=1)
    alpha = np.concatenate([alpha_co, alpha_tr])

    # gen treatment effects and assignment
    d = np.concatenate([np.zeros(N_co), np.ones(N_tr)])
    delta = np.concatenate([np.zeros(T0), np.arange(1, T1+1) + np.random.normal(0, 1, T1)])

    # gen covariates
    X = np.zeros((N, T, L))
    for i in range(L):
        for k in range(K):
            X[:, :, i] = lambda_.T@f + lambda_[k].reshape(-1, 1) + f[k] + np.random.normal(0, 1, size=(N, T)) + 1

    # gen outcome
    y = d.reshape(-1,1)@delta.reshape(1,-1) + X@beta + lambda_.T@f + alpha.reshape(-1,1) + xi + np.random.normal(0, 1, size=(N, T))

    # Construct DataFrame
    df = pd.DataFrame({
        'id': np.repeat(np.arange(101, N + 101), T),
        'time': np.tile(np.arange(1981, T + 1981), N),
        'y': y.flatten(),
        'tr_group': np.repeat(d, T),
        'post_period': np.tile(np.arange(1, T+1)>T0, N),
        'eff': np.tile(delta, N),
        })
    # treatment indicator 
    df['treated'] = df['tr_group']*df['post_period']
    # add covariates
    for i in range(L):
        df['x'+str(i+1)] = X[:,:,i].flatten()
    return df