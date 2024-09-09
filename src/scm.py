import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.utils.validation import check_X_y

class SCM(object):
    def __int__() -> None:
        pass

    def fit(self, df, id, time, outcome, treated, v):
        """"
        df: pandas dataframe
        id: str, name of the id column
        time: str, name of the time column
        outcome: str, name of the outcome column
        treated: str, name of the treatment column
        v: np.array, weighting matrix v, to assign weight to observations
        """
        # initialize the scm object with given data
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.treated = treated
        self.v = v

        # create post_period and tr_group columns
        df['post_period'] = df.groupby(time)[treated].transform('max')
        df['tr_group'] = df.groupby(id)[treated].transform('max')

        # control group pre-treatment period
        Y00 = df.query('post_period == 0 & tr_group == 0').pivot(index=time, columns=id, values=outcome).values
        # treated group pre-treatment period
        Y10 = df.query('post_period == 0 & tr_group == 1').pivot(index=time, columns=id, values=outcome).mean(axis=1)

        # initial guess for the weights: could be 0, uniform or based on some other logic
        initial_w = (np.ones(Y00.shape[1])/Y00.shape[1])

        if v is None:
            # If no weights are given, use the default weights
            v = np.diag(np.ones(Y00.shape[0])/Y00.shape[0])

        X, y = check_X_y(Y00, Y10)
        # Solve for the weights
        weights = self.solve_weights(X, y, initial_w, v)
        self.weights = weights

    # define a functino to solve for the weights
    def solve_weights(self, X, y, initial_w, v):
        """
        solve for the weights using the given data and initial weights.
        """     
        # define the objective function to minimize: the sum of squares of the residuals
        def fun_obj(w, X, y, v):
            return np.mean(np.sqrt((y - X @ w).T @ v @ (y - X @ w)))
        
        # define the constraints: the weights should sum to 1
        constraints = LinearConstraint(np.ones(X.shape[1]), lb=1, ub=1)

        # define the bounds for the weights: between 0 and 1
        bounds = Bounds(lb=0, ub=1)

        # use the SLSQP method which supports both bounds and constraints
        result = minimize(fun_obj, x0=initial_w, args=(X, y, v), method='SLSQP', constraints=constraints, bounds=bounds)
        return result.x
    
    def predict(self):
        """
        predict the counterfactual using the estimated weights.
        """
        # compute Y_syn
        Y0 = self.df.query('tr_group == 0').pivot(index=self.time, columns=self.id, values=self.outcome).values
        Y_syn = Y0 @ self.weights
        return Y_syn