import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class RegressionModel:
    def __init__(self, x, y, regression_type='ols', create_intercept=True):
        self.x = np.array(x)
        self.y = np.array(y)
        self.regression_type = regression_type
        self.create_intercept = create_intercept
        self.results = {}
        
        # Add intercept if specified
        if self.create_intercept:
            intercept = np.ones((self.x.shape[0], 1))
            self.x = np.hstack((intercept, self.x))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, beta):
        # Calculate gamma (X * beta)
        gamma = self.x.dot(beta)
        
        # Sigmoid function values
        p = self.sigmoid(gamma)
        
        # Calculate the likelihood for each observation and sum them
        likelihood = self.y * np.log(p) + (1 - self.y) * np.log(1 - p)
        
        # Return the negative log-likelihood (since we are minimizing)
        return -np.sum(likelihood)

    def gradient(self, beta):
        # Calculate gamma (X * beta)
        gamma = self.x.dot(beta)
        
        # Sigmoid function values
        p = self.sigmoid(gamma)
        
        # Calculate the gradient (derivative of log-likelihood w.r.t. beta)
        return self.x.T.dot(p - self.y)

    def logistic_regression(self):
        # Initial guess for beta
        beta_init = np.zeros(self.x.shape[1])

        # Optimize using scipy's minimize function
        result = minimize(fun=self.log_likelihood, x0=beta_init, jac=self.gradient, method='BFGS')
        beta_hat = result.x
        self.calculate_standard_errors(beta_hat)
        self.results['Logistic Regression'] = 'success'
    
    def calculate_standard_errors(self, beta_hat):
        # Calculate the variance-covariance matrix for standard errors
        gamma = self.x.dot(beta_hat)
        p = self.sigmoid(gamma)
        var_cov_matrix = np.linalg.inv(self.x.T.dot(np.diag(p * (1 - p))).dot(self.x))
        std_err = np.sqrt(np.diag(var_cov_matrix))
        
        # Store the results for each coefficient
        for i, name in enumerate(['Intercept'] + [f'x{i+1}' for i in range(self.x.shape[1] - 1)]):
            coef = beta_hat[i]
            z_stat = coef / std_err[i]
            p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
            
            self.results[name] = {
                'coefficient': coef,
                'standard_error': std_err[i],
                'z_stat': z_stat,
                'p_value': p_value
            }
    
    def fit_model(self):
        if self.regression_type == 'logit':
            self.logistic_regression()
        else:
            print("Currently, only logistic regression is implemented")

    def summary(self):
        print(f"{'Variable':<15} {'Coefficient':<15} {'Std. Error':<15} {'z-Statistic':<15} {'p-Value':<15}")
        print("-" * 70)
        
        for variable, stats in self.results.items():
            if isinstance(stats, dict):
                print(f"{variable:<15} {stats['coefficient']:<15.4f} {stats['standard_error']:<15.4f} "
                      f"{stats['z_stat']:<15.4f} {stats['p_value']:<15.4f}")

# Example usage
# x = np.array([[0.5, 1.5], [1, 2], [1.5, 1.2]])  # Replace with actual feature data
# y = np.array([0, 1, 1])                         # Replace with actual binary outcomes
# model = RegressionModel(x, y, regression_type='logit')
# model.fit_model()
# model.summary()
