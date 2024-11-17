import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Tuple

class HiddenMarkovModel:
    def __init__(self, n_states: int, n_features: int):
        """
        Initialize Hidden Markov Model
        
        Parameters:
        n_states (int): Number of hidden states
        n_features (int): Number of features in observations
        """
        self.n_states = n_states
        self.n_features = n_features
        
        # Initialize parameters
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize model parameters randomly"""
        # Initial state probabilities
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        
        # Transition probabilities
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        
        # Emission parameters (Gaussian)
        self.means = np.random.randn(self.n_states, self.n_features)
        self.covs = np.array([np.eye(self.n_features) for _ in range(self.n_states)])
        
    def emission_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate emission probabilities for an observation
        
        Parameters:
        x (np.ndarray): Observation vector of shape (n_features,)
        
        Returns:
        np.ndarray: Emission probabilities for each state
        """
        return np.array([
            multivariate_normal.pdf(x, mean=self.means[i], cov=self.covs[i])
            for i in range(self.n_states)
        ])
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm implementation
        
        Parameters:
        observations (np.ndarray): Sequence of observations
        
        Returns:
        Tuple[np.ndarray, float]: Forward probabilities and log likelihood
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        emission = self.emission_prob(observations[0])
        alpha[0] = self.pi * emission
        
        # Forward pass
        for t in range(1, T):
            emission = self.emission_prob(observations[t])
            for j in range(self.n_states):
                alpha[t, j] = emission[j] * np.sum(alpha[t-1] * self.A[:, j])
                
        # Scale to prevent numerical underflow
        scaling_factor = np.sum(alpha, axis=1, keepdims=True)
        alpha = alpha / (scaling_factor + 1e-10)
        
        # Calculate log likelihood
        log_likelihood = np.sum(np.log(scaling_factor + 1e-10))
        
        return alpha, log_likelihood
    
    def backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm implementation
        
        Parameters:
        observations (np.ndarray): Sequence of observations
        
        Returns:
        np.ndarray: Backward probabilities
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialize
        beta[-1] = 1
        
        # Backward pass
        for t in range(T-2, -1, -1):
            emission = self.emission_prob(observations[t+1])
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * emission * beta[t+1])
                
        # Scale to prevent numerical underflow
        scaling_factor = np.sum(beta, axis=1, keepdims=True)
        beta = beta / (scaling_factor + 1e-10)
        
        return beta
    
    def baum_welch(self, observations: np.ndarray, max_iter: int = 100, 
                   tol: float = 1e-6) -> List[float]:
        """
        Baum-Welch algorithm for parameter estimation
        
        Parameters:
        observations (np.ndarray): Training sequence
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        
        Returns:
        List[float]: Log likelihoods for each iteration
        """
        log_likelihoods = []
        
        for iteration in range(max_iter):
            # E-step
            alpha, log_likelihood = self.forward(observations)
            beta = self.backward(observations)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                    break
            
            # Calculate posteriors
            gamma = alpha * beta
            gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)
            
            # Calculate xi (transition posteriors)
            xi = np.zeros((len(observations)-1, self.n_states, self.n_states))
            for t in range(len(observations)-1):
                emission = self.emission_prob(observations[t+1])
                numerator = (alpha[t, :, None] * self.A * emission[None, :] * 
                           beta[t+1, None, :])
                denominator = np.sum(numerator)
                xi[t] = numerator / (denominator + 1e-10)
            
            # M-step
            # Update initial state probabilities
            self.pi = gamma[0]
            
            # Update transition probabilities
            self.A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0, keepdims=True).T + 1e-10)
            
            # Update emission parameters
            for j in range(self.n_states):
                gamma_sum = np.sum(gamma[:, j])
                self.means[j] = np.sum(gamma[:, j, None] * observations, axis=0) / (gamma_sum + 1e-10)
                
                diff = observations - self.means[j]
                self.covs[j] = (np.sum(gamma[:, j, None, None] * 
                               np.einsum('ni,nj->nij', diff, diff), axis=0) / 
                               (gamma_sum + 1e-10))
                
                # Add small diagonal term for numerical stability
                self.covs[j] += 1e-6 * np.eye(self.n_features)
        
        return log_likelihoods
    
    def viterbi(self, observations: np.ndarray) -> Tuple[List[int], float]:
        """
        Viterbi algorithm for finding most likely state sequence
        
        Parameters:
        observations (np.ndarray): Sequence of observations
        
        Returns:
        Tuple[List[int], float]: Most likely state sequence and its log probability
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        emission = self.emission_prob(observations[0])
        delta[0] = np.log(self.pi + 1e-10) + np.log(emission + 1e-10)
        
        # Forward pass
        for t in range(1, T):
            emission = self.emission_prob(observations[t])
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + np.log(emission[j] + 1e-10)
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        log_prob = np.max(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states.tolist(), log_prob
    
    def fit(self, observations: np.ndarray, max_iter: int = 100, 
            tol: float = 1e-6) -> List[float]:
        """
        Fit the model to observations
        
        Parameters:
        observations (np.ndarray): Training sequence
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        
        Returns:
        List[float]: Log likelihoods for each iteration
        """
        return self.baum_welch(observations, max_iter, tol)
    
    def score(self, observations: np.ndarray) -> float:
        """
        Calculate log likelihood of observations
        
        Parameters:
        observations (np.ndarray): Sequence of observations
        
        Returns:
        float: Log likelihood
        """
        _, log_likelihood = self.forward(observations)
        return log_likelihood

# Example usage
# def main():
#     # Generate synthetic data
#     np.random.seed(42)
#     n_samples = 100
#     n_features = 2
#     n_states = 3
    
#     # Create true parameters
#     true_means = np.array([[0, 0], [5, 5], [-5, 5]])
#     true_covs = np.array([np.eye(2) for _ in range(n_states)])
    
#     # Generate observations
#     states = np.random.randint(0, n_states, n_samples)
#     observations = np.array([
#         np.random.multivariate_normal(true_means[state], true_covs[state])
#         for state in states
#     ])
    
#     # Create and train model
#     model = HiddenMarkovModel(n_states=n_states, n_features=n_features)
#     log_likelihoods = model.fit(observations)
    
#     # Find most likely state sequence
#     predicted_states, log_prob = model.viterbi(observations)
    
#     print("Training completed!")
#     print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
#     print(f"Number of iterations: {len(log_likelihoods)}")

# if __name__ == "__main__":
#     main()