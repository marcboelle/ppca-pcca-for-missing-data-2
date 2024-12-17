import numpy as np
from scipy.linalg import eigh, sqrtm
from tqdm import tqdm

from typing import Optional

class PCA:
    def __init__(self, nb_components):
        """
        Initialize the PCA model.

        Parameters:
        - nb_components: Number of principal components to keep.
        """
        self.nb_components = nb_components 
        self.r2 = None  
        self.mean = None 
        self.components = None 

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        
        X_bis = X.copy()
        X_bis -= self.mean

        covariance = np.cov(X_bis.T)

        eigenvalues, eigenvectors = eigh(covariance)
        decr_eigenvalues, decr_eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        self.components = decr_eigenvectors[:, :self.nb_components] 
        self.r2 = np.sum(decr_eigenvalues[:self.nb_components]) / np.sum(decr_eigenvalues) 

    def transform(self, X):
        return (X - self.mean) @ self.components  
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class PPCA:
    def __init__(self, n_components, tol=1e-6, random_state=None):
        """
        Initialize the PPCA model.

        Parameters:
        - n_components: Number of latent dimensions.
        - max_iter: Maximum number of EM iterations.
        - tol: Convergence tolerance.
        - random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.tol = tol
        self.random_state = random_state
        self.W = None
        self.mu = None
        self.sigma2 = None

    def _estimate_latent_and_missing_values(self, x, W, sigma2, mu):
        obs_idx = ~np.isnan(x)
        missing_idx = np.isnan(x)

        x_obs = x[obs_idx]
        W_obs = W[obs_idx, :]
        mu_obs = mu[obs_idx]

        M_obs = W_obs.T @ W_obs + sigma2 * np.eye(W_obs.shape[1])
        # Compute latent
        z = np.linalg.inv(M_obs) @ W_obs.T @ (x_obs - mu_obs)

        W_miss = W[missing_idx, :]
        mu_miss = mu[missing_idx]
        x_filled = np.copy(x)
        x_filled[missing_idx] = W_miss @ z + mu_miss

        zzT = sigma2 * np.linalg.inv(M_obs) + np.outer(z, z)
        return z, zzT, x_filled

    def fit(self, X, max_iter=100):
        n, d = X.shape
        np.random.seed(self.random_state)
        
        # Initialize parameters
        self.mu = np.nanmean(X, axis=0)
        X_filled = np.copy(X)
        self.W = np.random.randn(d, self.n_components)
        self.sigma2 = 1.0

        for iteration in tqdm(range(max_iter)):
            # E-step
            Z = np.zeros((n, self.n_components))
            zzT = np.zeros((n, self.n_components, self.n_components))

            for i in range(n):
                # Fill missing values and compute estimates of z and zz^T
                z_estimate, zzT_estimate, x_full = self._estimate_latent_and_missing_values(
                    X[i, :], self.W, self.sigma2, self.mu
                )
                Z[i, :] = z_estimate
                zzT[i, :, :] = zzT_estimate
                X_filled[i, :] = x_full

            # M-step: update parameters
            mu_new = np.mean(X_filled, axis=0)
            sum_term1 = sum(np.outer(X_filled[i] - mu_new, Z[i]) for i in range(n))
            sum_term2 = sum(zzT[i] for i in range(n))

            W_new = sum_term1 @ np.linalg.inv(sum_term2)
            sigma2_new = np.sum(
                [
                    np.linalg.norm(X_filled[i] - mu_new) ** 2
                    - 2 * Z[i].T @ W_new.T @ (X_filled[i] - mu_new)
                    + np.trace(zzT[i] @ W_new.T @ W_new)
                    for i in range(n)
                ]
            ) / (n * d)

            # Check for convergence
            if (
                np.linalg.norm(W_new - self.W) < self.tol
                and np.abs(sigma2_new - self.sigma2) < self.tol
            ):
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.W, self.sigma2, self.mu = W_new, sigma2_new, mu_new

        else:
            print(f"Warning: PPCA did not converge after {max_iter} iterations.")
        return self

    def transform(self, X):
        n = X.shape[0]
        Z = np.zeros((n, self.n_components))
        for i in range(n):
            z, _, _ = self._estimate_latent_and_missing_values(X[i, :], self.W, self.sigma2, self.mu)
            Z[i, :] = z
        return Z

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class PPCA_closed_form:
    
    def __init__(self, nb_components: int, sigma2: Optional[float] = None, R: Optional[np.ndarray] = None) -> None:
        """
        PPCA model using closed-form formulas. If sigma2 is not given, it will compute the ML estimator.

        Parameters:
        - nb_components: Number of latent components.
        - sigma2: optional float value, otherwise the ML estimator will be used.
        - R: Optional rotation matrix. If None, the identity is used.
        """
        self.nb_components = nb_components
        self.mean = None
        self.W = None
        self.components = None
        self.sigma2 = sigma2
        self.inv_M = None
        self.R = R 
    
    def fit(self, X: np.ndarray) -> None:
        d = X.shape[1]
        self.mean = np.mean(X, axis=0)
        
        X_bis = X.copy()
        X_bis -= self.mean

        covariance = np.cov(X_bis.T)

        eigenvalues, eigenvectors = eigh(covariance)
        decr_eigenvalues, decr_eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        if self.sigma2 is None:
            self.sigma2 = 1 / (d - self.nb_components) * np.sum(decr_eigenvalues[self.nb_components:])

        diag = np.diag(np.sqrt(decr_eigenvalues[:self.nb_components] - self.sigma2))

        # Add optional rotation, else identity
        if self.R is None:
            self.R = np.eye(self.nb_components)

        self.W = decr_eigenvectors[:, :self.nb_components] @ diag @ self.R
        self.inv_M = np.linalg.inv(self.W.T @ self.W + self.sigma2 * np.eye(self.nb_components))

        self.components = decr_eigenvectors[:, :self.nb_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.W @ self.inv_M.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    


class PCCA:
    def __init__(self, n_components, max_iter=100, tol=1e-6, verbose=False):
        """
        Initialize the PCCA model.
        
        Parameters:
        - n_components: int, Number of latent components.
        - max_iter: int, Maximum number of iterations for EM.
        - tol: float, Convergence tolerance for EM.
        - verbose: bool, Whether to print detailed updates during training.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.W = None
        self.phi = None
        self.M = None
        self.mu_A = None
        self.mu_B = None
        self.sigma = None
        self.UA = None
        self.UB = None

    def fit(self, XA, XB, W_init=None, phi_init=None):
        _, d_A = XA.shape
        d_B = XB.shape[1]
        X = np.hstack((XA, XB))
        
        self.mu_A = np.nanmean(XA, axis=0)
        self.mu_B = np.nanmean(XB, axis=0)

        self.sigma = np.cov(np.hstack((XA, XB)).T, bias=True)

        # Initialize W and phi
        if W_init is None:
            W_init = np.random.randn(d_A + d_B, self.n_components)
        if phi_init is None:
            phi_init = np.eye(d_A + d_B)

        self.W, self.phi = W_init, phi_init
        self.M = np.eye(self.n_components)

        #Initialize UA and UB
        if self.UA is None:
            self.UA = np.random.randn(d_A, self.n_components)
        if self.UB is None:
            self.UB = np.random.randn(d_B, self.n_components)            

        for i in tqdm(range(self.max_iter)):
            # E-Step: Estimate missing values and latent variables
            X_filled = self._fill_missing_values(X, d_A, d_B)
            
            # Update covariance matrix
            self.sigma = np.cov(X_filled.T)

            # Update UA and UB
            self.UA, self.UB = self.update_UA_UB(d_A, d_B)

            # M-Step: Update W and phi
            inv_phi = np.linalg.inv(self.phi)
            M_next = np.linalg.inv(np.eye(self.n_components) + self.W.T @ inv_phi @ self.W)

            W_next = self.sigma @ inv_phi @ self.W @ M_next @ np.linalg.inv(
                M_next + M_next @ self.W.T @ inv_phi @ self.sigma @ inv_phi @ self.W @ M_next
            )
            phi_next = self.sigma - self.sigma @ inv_phi @ self.W @ M_next @ W_next.T
            phi_next[:d_A, d_A:] = 0
            phi_next[d_A:, :d_A] = 0

            # Check convergence
            if np.linalg.norm(W_next - self.W) < self.tol and np.linalg.norm(phi_next - self.phi) < self.tol:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations.")
                break

            self.W, self.phi, self.M = W_next, phi_next, M_next
    
    def update_UA_UB(self, d_A, d_B):
        sigmaA = self.sigma[:d_A, :d_A]
        sigmaB = self.sigma[d_A:, d_A:]
        sigmaAB = self.sigma[:d_A, d_A:]
        sqrt_inv_sigmaA = sqrtm(np.linalg.inv(sigmaA))
        sqrt_inv_sigmaB = sqrtm(np.linalg.inv(sigmaB))

        VA, _, VB_T = np.linalg.svd(sqrt_inv_sigmaA @ sigmaAB @ sqrt_inv_sigmaB)
        UA = sqrt_inv_sigmaA @ VA
        UB = sqrt_inv_sigmaB @ VB_T.T

        return UA, UB

    def _fill_missing_values(self, X, d_A, d_B):
        X_filled = np.copy(X)
        for i in range(X.shape[0]):
            _, _, x_full = self.estimate_latent_and_missing_values(X[i], d_A, d_B)
            X_filled[i] = x_full
        return X_filled

    def estimate_latent_and_missing_values(self, x, d_A, d_B, eps=1e-10):
        xA = x[:d_A]
        xB = x[d_A:]

        UA, UB = self.UA, self.UB

        # Identify observed and missing indices
        obs_idx = ~np.isnan(x)
        obs_idx_A, obs_idx_B = obs_idx[:d_A], obs_idx[d_A:]
        missing_idx = np.isnan(x)
        missing_idx_A, missing_idx_B = missing_idx[:d_A], missing_idx[d_A:]

        # Extract observed data and corresponding matrices
        xA_obs, xB_obs = xA[obs_idx_A], xB[obs_idx_B]
        muA_obs, muB_obs = self.mu_A[obs_idx_A], self.mu_B[obs_idx_B]
        UA_obs, UB_obs = UA[obs_idx_A, :], UB[obs_idx_B, :]

        # Estimate latent variables
        zA = self.M.T @ UA_obs[:, :self.n_components].T @ (xA_obs - muA_obs)
        zB = self.M.T @ UB_obs[:, :self.n_components].T @ (xB_obs - muB_obs)

        # Fill missing values
        x_filled = np.copy(x)
        if np.any(missing_idx_A):
            x_filled[:d_A][missing_idx_A] = self.W[:d_A][missing_idx_A] @ zA.T + self.mu_A[missing_idx_A]
        if np.any(missing_idx_B):
            x_filled[d_A:][missing_idx_B] = self.W[d_A:][missing_idx_B] @ zB.T + self.mu_B[missing_idx_B]

        return zA, zB, x_filled

    def transform(self, XA, XB):
        n_samples = XA.shape[0]
        proj_A, proj_B = [], []
        for i in range(n_samples):
            zA, zB, _ = self.estimate_latent_and_missing_values(
                np.hstack((XA[i], XB[i])), XA.shape[1], XB.shape[1]
            )
            proj_A.append(zA)
            proj_B.append(zB)
        return np.array(proj_A), np.array(proj_B)