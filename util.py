import jax 
import jax.numpy as jnp 
from jax import vmap 
from jax.scipy.linalg import cho_solve
from jax.numpy.linalg import cholesky
from tqdm import tqdm 
from copy import copy, deepcopy 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import cantera 
from concurrent.futures import ProcessPoolExecutor, as_completed


# For performing kernel matrix operations 
def K(X1, X2, kernel_func, kernel_params):
    return vmap(lambda x: vmap(lambda y: kernel_func(x, y, kernel_params))(X2))(X1)

# For batching the training data
def create_batches(X, Y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    
    if shuffle:
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        X = X[indices]
        Y = Y[indices]
    
    # Yield batches
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size, :]
        Y_batch = Y[i:i + batch_size]
        yield X_batch, Y_batch

'''
ADAM Optimization Routine
------------------------------------

I do quite a bit of optimizing in these gosh-darn ml scripts and it would be nice to have an encapsulated script for unconstrained ADAM optimization which I could plug into the whole thing instead of rewriting it each time.
'''
def ADAM(
    loss_func, p,
    keys_to_optimize,
    X=jnp.ones((1,1)), 
    Y=jnp.ones((1,1)),
    constr={},
    batch_size=250,
    epochs=100,
    lr=1e-8,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    shuffle=False,
    max_backoff=50
):
    def contains_nan(val_dict):
        return any(jnp.isnan(x).any() for x in val_dict.values())

    def adam_step(m, v, p, grad, lr, t):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (jnp.sqrt(v_hat) + epsilon)
        p = p - update
        return m, v, p

    def try_adam_step(grad_func, p, m, v, lr, t):
        new_p, new_m, new_v = deepcopy(p), {}, {}
        for key in keys_to_optimize:
            m_k, v_k, p_k = adam_step(m[key], v[key], p[key], grad[key], lr, t)
            if key in constr:
                p_k = constr[key](p_k)
            new_p[key], new_m[key], new_v[key] = p_k, m_k, v_k

        # Keep batch inputs
        new_p['X'], new_p['Y'] = p['X'], p['Y']
        loss, grad_new = grad_func(new_p)
        return loss, grad_new, new_p, new_m, new_v

    # Initialize optimizer states
    m = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}
    v = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}

    grad_func = jax.value_and_grad(loss_func)

    best_loss = jnp.inf
    best_p = deepcopy(p)

    # Breaking up the training data into batches and storing it in the parameters
    p['X'], p['Y'] = X[:batch_size, :], Y[:batch_size]
    _, grad = grad_func(p)

    iterator = tqdm(range(epochs))

    for epoch in iterator:
        for Xbatch, Ybatch in create_batches(X, Y, batch_size, shuffle=shuffle):
            # Setting the X batches 
            p['X'], p['Y'] = Xbatch, Ybatch

            # Making a trial learning rate 
            trial_lr = lr

            # Backing off learning rate in the case of NaNs found 
            for _ in range(max_backoff):
                loss, grad, trial_p, trial_m, trial_v = try_adam_step(
                    grad_func, p, m, v, trial_lr, epoch+1
                )

                if not (jnp.isnan(loss) or contains_nan(grad)):
                    break  # successful step
                trial_lr *= 0.5
            else:
                print("Too many NaNs. Stopping optimization.")
                return best_p  # return best found so far

            if loss < best_loss:
                best_loss, best_p = loss, deepcopy(trial_p)

            p, m, v = trial_p, trial_m, trial_v

            iterator.set_postfix_str(f"Loss: {loss:.5f}, LR: {trial_lr:.2e}")

    return best_p


# Radial basis function kernel 
def rbf(x,y,kernel_params, epsilon = 1e-8):
    assert x.shape[0] == y.shape[0], 'Input vectors have mismatched dimensions!'
    assert kernel_params.shape[0] == x.shape[0]+1, 'Kernel parameters are wrong dimension! '
    h = (x-y).ravel()
    return kernel_params[0]*jnp.exp(-jnp.sum(h**2 / (jnp.abs(kernel_params[1:])+epsilon)))

class SimpleGP:
    def __init__(self, X, Y, kernel, kernel_dim, noise_var = 1e-6, jitter=1e-6):
        # Storing the training data matrices 
        self.X, self.Y = copy(X), copy(Y) 
        self.kernel = kernel 
        self.kernel_dim = kernel_dim 
        # Storing the parameter dictionary object
        self.p = {
            'X':copy(X), 'Y':copy(Y),
            'k_param':jnp.ones(kernel_dim)*0.1, 
            'noise_var':noise_var
        }
        # Storing jitter
        self.jitter = jitter

    # Function for predicting outputs at new inputs 
    def predict(self, Xtest):
        # Unpack necessary parameters 
        noise_var, k_param = self.p['noise_var'], self.p['k_param']
        # Compute kernel matrices
        Ktrain = K(self.X, self.X, self.kernel, k_param) + (noise_var + self.jitter)* jnp.eye(self.X.shape[0])
        Ktest = K(Xtest, self.X, self.kernel, k_param)
        Ktestvar = K(Xtest, Xtest, self.kernel, k_param)
        # Cholesky and GP predictive mean and covariance
        L = cholesky(Ktrain)
        # Compute posterior mean and variance 
        Ymu = Ktest @ cho_solve((L, True), self.Y)
        Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)
        return Ymu, Ycov
    
    # For computing the log-evidence term of a single GP 
    def objective(self, p):
        # Form training kernel matrix 
        Ktrain = K(p['X'], p['X'], self.kernel, p['k_param']) + p['noise_var']* jnp.eye(p['X'].shape[0])
        # Take cholesky factorization 
        L = cholesky(Ktrain)
        # Compute log-determinant of Ktrain 
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        # Compute quadratic term of Ktrain Y.T @ Ktrain^{-1} Y 
        quadratic_term = p['Y'].T @ jax.scipy.linalg.cho_solve((L, True), p['Y']) 
        # Add total loss back out
        return (quadratic_term + logdet).squeeze()
    
    def optimize(self, params_to_optimize = ['k_param'], constr={}, lr = 1e-5, epochs = 1000, beta1 = 0.9, beta2 = 0.999, batch_size = 250, shuffle = False):

        # Optimizing with ADAM 
        self.p = deepcopy(ADAM(
            lambda p: self.objective(p), 
            self.p, params_to_optimize, 
            X=self.X, Y=self.Y, 
            constr=constr,
            batch_size = batch_size,
            epochs=epochs, 
            lr = lr, 
            beta1 = beta1, 
            beta2=beta2,
            shuffle = shuffle
        ))

class Hyperkriging:
    def __init__(self, d, kernel, kernel_dim, jitter = 1e-6):
        # Obtaining constructor parameters 
        self.d, self.kernel, self.jitter = d, kernel, jitter 
        self.K = len(d) 
        self.kernel_dim = kernel_dim

        # Initializing low-fidelity model 
        self.d[0]['model'] = SimpleGP(
                d[0]['X'],
                d[0]['Y'],
                kernel, kernel_dim,
                noise_var = d[0]['noise_var'],
            )

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = SimpleGP(
                features,
                d[level]['Y'],
                kernel, self.kernel_dim + level,
                noise_var = d[level]['noise_var'],
            )

    def predict(self, Xtest, level):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((test_features, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features)
    
    def optimize(self, level, params_to_optimize = ['k_param'], lr = 1e-5, epochs = 1000, beta1 = 0.9, beta2 = 0.999, batch_size = 250, shuffle = False):
        # Optimizing lowest-fidelity model 
        if level == 0:
            self.d[0]['model'].optimize(params_to_optimize=params_to_optimize, lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = copy(features)
            
            # Creating a model trained on this set of features 
            self.d[level]['model'].optimize(params_to_optimize =params_to_optimize, lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)