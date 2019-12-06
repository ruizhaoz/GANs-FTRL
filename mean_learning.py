import matplotlib.pyplot as plt
import numpy as np 

# Utility functions
# This saves every step of w and v in a vector respectively.
def plot_weights(w_D, v_G):
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(np.shape(w_D)[0]), w_D, label="discriminator")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(np.arange(np.shape(w_D)[0]), v_G, label="generator")
    plt.legend()
    plt.show()
    
    print("Recovered mean from generator: {}"
          .format(np.mean(v_G, axis=0)))
    print("Last mean from generator: {}".format(v_G[-1]))

# Training functions

def mean_learning_gd(v, T, eta_D, eta_G, lambda_D=0,
                     gamma_D=0, gamma_G=0, train_gen_every=1,
                     Nesterov=False, clip_D=np.Infinity):
    dim = np.shape(v)[0]
    T_aug = T * train_gen_every
    w = 1.4*np.ones((T_aug, dim))
    theta = 1.7*np.ones((T_aug, dim))
    m_D = 1.4*np.ones((T_aug, dim))
    m_G = 1.7*np.ones((T_aug, dim))

    for t in range(1,T_aug):
        # Discriminator
        if Nesterov==True:
            w_ahead = w[t-1] + gamma_D * m_D[t-1]
        else:
            w_ahead = w[t-1]
        m_D[t] = gamma_D * m_D[t-1] + eta_D * (v - theta[t-1])\
                    - 2 * eta_D * lambda_D * w_ahead * (1.0 - 1.0/np.linalg.norm(w_ahead))
        w[t] = np.clip(w[t-1] + m_D[t], -clip_D, clip_D)

        # Generator
        if t % train_gen_every == 0:
            m_G[t] = gamma_G * m_G[t-1] - eta_G * (w[t-1]) 
            theta[t] = theta[t-1] - m_G[t]
        else:
            m_G[t] = m_G[t-1]
            theta[t] = theta[t-1]
    
    return w, theta

def mean_learning_ftrl(v, T, eta_D, eta_G):
    C = 0.1
    dim = np.shape(v)[0]
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim)) 
    for t in range(2,T):
        # Discriminator
        eta = 2*C/np.sqrt(t)
        #w[t] = w[t-1] - 0.5 * (v + v - theta[t-1] - theta[t-2]) - eta * (np.linalg.norm(w[t-1]))**2
        #print((np.linalg.norm(w[t-1]))^2)
        w[t] = w[t-1] - 0.5  *eta_D * (v-theta[t-1]) - 0.5 *eta_D* (v-theta[t-2]) + eta *eta_D* w[t-1]
        #w[t] = w[t-1] + 2 * eta_D * (v-theta[t-1]) - eta_D * (v-theta[t-2])
        # Generator
        theta[t] = theta[t-1] - 0.5*eta_G* w[t-1] - 0.5*eta_G* w[t-2] + eta *eta_G* theta[t-1]
        #theta[t] = theta[t-1] + 2 * eta_G * w[t-1] - eta_G * w[t-2]
    return w, theta


def mean_learning_omd(v, T, eta_D, eta_G):
    dim = np.shape(v)[0]
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim))

    for t in range(2,T):
        # Discriminator
        w[t] = w[t-1] + 2 * eta_D * (v-theta[t-1]) - eta_D * (v-theta[t-2])
        # Generator
        theta[t] = theta[t-1] + 2 * eta_G * w[t-1] - eta_G * w[t-2]
    
    return w, theta

def mean_learning_stochastic_gd(v, T, eta_D, eta_G, minibatch,
                                train_gen_every=1, clip_D=1):
    dim = np.shape(v)[0]
    # Initialize weights and sample
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim))
    xs = np.random.multivariate_normal(mean=v, 
                                       cov = np.eye(dim), 
                                       size=(T, minibatch))
    zs = np.random.multivariate_normal(mean=np.zeros(dim),
                                       cov = np.eye(dim),
                                       size=(T, minibatch))

    # Compute average of minibatches
    bxs = np.mean(xs, axis=1)
    bzs = np.mean(zs, axis=1)

    for t in range(1,T):
        w[t] = np.clip(w[t-1] + eta_D * (bxs[t-1] - bzs[t-1] - theta[t-1]),
                         -clip_D, clip_D)
        if t % train_gen_every == 0:
            theta[t] = theta[t-1] + eta_G * (w[t-1])
        else:
            theta[t] = theta[t-1]
    
    return w, theta

def mean_learning_stochastic_omd(v, T, eta_D, eta_G, minibatch):
    dim = np.shape(v)[0]
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim))
    xs = np.random.multivariate_normal(mean=v,
                                       cov = np.eye(dim),
                                       size=(T, minibatch))
    zs = np.random.multivariate_normal(mean=np.zeros(dim),
                                       cov = np.eye(dim),
                                       size=(T, minibatch))

    bxs = np.mean(xs, axis=1)
    bzs = np.mean(zs, axis=1)

    for t in range(2,T):
        w[t] = w[t-1] + 2 * eta_D * (bxs[t-1] - bzs[t-1] - theta[t-1])\
                - eta_D * (bxs[t-2] - bzs[t-2] - theta[t-2])
        theta[t] = theta[t-1] + 2 * eta_G * (w[t-1]) - eta_G * (w[t-2])
    
    return w, theta

def mean_learning_stochastic_ftrl(v, T, eta_D, eta_G, minibatch):
    dim = np.shape(v)[0]
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim))
    xs = np.random.multivariate_normal(mean=v,
                                       cov = np.eye(dim),
                                       size=(T, minibatch))
    zs = np.random.multivariate_normal(mean=np.zeros(dim),
                                       cov = np.eye(dim),
                                       size=(T, minibatch))

    bxs = np.mean(xs, axis=1)
    bzs = np.mean(zs, axis=1)
    C = 0.5
    for t in range(2,T):
        eta = 2*C/np.sqrt(t)
        w[t] = w[t-1] +  1 * eta_D * (bxs[t-1] - bzs[t-1] - theta[t-1]) +  0.5 * eta_D * (bxs[t-2] - bzs[t-2] - theta[t-2])- eta_D * eta * w[t-1]
        theta[t] = theta[t-1] +  1 * eta_G * (w[t-1]) -  0.5* eta_G * (w[t-2]) + eta_G * eta * (bxs[t-1] - bzs[t-1] - theta[t-1]) 
    return w, theta

def mean_learning_stochastic_ftrl1(v, T, eta_D, eta_G, minibatch):
    dim = np.shape(v)[0]
    w = 1.4*np.ones((T, dim))
    theta = 1.7*np.ones((T, dim))
    xs = np.random.multivariate_normal(mean=v,
                                       cov = np.eye(dim),
                                       size=(T, minibatch))
    zs = np.random.multivariate_normal(mean=np.zeros(dim),
                                       cov = np.eye(dim),
                                       size=(T, minibatch))

    bxs = np.mean(xs, axis=1)
    bzs = np.mean(zs, axis=1)
    C = 0.5
    for t in range(2,T):
        eta = 2*C/np.sqrt(t)
        w[t] = w[t-1] +  0.5 * eta_D * (bxs[t-1] - bzs[t-1] - theta[t-1]) +  0.5 * eta_D * (bxs[t-2] - bzs[t-2] - theta[t-2])- eta_D * eta * w[t-1]
        theta[t] = theta[t-1] +  0.5 * eta_G * (w[t-1]) -  0.5* eta_G * (w[t-2]) + eta_G * eta * (bxs[t-1] - bzs[t-1] - theta[t-1]) 
    return w, theta


# Optimistic SGD Training on Sampled Loss with minibatching=2000, here you can change to different functions and different parameters
w, theta = mean_learning_stochastic_omd(v, 100000, 0.01, 0.01, 2000)

plot_weights(w, theta)