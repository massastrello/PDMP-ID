import torch
import pysnooper
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

def arrivalTimePoisson(lambd, t):
    return 1 - torch.exp(-lambd*t)


def buildDataset(system, initial_conds, length, steps):
    '''
    Builds a dataset of jump and flow points:
    [st
    :param system:
    :param initial_conds:
    :param length:
    :param steps:
    :return:
    '''
    dataset = []
    for x0 in initial_conds:
        sol = system.trajectory(x0, length, steps)
        dataset.append([torch.cat((el.view(-1), torch.zeros(1)), 0) for el in sol])
    dataset.append(torch.cat((el[2].view(-1), torch.ones(1)), 0) for el in system.log)
    return dataset

# Distributions
def CumExpDist(t,tau):
    return 1 - np.exp(-tau*t)

def SpatialExpDist(x,tau):
    return 1- np.exp(-tau*np.linalg.norm(x))

# Jump Map
def jump(x,t):
    mu = (x + 1*np.random.randn(1,2)).tolist()
    Sigma = [[5,0],[0,5]]
    return np.random.multivariate_normal(mu[0],Sigma)

# Hybrid Sys. Solver
def HDSint(max_events, t, fun, x0, mu, Sigma, tau):
    event_counter = 0
    x_tot = [0,0,0]
    x_event = [0,0,0]
    x_reset = [0,0]
    C = [0,0,0]
    t_tot = []
    print('Progress:')
    while event_counter<max_events:
        sol = odeint(fun,x0,t)
        for i in range(len(sol)):
            P = multivariate_normal.pdf(sol[i],mu,Sigma)*CumExpDist(t[i],tau)
            Event = np.random.binomial(1,P)
            if Event:
                flag = 1;
                x0 = jump(sol[i],t[i])
                x_event = np.vstack((x_event,np.hstack((sol[i],t[i]))))
                x_reset = np.vstack((x_reset,x0))
                break
        C = np.vstack((x_tot,np.hstack((sol[:i-1],t[:i-1].reshape(i-1,1)))))
        x_tot = np.vstack((x_tot,np.hstack((sol[:i],t[:i].reshape(i,1)))))
        if event_counter==0:
            t_tot = np.hstack((t_tot,t[:i])) 
        else:
            t_tot = np.hstack((t_tot,t[:i]+t_tot[-1]))
        if flag:
            flag = 0
            event_counter += 1
        else:
            x0 = sol[-1]
        if (100*event_counter)%(max_events*25)==0:
            print((100*event_counter)/max_events)
    return t_tot, x_tot[1:], x_event[1:], x_reset[1:], C[1:]


# "Sfoltisce" the flow data randomly
def thin_flow_samples(C,N):
    tensor = torch.tensor(C)
    perm = torch.randperm(tensor.size(0))
    idx = perm[:N]
    return tensor[idx].numpy()


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, size):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, size), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD