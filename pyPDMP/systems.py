import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Bernoulli, Exponential
from torchdiffeq import odeint



class System(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, t, xi):
        pass



class LinearSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k = kwargs['k']
        self.b = kwargs['b']

    def __call__(self, t, xi):
        n = xi.size()[0]//2
        return torch.cat((xi[n:], -self.k*xi[:n] -self.b*xi[n:]), 0)



class LinearStochasticSystem(LinearSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # parameters for time dependent exponential distribution, probability to jump
        self.lambd = kwargs['lambd']

        # parameters for spatial multivariate gaussian, probability to jump
        self.mu_jump = kwargs['mu_jump']
        self.std_jump = kwargs['std_jump']

        # parameters for spatial multivariate gaussian over jump coordinates
        self.std_s = kwargs['std_s']

        # logger for jump info
        self.jcount = 0
        self.log = []

    def __call__(self, t, xi):
        n = xi.size(0)
        # spatial multivariate gaussian
        m_gauss = MultivariateNormal(self.mu_jump*torch.ones(n), self.std_jump*torch.eye(n))
        # poisson process, probability of arrival at time t
        exp_p = 1 - torch.exp(Exponential(self.lambd).log_prob(t))
        # probabilities of independent samples multiplied together
        p = torch.exp(m_gauss.log_prob(xi)) * exp_p

        # one sample from bernoulli trial
        event = Bernoulli(p).sample([1])
        if event:
            coord_before = xi
            xi = self.jump(xi, t) # flatten resulting sampled location
            coord_after = xi

            # saving jump coordinate info
            self.log_jump(t, coord_before, coord_after)

        # perform regular step
        print(xi)
        return super().__call__(t, xi)

    def log_jump(self, t, coord_before, coord_after):
        self.jcount += 1
        self.log.append([t, coord_before, coord_after])

    def jump(self, xi, t):
        mu = xi + torch.rand(xi.shape)
        std_s = self.std_s*torch.ones(xi.shape)
        # sample one location from spatial jump distribution
        return MultivariateNormal(mu, std_s*torch.eye(xi.size(0))).sample([1]).flatten()


class pSGLD(System):
    '''
    TO DO: not yet working
    '''
    def __init__(self, D, N, stepsize, beta1, lam):
        super().__init__()
        self.stepsize = stepsize
        self.N = N;
        self.D = D;
        self.lam = lam
        self.beta1 = beta1
        self.grad2 = np.zeros(D)

    def __call__(self, t, xi):
        pass

    def reset_preconditioner(self):
        self.grad2 = np.zeros(self.D)

    def update(self, g):
        self.grad2 = g * g * (1 - self.beta1) + self.beta1 * self.grad2
        preconditioner = 1. / (np.sqrt(self.grad2) + self.lam)
        return self.stepsize * preconditioner * g + np.sqrt(
            self.stepsize * 2 / self.N * preconditioner) * np.random.randn(len(g))


if __name__ == "__main__":
    print('Simple test:')
    t = torch.linspace(0, 10, 2000)
    x0 = torch.rand(10)
    m = LinearSystem(k=2, b=2)
    sol = odeint(m, x0, t)
    #print(sol)
    m = LinearStochasticSystem(k=2, b=2, lambd=0.4, mu_jump=0.3, std_jump=0.2, std_s=5)
    x0 = torch.rand(10)
    sol = odeint(m, x0, t)
    #print(sol)
    #print(m.log)
