import torch
from utils import arrivalTimePoisson
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



class LinearStochasticSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # parameters for time dependent exponential distribution, probability to jump
        self.lambd = kwargs['lambd']

        # parameters for spatial multivariate gaussian, probability to jump
        self.mu_jump = kwargs['mu_jump']
        self.std_jump = kwargs['std_jump']

        # parameters for spatial multivariate gaussian over jump coordinates
        self.std_s = kwargs['std_s']

    def __call__(self, t, xi):
        # check for jump event according to probability p
        p = torch.distributions._multivariate_normal(xi, mu_jump, std_jump)*arrivalTimePoisson(t[i], self.lambd)
        event = np.random.binomial(1, P)
        if event: xi, t = self.jump(xi, t)

        # perform regular step
        return super().step(xi, t)

    def jump(self, xi, t):
        mu = xi + torch.rand(xi.shape)
        std_s = self.std_s*torch.ones(xi.shape)
        return torch.distributions._multivariate_normal(mu[0], std_s), t



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
    print(sol)
    m = LinearStochasticSystem(k=2, b=2, lambd=0.4, mu_jump=1, std_jump=2, std_s=5)
    x0 = torch.rand(10)
    sol = odeint(m, x0, t)
