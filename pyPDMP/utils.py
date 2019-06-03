import torch
import pysnooper

def arrivalTimePoisson(lambd, t):
    return 1 - torch.exp(lambd*t)

@pysnooper.snoop()
def buildDataset(system, initial_conds, length, steps):
    # TO DO: bugged -__-
    dataset = []
    for x0 in initial_conds:
        sol = system.trajectory(x0, length, steps)
        dataset.append([torch.stack((el.view(-1), torch.zeros(0)), 1) for el in sol])
    dataset.append(torch.stack((el[2].view(-1), torch.ones(1)), 1) for el in system.log)
    return dataset