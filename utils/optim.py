import math
from collections import Counter
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler

# from .options import get_firstvalue


class TrainOptimizer:
    """
    Contains optimizers and lr schedulers. 
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def create_optimizer(net: torch.nn.Module, optimizer: str, optim_option: dict):
        if not optimizer or optimizer == "Adam":
            lr = optim_option['lr']
            betas = optim_option.get("betas", (0.9, 0.999))
            weight_decay = optim_option.get("weight_decay", 0)
            optimizer_ins = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer == "SGD":
            lr = optim_option['lr']
            momentum = optim_option.get("momentum", 0.9)
            weight_decay = optim_option.get("weight_decay", 0)
            optimizer_ins = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError

        return optimizer_ins

    @staticmethod
    def create_lr_schedulers(optimizer: torch.optim.Optimizer, scheduler: str, scheduler_option: dict):
        if not scheduler :
            return ConstantScheduler()
        elif scheduler == "CosineAnnealingLR":
            total_epoch = scheduler_option['total_epoch']
            eta_min = scheduler_option['eta_min']
            # last_epoch = kwargs['last_epoch']
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_epoch, eta_min=eta_min)
        elif scheduler == "StepLR":
            total_epoch = scheduler_option['total_epoch']
            step_size = scheduler_option['step_size']
            gamma = scheduler_option['gamma']
            # last_epoch = kwargs['last_epoch']
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        elif scheduler == "CosineAnnealingLR_Restart":
            total_epoch = scheduler_option['total_epoch']
            restart_epoch = scheduler_option['restart_epoch']
            eta_min = scheduler_option['eta_min']

            if isinstance(restart_epoch, list):
                T_period = [restart_epoch[0]].extend([restart_epoch[i] - restart_epoch[i-1] for i in range(1, len(restart_epoch))])
                restarts = restart_epoch
                restart_weights = [1] * len(restart_epoch)

            else:
                restart_times = total_epoch // restart_epoch
                T_period = [restart_epoch ] * restart_times
                restarts = [restart_epoch * i for i in range(1, restart_times)]
                restart_weights = [1] * (restart_times - 1)

            scheduler = CosineAnnealingLR_Restart(optimizer, T_period, restart_step=restarts, weights=restart_weights, eta_min=eta_min)
        elif scheduler == "CosineAnnealingLR_WarmUpRestart":
            total_epoch = scheduler_option['total_epoch']
            restart_epoch = scheduler_option['restart_epoch']
            eta_min = scheduler_option.get("eta_min", 0)
            warmup_epoch = scheduler_option.get("warmup_epoch", 0)

            scheduler = CosineAnnealingLR_WarmUpRestart(optimizer, total_epoch, restart_epoch, restart_epoch, eta_min=eta_min, warmup_step=warmup_epoch)
        else:
            raise NotImplementedError
        
        return scheduler

    # def optimizer_step(self):
    #     pass

    # def scheduler_step(self):
    #     pass


class ConstantScheduler():
    def __init__(self, **kwargs) -> None:
        pass

    def step(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state: dict):
        pass


## Copy from basicSR: https://github.com/xinntao/BasicSR
# class MultiStepLR_Restart(_LRScheduler):
#     """
#     Restart MultiStepLR scheduler

#     milestones : lr decay at each step. For example, = [50, 100, 150, 200] ;
#     restarts = [t1, t1+t2, t1+t2+t3, ...]: Restart the Scheduler after the step ti ;
#     weights : restart weights at each restart time. 
#     gamma : lr decay rate.
#     """
#     def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
#                  clear_state=False, last_epoch=-1):
#         self.milestones = Counter(milestones)
#         self.gamma = gamma
#         self.clear_state = clear_state
#         self.restarts = restarts if restarts else [0]
#         self.restarts = [v + 1 for v in self.restarts]
#         self.restart_weights = weights if weights else [1]
#         assert len(self.restarts) == len(
#             self.restart_weights), 'restarts and their weights do not match.'
#         super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch in self.restarts:
#             if self.clear_state:
#                 self.optimizer.state = defaultdict(dict)
#             weight = self.restart_weights[self.restarts.index(self.last_epoch)]
#             return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
#         if self.last_epoch not in self.milestones:
#             return [group['lr'] for group in self.optimizer.param_groups]
#         return [
#             group['lr'] * self.gamma**self.milestones[self.last_epoch]
#             for group in self.optimizer.param_groups
#         ]


# class CosineAnnealingLR_Restart(_LRScheduler):
#     """
#     Restart ConsineAnnealingLR scheduler
#     T_period = [t1, t2, t3 t4]: Each Period for CosineAnnealingLR max step;
#     restarts = [t1, t1+t2, t1+t2+t3]: Restart the Scheduler after the step, the step i;
#     weights: restart weights at each restart time. 
#     """
#     def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):

#         self.T_period = T_period
#         self.T_max = self.T_period[0]  # current T period
#         self.eta_min = eta_min
#         self.restarts = restarts if restarts else [0]
#         self.restarts = [v + 1 for v in self.restarts]
#         self.restart_weights = weights if weights else [1]
#         self.last_restart = 0
#         # if isinstance(warmup_step, list):
#         #     self.warmup_steps = warmup_step
#         # else:
#         #     self.warmup_steps = [warmup_step + ti for ti in T_period]
#         assert len(self.restarts) == len(
#             self.restart_weights), 'restarts and their weights do not match.'
#         super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch == 0:
#             return self.base_lrs 
#         elif self.last_epoch in self.restarts:
#             # restarts
#             self.last_restart = self.last_epoch
#             self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
#             weight = self.restart_weights[self.restarts.index(self.last_epoch)]
#             return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
#         elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
#             # At each restart ?
#             return [
#                 group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
#                 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
#             ]
        
#         return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
#                 (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
#                 (group['lr'] - self.eta_min) + self.eta_min
#                 for group in self.optimizer.param_groups]




## version with 
# class CosineAnnealingLR_WarmUpRestart(_LRScheduler):
#     """
#     Restart ConsineAnnealingLR scheduler with linear warmup.

#     Each step is denoted as t=1, ...,Tmax in a cycle.
#         eta = lr * (t / warmup)   if t <= warmup
#         eta = eta_min + (eta_max - eta_min) / 2 * (1 + cos((t-warmup) / (Tmax - warmup)))

#     T_total : Total iteration steps;
#     each_Tmax : The Cosine cycle steps (lr drops to the minimum) ;
#     restart_step: restart at each cycle steps ;
#     eta_min : Minimum eta value;
#     warmup_step : the warmup steps at each restart cycle ;
#     """
#     def __init__(self, optimizer, T_total: int, each_Tmax: int = None, restart_step: int = None, weights=None, eta_min=0, last_epoch=-1, warmup_step : int = 0):
#         assert warmup_step < each_Tmax
#         assert each_Tmax <= T_total

#         self.T_total = T_total
#         self.T_max = each_Tmax if each_Tmax else T_total
#         self.eta_min = eta_min
#         self.restart_step = restart_step if restart_step and restart_step > 0 else T_total
#         self.warmup_step = warmup_step 
#         # self.T_max = self.T_max
        
#         # self.restarts = restart_step if restart_step else [0]
#         # self.restarts = [v + 1 for v in self.restarts]
#         # self.restart_weights = weights if weights else [1]
#         self.last_restart = 0

#         super(CosineAnnealingLR_WarmUpRestart, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch == 0:
#             self.last_restart = self.last_epoch
#             if self.warmup_step == 0:
#                 return self.base_lrs 
#             else:
#                 return [base_lr * (1. / self.warmup_step) for base_lr in self.base_lrs]
#         elif self.last_epoch % self.restart_step == 0:
#             # restart cycle
#             self.last_restart = self.last_epoch
#             if self.warmup_step == 0:
#                 return [group['initial_lr'] for group in self.optimizer.param_groups]
#             else:
#                 rate = 1. / self.warmup_step
#                 return [group['initial_lr'] * rate for group in self.optimizer.param_groups]
#         elif self.last_epoch + 1 - self.last_restart < self.warmup_step:
#             ## linear warmup
#             return [
#                 group['initial_lr'] * ( (self.last_epoch + 1 - self.last_restart) / self.warmup_step )
#                 for group in self.optimizer.param_groups
#             ]
#         elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
#             ## when epoch goes over Tmax or achieve 2T+T (in the trow of cosine function)
#             return [
#                 group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
#                 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
#             ]
#         else:
#             ## cosine annealing 
#             return [
#                 (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
#                 (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
#                 (group['lr'] - self.eta_min) + self.eta_min
#                 for group in self.optimizer.param_groups
#                 ]


class CosineAnnealingLR_WarmUpRestart(_LRScheduler):
    """
    Restart ConsineAnnealingLR scheduler with linear warmup.

    Each step is denoted as t=1, ...,Tmax in a cycle.
        eta = lr * (t / warmup)   if t <= warmup
        eta = eta_min + (eta_max - eta_min) / 2 * (1 + cos((t-warmup) / (Tmax - warmup)))

    T_total : Total iteration steps;
    each_Tmax : The Cosine cycle steps (lr drops to the minimum) ;
    restart_step: restart at each cycle steps ;
    eta_min : Minimum eta value;
    warmup_step : the warmup steps at each restart cycle ;
    """
    def __init__(self, optimizer, T_total: int, each_Tmax: int = None, restart_step: int = None, weights=None, eta_min=0, last_epoch=-1, warmup_step : int = 0):
        assert warmup_step < each_Tmax
        assert each_Tmax <= T_total

        self.T_total = T_total
        self.T_max = each_Tmax if each_Tmax else T_total
        self.warmup_step = warmup_step 
        self.T_max = self.T_max - warmup_step
        self.restart_step = restart_step if restart_step and restart_step > 0 else T_total
        self.eta_min = eta_min
        
        # self.restarts = restart_step if restart_step else [0]
        # self.restarts = [v + 1 for v in self.restarts]
        # self.restart_weights = weights if weights else [1]
        self.last_restart = 0

        super(CosineAnnealingLR_WarmUpRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            ## first step = 0
            self.last_restart = self.last_epoch
            if self.warmup_step == 0:
                return self.base_lrs
            else:
                return [base_lr * (1. / self.warmup_step) for base_lr in self.base_lrs]
        elif self.last_epoch % self.restart_step == 0:
            # restart cycle
            self.last_restart = self.last_epoch
            if self.warmup_step == 0:
                return [group['initial_lr'] for group in self.optimizer.param_groups]
            else:
                rate = 1. / self.warmup_step
                return [group['initial_lr'] * rate for group in self.optimizer.param_groups]
        elif self.last_epoch + 1 - self.last_restart < self.warmup_step:
            ## linear warmup
            return [
                group['initial_lr'] * ( (self.last_epoch + 1 - self.last_restart) / self.warmup_step )
                for group in self.optimizer.param_groups
            ]
        else:
            ## cosine annealing 
            return [
                self.eta_min + 0.5 * (group['initial_lr'] - self.eta_min) * (1 + math.cos((self.last_epoch + 1 - self.last_restart - self.warmup_step) / self.T_max * math.pi) )
                for group in self.optimizer.param_groups
                ]

if __name__ == "__main__":
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=5e-4, weight_decay=0,
                                 betas=(0.9, 0.99))
    import matplotlib.pyplot as plt
    # T_period = [10, 10, 10, 10]
    # restarts = [10, 20, 30]
    # weights = [1, 1, 1]
    # eta_min = 1e-4
    # scheduler = CosineAnnealingLR_Restart(optimizer=optimizer, T_total=T_period, restart_step=restarts, weights = weights, eta_min=eta_min)

    # scheduler = CosineAnnealingLR_WarmUpRestart(optimizer=optimizer, T_total=50, each_Tmax=25, restart_step=25, warmup_step=5, eta_min=1e-7)
    scheduler = CosineAnnealingLR_WarmUpRestart(optimizer=optimizer, T_total=100, each_Tmax=25, restart_step=25, warmup_step=5, eta_min=5e-6)
    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    plt.plot(lrs)

