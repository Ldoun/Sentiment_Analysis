from functools import partial
import torch.optim.lr_scheduler as sch

class base():
    def __init__(self, *args):
        pass

    def step(self):
        pass

class LinearAnnealingLR(sch._LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            return [base_lr * self._step_count / self.num_annealing_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.num_total_steps - self._step_count) / (self.num_total_steps - self.num_annealing_steps) for base_lr in self.base_lrs]


def get_sch(scheduler, optimizer, **kwargs):
    if scheduler=='None':
        return base(optimizer)
    elif scheduler=='cosine':
        return sch.SequentialLR(
            optimizer,
            schedulers=[
                LinearAnnealingLR(optimizer, num_annealing_steps=kwargs['warmup_epochs'], num_total_steps=kwargs['warmup_epochs'],),
                sch.CosineAnnealingLR(optimizer, T_max=30)
            ],
            milestones=[kwargs['warmup_epochs']]
        )
    elif scheduler=='warmup':
        return LinearAnnealingLR(optimizer, num_annealing_steps=kwargs['warmup_epochs'], num_total_steps=kwargs['warmup_epochs'],)