from torch.optim.lr_scheduler import _LRScheduler
import math

# for common methods, attributes
class CustomLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, max_lr=1e-3, min_lr=1e-5):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.history = {}

        super(CustomLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    # for history
    def append_loss_history(self, loss):
        self.history.setdefault('loss', []).append(loss)

    def append_lr_history(self, new_lr):
        self.history.setdefault('lr', []).append(new_lr)
        self.history.setdefault('iterations', []).append(self._step_count)

# Cyclical learning rates
class CyclicalLR(CustomLR):
    def __init__(self, optimizer, step_size, last_epoch=-1, max_lr=1e-3, min_lr=1e-5, mode="triangular", gamma=0.94):

        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma # for exp_range


        if mode == 'triangular':
            self.scale_fn = lambda x : 1
        elif mode == 'triangular2':
            self.scale_fn = lambda x : 1 / 2 ** (x - 1)
        elif mode == 'exp_range':
            self.scale_fn = lambda x : self.gamma ** (x)
        else:
            raise NotImplementedError

        super(CyclicalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        prg = self._step_count / self.step_size

        cycle = math.floor(1 + prg / 2)
        x = abs(prg - 2 * cycle + 1)
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, 1 - x) * self.scale_fn(cycle)

        self.history.setdefault('cycle', []).append(cycle)
        self.history.setdefault('x', []).append(x)
        self.append_lr_history(new_lr)

        # because these schedulers are not calculated using previous learning rate
        # So all learning rates are same
        return [new_lr] * len(self.base_lrs)


# Finding the optimal learning rate range in https://www.jeremyjordan.me/nn-learning-rate/
class FinderLR(CustomLR):
    def __init__(self, optimizer, last_epoch=-1, max_lr=1e-3, min_lr=1e-5, steps_per_epoch=None, epochs=None):
        self.total_iterations = steps_per_epoch * epochs

        super(FinderLR, self).__init__(optimizer, last_epoch, max_lr, min_lr)

    def get_lr(self):
        x = self._step_count / self.total_iterations
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * x

        self.append_lr_history(new_lr)

        return [new_lr] * len(self.base_lrs)

# Stochastic Gradient Descent with Restarts (SGDR)
class SGDRLR(CustomLR):
    def __init__(self, optimizer, step_size, last_epoch=-1, max_lr=1e-3, min_lr=1e-5):
        self.step_size = step_size

        super(SGDRLR, self).__init__(optimizer, last_epoch, max_lr, min_lr)

    def get_lr(self):
        # the number of steps since the last restart
        curr_step = self._step_count % self.step_size
        new_lr = self.min_lr + (self.max_lr - self.min_lr) / 2 * (1 + math.cos(curr_step / self.step_size * math.pi))

        self.append_lr_history(new_lr)

        return [new_lr] * len(self.base_lrs)