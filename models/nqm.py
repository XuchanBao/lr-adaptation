import numpy as np


class NQM:
    """
    A Noisy Quadratic Model.

    Attributes:
        H: the eigenspectrum
        C: the covariance of the gradient noise for a single query
        theta: the current iterate
    """

    def __init__(self, ndim):
        self.H = 1.0 / np.arange(1, ndim + 1)
        self.C = self.H
        self.theta = np.random.randn(ndim)
        self.ema = self.theta

    def update(self, lr, batch_size, noise, ema_decay=0):
        """
        Compute the next iterate based on a noisy gradient query.
        Maintain an exponential moving average of the iterates.

        Args:
            lr: the current learning rate
            batch_size: the batch size
            noise: zero-mean isotropic noise
        """
        self.theta = (1 - lr * self.H) * self.theta + \
                     lr * np.sqrt(self.C / batch_size) * noise

        self.ema = ema_decay * self.ema + (1 - ema_decay) * self.theta

    def loss(self, ema=False):
        """
        Compute the loss at the current iterate.

        Args:
            ema: if True, computes the loss using the EMA of the iterates
        """
        theta = self.ema if ema else self.theta
        return 0.5 * np.sum(self.H * theta ** 2)
