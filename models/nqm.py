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

    def update(self, lr, batch_size, noise):
        """
        Compute the next iterate based on a noisy gradient query.

        Args:
            lr: the current learning rate
            batch_size: the batch size
            noise: zero-mean isotropic noise
        """
        self.theta = (1 - lr * self.H) * self.theta + \
                     lr * np.sqrt(self.C / batch_size) * noise

    def loss(self):
        """
        Compute the loss at the current iterate.
        """
        return 0.5 * np.sum(self.H * self.theta ** 2)
