import numpy as np
import wandb
from models.nqm import NQM

config = dict(
    batch_size=100,
    num_steps=1000,
    lr=None,
    ndim=10000,
    ema=0.99
)


def get_opt_lr(theta, H, noise):
    """
    Return the learning rate that minimizes the NQM loss for the current gradient query.

    Args:
        theta: the current iterates
        H: the eigenspectrum of the NQM
        noise: the noise from the current gradient query
    """
    top = theta.T @ (H ** 2 * theta) - theta.T @ (H * noise)
    bottom = theta.T @ (H ** 3 * theta) - 2 * (theta.T @ (H ** 2 * noise)) + noise.T @ (H * noise)

    lr = top / bottom
    return lr


def train(config):
    """
    Train the NQM.
    """
    ndim = config['ndim']
    batch_size = config['batch_size']
    ema_decay = config['ema']
    model = NQM(ndim)

    for step in range(1, config['num_steps'] + 1):
        noise = np.sqrt(model.C / batch_size) * np.random.randn(ndim)
        lr = get_opt_lr(model.theta, model.H, noise) if config['lr'] is None else config['lr']

        model.update(lr, batch_size, noise, ema_decay)
        loss = model.loss()
        test_loss = model.loss(ema=True)

        print("step: {0}, train loss: {1:.4f}, test loss {2:.4f}".format(step, loss, test_loss))
        wandb.log({"lr": lr, "loss": loss})


if __name__ == "__main__":
    wandb.init(project="nqm", config=config)
    train(config)
