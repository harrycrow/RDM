import torch
import math
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def _qr_retraction(matrix_stack):
    dim = matrix_stack.shape[-1]
    q, r = torch.qr(matrix_stack.cpu())
    d = torch.diagonal(r, offset=0, dim1=1, dim2=2)
    ph = d.sign()
    return q * ph.view(-1, 1, dim)

  
class Radagrad(Optimizer):
    """Implements Riemannian Adagrad algorithm on the Stiefel Manifold with QR retraction.
    """

    def __init__(self, params, dim, lr=1e-1, beta=0.99, lr_decay=0, lap_reg=0, eps=1e-8):
        if not 0.0 <= dim:
            raise ValueError("Invalid embedding dimension: {}".format(dim))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta: {}".format(beta))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= lap_reg:
            raise ValueError("Invalid laplacian regularizer value: {}".format(lap_reg))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(dim=dim, lr=lr, beta=beta, lr_decay=lr_decay, lap_reg=lap_reg, eye=torch.eye(dim), eps=eps)
        super(Radagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros(p.data.shape[0], 1)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                dim = group['dim']
                beta = group['beta']
                eye = group['eye']
                eps = group['eps']
                lap_reg = group['lap_reg']
                square_avg = state['square_avg']
                with torch.no_grad():
                    if grad.is_sparse:
                        grad = grad.coalesce()
                        grad_indices = grad._indices()
                        g = grad._values().view(-1, dim, dim)
                        x = p.data[grad_indices[0]].view(-1, dim, dim)
                        g = g + lap_reg * torch.sign(x - eye)
                        g = g - torch.matmul(torch.matmul(x, torch.transpose(g, 1, 2)), x)
                        sa = square_avg[grad_indices[0]].view(-1)
                        sa_u = g.pow(2).sum(dim=(1, 2))#.sub_(sa).mul_(1 - beta)
                        sa = sa.add_(sa_u).sqrt_().add_(eps).view(-1, 1, 1)
                        x_n = _qr_retraction(x - clr * g / sa)
                        x_u = x_n - x
                        p.data.add_(grad.new(grad_indices, x_u.reshape(-1, dim * dim), grad.size()))
                        square_avg.add_(grad.new(grad_indices, sa_u.view(-1, 1), square_avg.size()))
                    else:
                        g = grad.view(-1, dim, dim)
                        x = p.data.view(-1, dim, dim)
                        g = g + lap_reg * torch.sign(x - eye)
                        g = g - torch.bmm(torch.bmm(x, torch.transpose(g, 1, 2)), x)
                        sa = square_avg.view(-1)
                        sa_u = g.pow(2).sum(dim=(1, 2))#.sub_(sa).mul_(1 - beta)
                        sa = sa.add_(sa_u).sqrt_().add_(eps).view(-1, 1, 1)
                        x_n = _qr_retraction(x - clr * g / sa)
                        x_u = x_n - x
                        p.data.add_(x_u.reshape(-1, dim * dim))
                        square_avg.add_(sa_u.view(-1, 1))
        return loss
