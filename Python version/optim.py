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


class ProjAdagrad(Optimizer):
    """Implements Adagrad algorithm with projection on the non-negative orthant.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-3, lr_decay=0, weight_decay=15, initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(ProjAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()
                    x = p.data[grad_indices[0]]
                    grad_values = grad_values + x * group['weight_decay']
                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group['eps'])
                    x_n = F.relu(x - clr * grad_values / std_values)
                    x_u = x_n - x
                    p.add_(make_sparse(x_u))
                else:
                    x = p.data
                    grad = grad + x * group['weight_decay']
                    state['sum'].addcmul_(grad, grad, value=1)
                    std = state['sum'].sqrt().add_(group['eps'])
                    x_n = F.relu(x - clr * grad / std)
                    x_u = x_n - x
                    p.add_(x_u)
        return loss

    
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
