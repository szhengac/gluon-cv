import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class BlockNorm(HybridBlock):
    """Block normalization layer.
    Normalizes the input at each cluster
    """

    def __init__(self, axis=1, window_size=3, momentum=0.99, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, n_gpus=1, **kwargs):
        super(BlockNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}

        if in_channels != 0:
            self.in_channels = in_channels
        self.axis = axis
        self.eps = epsilon
        self.momentum = momentum
        self.window_size = window_size

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)
        self.window_mean = self.params.get('window_mean', grad_req='null',
                                           shape=(in_channels,),
                                           init='zeros',
                                           allow_deferred_init=True,
                                           differentiable=False)
        self.window_var = self.params.get('window_var', grad_req='null',
                                          shape=(in_channels,),
                                          init='zeros',
                                          allow_deferred_init=True,
                                          differentiable=False)
        self.batch_means = self.params.get('batch_means', grad_req='null',
                                           shape=(window_size, in_channels),
                                           init='zeros',
                                           allow_deferred_init=True,
                                           differentiable=False)
        self.batch_vars = self.params.get('batch_vars', grad_req='null',
                                          shape=(window_size, in_channels),
                                          init='zeros',
                                          allow_deferred_init=True,
                                          differentiable=False)

        shape = [1] * 4
        shape[axis] = in_channels
        self._shape = tuple(shape)
        self.occupied_size = [0] * n_gpus
        self.idx = [0] * n_gpus

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BlockNorm, self).cast(dtype)

    def __call__(self, x):
        return super(BlockNorm, self).__call__(x)

    def forward(self, x):
        if autograd.is_training():
            ctx = x.context
            out, x_mean, x_var, x_varr,\
            window_mean, window_var,\
            batch_means, batch_vars,\
            running_mean, running_var = super(BlockNorm, self).forward(x)
            self.occupied_size[ctx.device_id] = min(self.occupied_size[ctx.device_id] + 1, self.window_size)
            with autograd.pause():
                self.window_mean.data(ctx)[:] = window_mean
                self.window_var.data(ctx)[:] = window_var
                self.batch_vars.data(ctx)[:] = batch_vars
                self.batch_means.data(ctx)[self.idx[ctx.device_id], :] = x_mean
                self.batch_vars.data(ctx)[self.idx[ctx.device_id], :] = x_var / self.occupied_size[ctx.device_id]
                self.running_mean.data(ctx)[:] = running_mean
                self.running_var.data(ctx)[:] = running_var
            self.idx[ctx.device_id] = (self.idx[ctx.device_id] + 1) % self.window_size
        else:
            out = super(MultiBatchNorm, self).forward(x)
        return out

    def hybrid_forward(self, F, x, gamma, beta, window_mean, window_var,
                       batch_means, batch_vars, running_mean, running_var):
        if autograd.is_training():
            N = np.prod(x.shape) / x.shape[self.axis]
            x_mean = F.mean(x, axis=self.axis, exclude=True)
            occupied_size = self.occupied_size[x.context.device_id]
            if occupied_size == self.window_size:
                mean_increment = (x_mean - batch_means[self.idx[x.context.device_id], :]) / self.window_size
                batch_vars = F.broadcast_add(
                    batch_vars - F.broadcast_mul((2.0 / self.window_size) * mean_increment.reshape((1, -1)),
                                                 F.broadcast_sub(batch_means,
                                                                 window_mean.reshape((1, -1)))),
                    F.square(mean_increment.reshape((1, -1))) / self.window_size)
                window_mean = window_mean + mean_increment
                x_var = F.mean(F.square(F.broadcast_sub(x, window_mean.reshape(self._shape))),
                               axis=self.axis, exclude=True)
                window_var = window_var + F.square(mean_increment) + x_var / self.window_size - batch_vars[self.idx[x.context.device_id], :]
                N *= self.window_size
            else:
                mean_increment = (x_mean - window_mean) / (occupied_size + 1)
                if occupied_size >= 1:
                    batch_vars = F.broadcast_add(
                        batch_vars - F.broadcast_mul((2.0 / occupied_size) * mean_increment.reshape((1, -1)),
                                                     F.broadcast_sub(batch_means,
                                                                     window_mean.reshape((1, -1)))),
                        F.square(mean_increment.reshape((1, -1))) / occupied_size)
                    batch_vars = batch_vars * (occupied_size / (occupied_size + 1.0))
                window_mean = window_mean + mean_increment
                x_var = F.mean(F.square(F.broadcast_sub(x, window_mean.reshape(self._shape))),
                               axis=self.axis, exclude=True)
                window_var = (occupied_size / (occupied_size + 1.0)) * (window_var + F.square(mean_increment)) \
                             + x_var / (occupied_size + 1)
                N *= occupied_size + 1
            x_varr = F.mean(F.square(F.broadcast_sub(x, x_mean.reshape(self._shape))),
                            axis=self.axis, exclude=True)
            running_mean = self.momentum * running_mean \
                           + (1.0 - self.momentum) * window_mean
            running_var = self.momentum * running_var \
                          + (((1.0 - self.momentum) * N) / (N - 1)) * window_var
            x = F.broadcast_div(F.broadcast_sub(x, window_mean.reshape(self._shape)),
                                F.sqrt(window_var.reshape(self._shape) + self.eps))
            out = F.broadcast_add(F.broadcast_mul(gamma.reshape(self._shape), x),
                                  beta.reshape(self._shape))
            return out, x_mean, x_var, x_varr, window_mean, window_var, \
                   batch_means, batch_vars, running_mean, running_var
        else:
            return F.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd',
                               **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
