import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class MultiBatchNorm(HybridBlock):
    """Multi batch normalization layer.
    Normalizes the input at each cluster
    """

    def __init__(self, axis=1, window_size=3, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, **kwargs):
        super(MultiBatchNorm, self).__init__(**kwargs)
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
        self.idx = 0
        self.occupied_size = 0

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(MultiBatchNorm, self).cast(dtype)

    def __call__(self, x):
        return super(MultiBatchNorm, self).__call__(x)

    def forward(self, x):
        if autograd.is_training():
            out, x_mean, x_var,\
            window_mean, window_var,\
            batch_means, batch_vars,\
            running_mean, running_var = super(MultiBatchNorm, self).forward(x)
            self.occupied_size = min(self.occupied_size + 1, self.window_size)
            with autograd.pause():
                self.window_mean.data(x.context)[:] = window_mean
                self.window_var.data(x.context)[:] = window_var
                self.batch_means.data(x.context)[:] = batch_means
                self.batch_vars.data(x.context)[:] = batch_vars
                self.batch_means.data(x.context)[self.idx, :] = x_mean
                self.batch_vars.data(x.context)[self.idx, :] = x_var / self.occupied_size
                self.running_mean.data(x.context)[:] = running_mean
                self.running_var.data(x.context)[:] = running_var

                print(mx.nd.sum(self.window_mean.data(x.context) - mx.nd.sum(self.batch_means.data(), axis=0) / self.occupied_size))
                print(mx.nd.sum(self.window_var.data(x.context) - mx.nd.sum(self.batch_vars.data(), axis=0)))
            self.idx = (self.idx + 1) % self.window_size
        else:
            out = super(MultiBatchNorm, self).forward(x)
        return out

    def hybrid_forward(self, F, x, gamma, beta, window_mean, window_var,
                       batch_means, batch_vars, running_mean, running_var):
        if autograd.is_training():
            N = np.prod(x.shape) / x.shape[self.axis]
            x_mean = F.mean(x, axis=self.axis, exclude=True)
            if self.occupied_size >= self.window_size:
                print('22222222222')
                mean_increment = (x_mean - batch_means[self.idx, :]) / self.window_size
                batch_vars = F.broadcast_add(
                    batch_vars - F.broadcast_mul((2.0 / self.window_size) * mean_increment.reshape((1, -1)),
                                                 F.broadcast_sub(batch_means,
                                                                 window_mean.reshape((1, -1)))),
                    F.square(mean_increment.reshape((1, -1))) / self.window_size)
                window_mean = window_mean + mean_increment
                x_var = F.mean(F.square(F.broadcast_sub(x, window_mean.reshape(self._shape))),
                               axis=self.axis, exclude=True)
                window_var = window_var + F.square(mean_increment) \
                             + x_var / self.window_size - batch_vars[self.idx, :]
                N *= self.window_size
            else:
                print('11111111111')
                mean_increment = (x_mean - window_mean) / (self.occupied_size + 1)
                if self.occupied_size >= 1:
                    batch_vars = F.broadcast_add(
                        batch_vars - F.broadcast_mul((2.0 / self.occupied_size) * mean_increment.reshape((1, -1)),
                                                     F.broadcast_sub(batch_means,
                                                                     window_mean.reshape((1, -1)))),
                        F.square(mean_increment.reshape((1, -1))) / self.occupied_size)
                    batch_vars = batch_vars * (self.occupied_size / (self.occupied_size + 1.0))
                window_mean = window_mean + mean_increment
                x_var = F.mean(F.square(F.broadcast_sub(x, window_mean.reshape(self._shape))),
                               axis=self.axis, exclude=True)
                window_var = (self.occupied_size / (self.occupied_size + 1.0)) * (window_var + F.square(mean_increment)) \
                             + x_var / (self.occupied_size + 1)
                N *= self.occupied_size + 1
            running_mean = self.momentum * running_mean \
                           + (1.0 - self.momentum) * window_mean
            running_var = self.momentum * running_var \
                          + (((1.0 - self.momentum) * N) / (N - 1)) * window_var
            x = F.broadcast_div(F.broadcast_sub(x, window_mean.reshape(self._shape)),
                                F.sqrt(window_var.reshape(self._shape) + self.eps))
            out = F.broadcast_add(F.broadcast_mul(gamma.reshape(self._shape), x),
                                  beta.reshape(self._shape))
            return out, x_mean, x_var, window_mean, window_var, \
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
