import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class BGNorm(HybridBlock):
    def __init__(self, axis=1, num_groups=32, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, **kwargs):
        super(BGNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        self.axis = axis
        if in_channels != 0:
            self.in_channels = in_channels

        self.momentum = momentum
        self.p = 0.5
        self.eps = epsilon

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('cluster_means', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('cluster_vars', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BGNorm, self).cast(dtype)

    def __call__(self, x):
        return super(BGNorm, self).__call__(x)

    def forward(self, x):
        if autograd.is_training():
            N, _, W, H = x.shape
            self.K = N * W * H
            out, running_mean, running_var = super(BGNorm, self).forward(x)
            with autograd.pause():
                self.running_mean.data(x.context)[:] = running_mean
                self.running_var.data(x.context)[:] = running_var
        else:
            out = super(BGNorm, self).forward(x)
        return out

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        if autograd.is_training():
            x_mean = F.mean(x, axis=self.axis, exclude=True, keepdims=True)
            x_var = 0.5 * (F.mean(F.square(F.broadcast_sub(2 * x, x_mean)),
                                  axis=self.axis, exclude=True, keepdims=True)
                           + F.square(x_mean))
            x = F.broadcast_div(x, F.sqrt(x_var + self.eps))
            running_mean = self.momentum * running_mean + (1 - self.momentum) * running_mean
            running_var = self.momentum * running_var + ((1 - self.momentum) * (self.K / (self.K - 1))) * running_var
            out = F.broadcast_add(F.broadcast_mul(x, gamma.reshape((1, -1, 1, 1))),
                                  beta.reshape((1, -1, 1, 1)))
            return out, running_mean, running_var
        else:
            return F.BatchNorm(x, gamma, beta, running_mean, running_var,
                               name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
