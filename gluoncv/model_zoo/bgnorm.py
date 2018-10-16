import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class BGNorm(HybridBlock):
    def __init__(self, axis=1, num_groups=32, momentum=0.9, epsilon=1e-5, center=False, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(BGNorm, self).__init__(**kwargs)
        assert axis == 1
        self.axis = axis
        if in_channels != 0:
            self.in_channels = in_channels
            assert in_channels % num_groups == 0

        self.num_groups = num_groups
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

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BGNorm, self).cast(dtype)

    def hybrid_forward(self, F, x, gamma, beta):
        x_norm = F.norm(x, axis=(2, 3), ord=1, keepdims=True)
        x_norm = F.mean(x_norm.reshape((0, -4, self.num_groups, -1, 0, 0)), axis=2, keepdims=True)
        x = x.reshape((0, -4, self.num_groups, -1, 0, 0))
        #x_norm = F.norm(x, axis=(2, 3), keepdims=True)
        x = F.broadcast_div(x, x_norm + self.eps)
        x = F.broadcast_sub(x, F.mean(x, axis=(0, self.axis), exclude=True, keepdims=True))
        x = x.reshape((0, -3, 0, 0))
        out = F.broadcast_add(F.broadcast_mul(x, gamma.reshape((1, -1, 1, 1))),
                              beta.reshape((1, -1, 1, 1)))
        return out

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
