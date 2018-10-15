import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class BGNorm(HybridBlock):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, axis=1, num_groups=32, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, n=0, N=0, **kwargs):
        super(BGNorm, self).__init__(**kwargs)
        assert axis == 1
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
        self.n = n
        self.N = N

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BGNorm, self).cast(dtype)

    def hybrid_forward(self, F, x, global_mean, global_var, gamma, beta):
        x = x.reshape((0, -4, self.num_groups, -1, 0, 0))
        x_mean = F.mean(x, axis=(0, 1), exclude=True, keepdims=True)
        increment = (1 - self.n / self.N) * (x_mean - global_mean)
        global_mean = global_mean + increment
        x_var = F.sum(F.square(F.broadcast_sub(x, global_mean)), axis=(0, 1), exclude=True, keepdims=True)
        global_var = (global_var + F.square(increment)) * (self.n / self.N) + x_var / self.N
        x = F.broadcast_div(F.broadcast_sub(x, global_mean), F.sqrt(global_var + self.eps))
        x = x.reshape((0, -3, 0, 0))
        out = F.broadcast_add(F.broadcast_mul(x, gamma.reshape((1, -1, 1, 1))),
                              beta.reshape((1, -1, 1, 1)))
        return out, global_mean, global_var

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
