import numpy as np
import threading
import mxnet as mx
from mxnet import autograd, test_utils
from mxnet.gluon.block import HybridBlock, Block
from mxnet.ndarray import NDArray
from mxnet.gluon import nn


class Norm(Block):
    def __call__(self, x):
        return super(Norm, self).__call__(x)

    def forward(self, x):
        raise NotImplementedError


class ClusterNorm(HybridBlock, Norm):
    """Cluster normalization layer.
    Normalizes the input at each cluster
    """

    def __init__(self, axis=1, num_clusters=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 cluster_mean_initializer='zeros', cluster_variance_initializer='zeros',
                 in_channels=0, **kwargs):
        super(ClusterNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'num_clusters': num_clusters}

        if in_channels != 0:
            self.in_channels = in_channels
        self.eps = epsilon
        self.momentum = momentum

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.cluster_means = self.params.get('cluster_means', grad_req='null',
                                             shape=(num_clusters, in_channels),
                                             init=cluster_mean_initializer,
                                             allow_deferred_init=True,
                                             differentiable=False)
        self.cluster_vars = self.params.get('cluster_vars', grad_req='null',
                                            shape=(num_clusters, in_channels),
                                            init=cluster_variance_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.proj = nn.Dense(num_clusters)

        shape = [0, 1, 1, 1]
        shape[axis] = in_channels
        self._shape = tuple(shape)
        shape[0] = 1
        self._param_shape = tuple(shape)
        self.t = 0

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(ClusterNorm, self).cast(dtype)

    def __call__(self, x):
        return super(ClusterNorm, self).__call__(x)

    def forward(self, x):
        x_mean = mx.nd.mean(x, axis=(0, self._kwargs['axis']), exclude=True, keepdims=True)
        prob = mx.nd.softmax(self.proj(x_mean.reshape((0, -1))))
        if autograd.is_training():
            x_var = mx.nd.mean((x - x_mean)**2, axis=(0, self._kwargs['axis']), exclude=True, keepdims=True)
            prob_sum = mx.nd.sum(prob, axis=0, keepdims=True).transpose()
            weighted_mean = mx.nd.dot(prob, x_mean.reshape((0, -1)), transpose_a=True) / prob_sum
            weighted_var = mx.nd.dot(prob, x_var.reshape((0, -1)), transpose_a=True) / prob_sum
            self.t += 1
            coef = 1. - self.momentum ** self.t
            self.cluster_means.data()[:] = (self.momentum / coef) * self.cluster_means.data() \
                                           + ((1.0 - self.momentum) / coef) * weighted_mean
            self.cluster_vars.data()[:] = (self.momentum / coef) * self.cluster_vars.data() \
                                          + ((1.0 - self.momentum) / coef) * weighted_var
        return super(ClusterNorm, self).forward(x, prob)

    def hybrid_forward(self, F, x, prob, gamma, beta, cluster_means, cluster_vars):
        soft_means = F.dot(prob, cluster_means)
        soft_vars = F.dot(prob, cluster_vars)
        normalized_x = F.broadcast_div(F.broadcast_sub(x, soft_means.reshape(self._shape)),
                                       F.sqrt(soft_vars.reshape(self._shape) + self.eps))
        out = F.broadcast_add(F.broadcast_mul(gamma.reshape(self._param_shape), normalized_x),
                              beta.reshape(self._param_shape))
        return out

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
