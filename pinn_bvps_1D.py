import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
from optax import adam, cosine_decay_schedule

from pinns.bvps_1D import poisson
from pinns.nn import Siren


class PINN(poisson):
    def __init__(self, width=64, depth=5, w0=8.0):
        super().__init__()
        layers = [1] + [width for _ in range(depth - 1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # (Nt, Nx)
        self.u = jax.vmap(self._u, (None, 0))

    def _u(self, params, x):  # scalar function
        inputs = x
        output = self.apply(params, inputs).squeeze()
        return output


pinn = PINN()
init_key, train_key = jr.split(jr.key(0))
init_params = pinn.init(init_key)

nIter = 1 * 10**4
lr = cosine_decay_schedule(1e-04, nIter)
optimizer = jaxopt.OptaxSolver(fun=pinn.loss, opt=adam(lr))

Nx = 128
domain_tr = pinn.X * jnp.linspace(*pinn.x_bd, Nx)


pinn.train(optimizer, domain_tr, train_key, init_params, nIter=nIter)
pinn.drawing(save=True)
