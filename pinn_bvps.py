import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import jaxopt
from optax import adam, cosine_decay_schedule
import pickle

from pinns.bvps import helmholtz
from pinns.nn import Siren


class PINN(helmholtz):
    def __init__(self, width=64, depth=5, w0=8.0):
        super().__init__()
        layers = [2] + [width for _ in range(depth - 1)] + [2]
        self.init, self.apply = Siren(layers, w0)
        # (Nt, Nx)
        self.u = jax.vmap(jax.vmap(self._u, (None, 0, None), 0), (None, None, 0), 1)

    def _u(self, params, x, y):  # scalar function
        inputs = jnp.hstack([x, y])
        output = self.apply(params, inputs).squeeze()
        return output


load = True
pinn = PINN()
init_key, train_key = jr.split(jr.key(0))
init_params = pinn.init(init_key)
loss_log = []
if load:
    with open('final_params.model', 'rb') as file:  # continue from scaterer 1 atm
        init_params, loss_log = pickle.load(file)

nIter = 1 * 10**5
lr = cosine_decay_schedule(6e-04, nIter)
optimizer = jaxopt.OptaxSolver(fun=pinn.loss, opt=adam(lr))

Nx, Ny = 256, 256
domain_tr = [
    pinn.X * jnp.linspace(*pinn.x_bd, Nx),
    pinn.Y * jnp.linspace(*pinn.y_bd, Ny)
]


pinn.train(optimizer, domain_tr, train_key, init_params, loss_log=loss_log, nIter=nIter)
with open('final_params.model', 'wb') as file:
    pickle.dump((pinn.opt_params, pinn.loss_log), file)
pinn.drawing(save=True)
