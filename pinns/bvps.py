from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental.jet import jet
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import trange


class bvps:
    name = "bvps"

    # solution, initial condition and pde must be overrided
    def u(self):
        raise NotImplementedError

    def pde(self):
        raise NotImplementedError

    def loss_bc(self, params):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, x, y):
        loss = (
            self.pde(params, x, y).mean()
            + self.loss_bc(params)
        )
        return loss

    def train(self, optimizer, domain, key, params, nIter=5 * 10**4):
        print(self.equation)
        X, Y = self.X, self.Y
        x_L, x_R = self.x_bd
        y_T, y_B = self.y_bd
        Nx, Ny = domain[0].size, domain[1].size
        state = optimizer.init_state(params, *domain)
        loss_log = []

        @jax.jit
        def step(params, state, *args, **kwargs):
            params, state = optimizer.update(params, state, *args, **kwargs)
            return params, state

        for it in (pbar := trange(1, nIter + 1)):
            params, state = step(params, state, *domain)
            if it % 100 == 0:
                loss = state.value
                loss_log.append(loss)
                # domain sampling
                key, *subkey = jr.split(key, 3)
                domain[0] = X * jr.uniform(subkey[0], (Nx,), minval=x_L, maxval=x_R)
                domain[1] = Y * jr.uniform(subkey[1], (Ny,), minval=y_T, maxval=y_B)
                pbar.set_postfix({"pinn loss": f"{loss:.3e}"})

        self.opt_params, self.loss_log = params, loss_log

    def drawing(self, save=True):
        print("Drawing...")
        dir = f"figures/{self.name}"
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        # loss log
        ax1.semilogy(jnp.arange(len(self.loss_log))*100, self.loss_log, label="PINN Loss")
        ax1.set_xlabel("Training steps")
        ax1.set_ylabel("Loss")
        # Solution profile
        opt_params = self.opt_params
        domain = (
            self.X * jnp.linspace(*self.x_bd, 200),
            self.Y * jnp.linspace(*self.y_bd, 200),
        )
        pred = self.u(opt_params, *domain)
        im = ax2.imshow(pred.T, origin="upper", cmap="jet", aspect="auto")
        fig.colorbar(im)
        if save:
            fig.savefig(dir)
        else:
            fig.show()

        dir = f"figures/{self.name}_loss"
        fig, ax1 = plt.subplots(ncols=1, figsize=(6, 5))
        pred = self.pde(opt_params, *domain)
        color_norm = LogNorm(vmin=jnp.min(pred), vmax=jnp.max(pred))
        im = ax1.imshow(pred.T, origin="upper", cmap="jet", aspect="auto", norm=color_norm)
        fig.colorbar(im)
        if save:
            fig.savefig(dir)
        else:
            fig.show()
        print("Done!")


class poisson(bvps):
    name = "poisson"
    equation = "u_xx + u_yy - f(x, y) = 0"
    X = 1.0
    Y = 1.0
    x_bd = jnp.array([-1, 1])
    y_bd = jnp.array([-1, 1])

    def __init__(self, omega=jnp.pi):
        self.omega = omega
        _f = self.f_gaussian

        self.f = jax.vmap(jax.vmap(_f, (0, None), 0), (None, 0), 1)

    def f_sin(self, x, y):
        return -self.omega**2*jnp.sin(self.omega*x)

    def f_gaussian(self, x, y):
        variance = 0.01

        r_squared = x**2+y**2
        coefficient = 1/jnp.sqrt(2*jnp.pi*variance)
        return coefficient*jnp.exp(-r_squared/(2*variance))

    def pde(self, params, x, y):
        _, (_, u_xx) = jet(
            lambda x: self.u(params, x, y),
            (x,),
            ((jnp.ones(x.shape), jnp.zeros(x.shape)),),
        )
        _, (_, u_yy) = jet(
            lambda y: self.u(params, x, y),
            (y,),
            ((jnp.ones(y.shape), jnp.zeros(y.shape)),),
        )

        pde = (u_xx + u_yy - self.f(x, y)) ** 2
        return pde

    def loss_bc(self, params):
        n = 100

        x = self.X * self.x_bd
        y = self.Y * self.y_bd
        x_sample = jnp.linspace(*x, n)
        y_sample = jnp.linspace(*y, n)
        # Dirichlet on x
        u = self.u(params, x, y_sample)
        x_condition = jnp.mean(u**2)

        # Dirichlet on y
        u = self.u(params, x_sample, y)
        y_condition = jnp.mean(u**2)
        # Periodic on y
        # u = self.u(params, x_sample, y)
        # y_condition = jnp.mean((u[..., -1] - u[..., 0]) ** 2)
        # Neumann on y
        #_, u_y = jax.jvp(lambda y: self.u(params, x_sample, y), (y,), (jnp.ones(y.shape),))
        #y_condition = jnp.mean(u_y**2)

        loss_bc = x_condition + y_condition
        return loss_bc

