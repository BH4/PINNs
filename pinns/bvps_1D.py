from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax.experimental.jet import jet
from tqdm import trange


class bvps:
    name = "bvps"

    # solution, initial condition and pde must be overrided
    def u(self):
        raise NotImplementedError

    def pde(self):
        raise NotImplementedError

    # Default: periodic on x
    def loss_bc(self, params):
        x = self.X * self.x_bd
        u = self.u(params, x)
        loss_bc = jnp.mean((u[..., -1] - u[..., 0]) ** 2)
        return loss_bc

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, x):
        loss = (
            self.pde(params, x).mean()
            + self.loss_bc(params)
        )
        return loss

    def train(self, optimizer, domain, key, params, nIter=5 * 10**4):
        print(self.equation)
        X = self.X
        x_L, x_R = self.x_bd
        Nx = domain.size
        state = optimizer.init_state(params, domain)
        loss_log = []

        @jax.jit
        def step(params, state, *args, **kwargs):
            params, state = optimizer.update(params, state, *args, **kwargs)
            return params, state

        for it in (pbar := trange(1, nIter + 1)):
            params, state = step(params, state, domain)
            if it % 100 == 0:
                loss = state.value
                loss_log.append(loss)
                # domain sampling
                key, subkey = jr.split(key, 2)
                domain = X * jr.uniform(subkey, (Nx,), minval=x_L, maxval=x_R)
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
        domain = self.X * jnp.linspace(*self.x_bd, 200)
        pred = self.u(opt_params, domain)
        ax2.plot(domain, pred)
        # im = ax2.imshow(pred, origin="lower", cmap="jet", aspect="auto")
        # ax2.axis("off")
        # fig.colorbar(im)
        if save:
            fig.savefig(dir)
        else:
            fig.show()
        print("Done!")


class poisson(bvps):
    name = "poisson_1D"
    equation = "u_xx + w^2 sin(w x) = 0"
    X = 1.0
    x_bd = jnp.array([-1, 1])

    def __init__(self, omega=jnp.pi):
        self.omega = omega

    def pde(self, params, x):
        _, (_, u_xx) = jet(
            lambda x: self.u(params, x),
            (x,),
            ((jnp.ones(x.shape), jnp.zeros(x.shape)),),
        )

        pde = (u_xx + self.omega**2 * jnp.sin(self.omega*x)) ** 2
        return pde

    def loss_bc(self, params):
        x = self.X * self.x_bd
        # Dirichlet on x
        u = self.u(params, x)
        loss_bc = jnp.mean(u**2)
        return loss_bc


class helmholtz(bvps):
    name = "helmholtz_1D"
    equation = "u_xx + k^2 u = -f(x)"
    X = 1.0
    x_bd = jnp.array([-1, 1])

    def __init__(self, k=jnp.pi, source_location=0.0, source_strength=1.0):
        self.k = k

        self.source_location = source_location
        self.source_strength = source_strength
        self.source_variance = 0.001*self.X

    def f(self, x):
        r_squared = (x-self.source_location)**2
        coefficient = self.source_strength/jnp.sqrt(2*jnp.pi*self.source_variance)
        return coefficient*jnp.exp(-r_squared/(2*self.source_variance))

    def pde(self, params, x):
        u, (_, u_xx) = jet(
            lambda x: self.u(params, x),
            (x,),
            ((jnp.ones(x.shape), jnp.zeros(x.shape)),),
        )

        u_xx_real = u_xx[..., 0]
        u_xx_imag = u_xx[..., 1]
        u_real = u[..., 0]
        u_imag = u[..., 1]

        pde_residual_real = u_xx_real + self.k**2*u_real + self.f(x)
        pde_residual_imag = u_xx_imag + self.k**2*u_imag
        return pde_residual_real**2 + pde_residual_imag**2
        # return jnp.real(pde_residual*jnp.conjugate(pde_residual))

    def loss_bc(self, params):
        # Absorbing boundary conditions
        # u_x = iku(x)
        # k swaps sine on each side

        x = self.X * self.x_bd
        u, u_x = jax.jvp(lambda x: self.u(params, x), (x,), (jnp.ones(x.shape),))

        right_side = (u_x[1, 0]+self.k*u[1, 1])**2 + (u_x[1, 1]-self.k*u[1, 0])**2
        left_side = (u_x[0, 0]-self.k*u[0, 1])**2 + (u_x[0, 1]+self.k*u[0, 0])**2

        loss_bc = (right_side+left_side)/2
        return loss_bc
