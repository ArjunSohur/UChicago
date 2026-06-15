import jax.numpy as jnp
import jax
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
mu1, mu2 = 1-0.1, 0.1

def vectorField(state):
    x, y, z, vx, vy, vz = state.T
    r1 = jnp.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = jnp.sqrt((x - mu1)**2 + y**2 + z**2)

    dx = vx
    dy = vy
    dz = vz
    dvx = 2*vy + x - mu1/(r1**3)*(x + mu2) - mu2/(r2**3)*(x - mu1)
    dvy = -2*vx + y - mu1/(r1**3)*y - mu2/(r2**3)*y
    dvz =  - mu1/(r1**3)*z - mu2/(r2**3)*z

    return jnp.array([dx, dy, dz, dvx, dvy, dvz])


def integrate_system(initial_state, t_span, t_eval=None):
    """
    Integrate the system using scipy's solve_ivp.
    
    Args:
        initial_state: Initial condition [x, y, z, vx, vy, vz]
        t_span: Tuple (t0, tf) for integration time span
        t_eval: Time points where solution is evaluated
    
    Returns:
        Solution object from solve_ivp
    """
    def rhs(t, state):
        return np.array(vectorField(jnp.array(state)))
    
    sol = solve_ivp(rhs, t_span, initial_state, t_eval=t_eval, 
                   method='DOP853', rtol=1e-10, atol=1e-12)
    return sol

def energy(state):
    x, y, z, vx, vy, vz = state
    r1 = jnp.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = jnp.sqrt((x - mu1)**2 + y**2 + z**2)
    pot = -(x*x + y*y)/2 - mu1/r1 - mu2/r2 - mu1*mu2/2
    kin = (vx*vx + vy*vy + vz*vz)/2
    return pot + kin 

def rk4_step(state):
    n = 1000
    dt = 1.e-3
    v = vectorField(state)
    x0 = jnp.copy(state)
    batch_size = x0.shape[0]
    orbit = jnp.zeros((n,6,batch_size))
    for i in tqdm(range(n)):
        v = vectorField(x0).T
        v2 = vectorField(x0 + dt/2*v).T
        v3 = vectorField(x0 + dt/2*v2).T
        v4 = vectorField(x0 + dt*v3).T
        x0 = x0 + dt/6*(v + 2*v2 + 2*v3 + v4)
        orbit = orbit.at[i].set(x0.T)
    return orbit
        
def tangent(state,key):
    n = 10000
    dt = 1.e-3
    v = vectorField(state)
    x0 = jnp.copy(state)
    batch_size = x0.shape[0]
    orbit = jnp.zeros((n,6,batch_size))
    u = jnp.zeros((n, 6, 6, batch_size))

    ui = jax.random.normal(key,(batch_size, 6, 6))
    le = jnp.zeros((batch_size,6))
    for i in tqdm(range(n)):
        v = vectorField(x0).T
        dv = jax.vmap(jax.jacfwd(vectorField))(x0)
        x0 = x0 + dt*v
        ui = ui + dt*jax.vmap(lambda A, x: A @ x)(dv, ui)
        ui, nui = jax.vmap(jnp.linalg.qr)(ui)
        le += jnp.log(jax.vmap(lambda A: jnp.abs(jnp.diag(A)))(nui))/n/dt
        orbit = orbit.at[i].set(x0.T)
        u = u.at[i].set(ui.T)
    return le, orbit, u

def plot_orbit_xy(sol, title="Orbit in X-Y Plane"):
    batch_size, _, times = sol.shape
    plt.figure(figsize=(8, 6))
    for i in range(batch_size):
        plt.plot(sol[i,0], sol[i,1], 'k-', linewidth=1.5)
        plt.plot(sol[i,0,0], sol[i,1,0], 'go', markersize=1)
        plt.plot(sol[i,0,-1], sol[i,1,-1], 'ro', markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.show()
    plt.savefig('orbit.png')


def plot_lvs_xy(sol, pert, title="Orbits and perturbations in X-Y plane"):
    batch_size, _, times = sol.shape
    fig, ax = plt.subplots(figsize=(12,8))
    eps = 1.e-1
    for i in range(5):
        ax.plot(sol[i,0], sol[i,1], 'k-', linewidth=1.5)
        xp = sol[i,0,::100]+eps*pert[i,0,0,::100]
        xm = sol[i,0,::100]-eps*pert[i,0,0,::100]
        yp = sol[i,1,::100]+eps*pert[i,1,0,::100]
        ym = sol[i,1,::100]-eps*pert[i,1,0,::100]
        x = jnp.array([xm, xp]).reshape(-1,2).T
        y = jnp.array([ym, yp]).reshape(-1,2).T
        ax.plot(x, y, 'r', lw=1.5)
        ax.plot(sol[i,0,0], sol[i,1,0], 'go', markersize=1)
        ax.plot(sol[i,0,-1], sol[i,1,-1], 'ro', markersize=1)
    ax.plot([-mu2],[0], "rP", markersize=5, label="m1")
    ax.plot([mu1],[0], "bP", markersize=5, label="m2")
    ax.set_xlabel('x',fontsize=24)
    ax.set_ylabel('y',fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=24, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.axis('equal')
    plt.tight_layout()
    #plt.show()
    plt.savefig('lvs.png')

def plot_les(les):
    fig, ax = plt.subplots()
    ax.plot(les, ".", ms=6.0)
    ax.set_xlabel("orbit", fontsize=20)
    ax.set_ylabel("Lyapunov exponent", fontsize=20)
    ax.grid(True)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig('les.png')




if __name__=="__main__":
    batch_size = 100
    rng = jax.random.PRNGKey(42)
    L2_point = jnp.array([1 + mu2, 0, 0, 0, 0, 0])  # L2 Lagrange point
    perturbation_scale = 1e-3
    perturbations = jax.random.normal(rng, (batch_size, 6)) * perturbation_scale
    state = L2_point + perturbations
    #orbit = rk4_step(state).T # batch_size x 6 x n_time
    #ene = energy(orbit[0])
    le, orbit, perts = tangent(state, rng)
    plot_lvs_xy(orbit.T, perts.T)
    plot_les(le)
