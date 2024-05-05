import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import numpy as np

from functools import partial


# Calculates the Lagrangian for the double pendulum system.
def lagrangian(q, q_dot, m1, m2, l1, l2, g):
    
    theta1, theta2 = q
    omega1, omega2 = q_dot
    
    # Kinetic Energy
    T1 = 0.5 * m1 * (l1 * omega1)**2
    T2 = 0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2 + 2 * l1 * l2 * omega1 * omega2 * jnp.cos(theta1 - theta2))
    T = T1 + T2
    
    # Potential Energy
    y1 = -l1 * jnp.cos(theta1)
    y2 = y1 - l2 * jnp.cos(theta2)
    V = m1 * g * y1 + m2 * g * y2
    
    return T - V


# Returns the values for the analytical form of the double pendulum system.
def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    
    theta1, theta2, omega1, omega2 = state
    
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(theta1 - theta2)
    a2 = (l1 / l2) * jnp.cos(theta1 - theta2)
    
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (omega2**2) * jnp.sin(theta1 - theta2) - (g / l1) * jnp.sin(theta1)
    f2 = (l1 / l2) * (omega1**2) * jnp.sin(theta1 - theta2) - (g / l2) * jnp.sin(theta2)
    
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    
    return jnp.stack([omega1, omega2, g1, g2])


# Numerically integrate the analytical form of the double pendulum system.
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)


# Convert the Lagrangian into the equation of motion.
def equation_of_motion(lagrangian, state, t=None):
  
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t)) @ (jax.grad(lagrangian, 0)(q, q_t)
         - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    
    return jnp.concatenate([q_t, q_tt])


# Numerically integrate the equation of motion (from the Lagrangian) to get the motion of the system.
def solve_lagrangian(lagrangian, initial_state, **kwargs):
    
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(equation_of_motion, lagrangian), initial_state, **kwargs)
        
    return f(initial_state)


# Keeps the angles between -pi and pi.
def normalize_dp(state):
    
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])


# Runge-Kutta integration step.
def rk4_step(f, x, t, h):
    
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)