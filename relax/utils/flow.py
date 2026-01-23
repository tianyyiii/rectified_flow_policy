from typing import Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class FlowModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

class MeanFlowModel(Protocol):
    def __call__(self, x: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        ...

class ShortCutModel(Protocol):
    def __call__(self, x: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        ...

class LagrangianModel(Protocol):
    def __call__(self, x: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class OTFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        #x = jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: float, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()

@dataclass(frozen=True)
class MeanFlow:
    num_timesteps: int

    # We only consider one-step sampling in the experiments
    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = (self.num_timesteps - t) * dt
            drift = model(x, tau - dt, tau)
            x_next = x - drift
            # x_next = x - drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: float, x_start: jax.Array, noise: jax.Array):
        return (1 - t) * x_start + t * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert r.ndim == 1 and t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v = noise - x_start
        zero_r = jnp.zeros_like(r, dtype=jnp.float32)
        one_t  = jnp.ones_like(t, dtype=jnp.float32)
        u_pred, dudt = jax.jvp(model, (x_t, r, t), (v, zero_r, one_t))
        u_tgt = jax.lax.stop_gradient(v - (t - r)[:, None] * dudt)
        loss = weights * optax.squared_error(u_pred, u_tgt)
        return loss.mean()
    
@dataclass(frozen=True)
class ShortCutModel:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(x, tau, tau + dt)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: float, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def p_map(self, model: MeanFlowModel, r: jax.Array, t: jax.Array, x_r: jax.Array):
        if t.ndim == 1 and x_r.ndim == 2:
            return x_r + (t - r)[:, None] * model(x_r, r, t)
        else:
            return x_r + (t - r) * model(x_r, r, t)

    def weighted_pb_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(x_t, t, t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()
    
    def weighted_pd_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_r = jax.vmap(self.q_sample)(r, x_start, noise)
        x_pred = self.p_map(model, r, t, x_r)
        x_half = self.p_map(model, r, (r + t) / 2, x_r)
        x_tgt = self.p_map(model, (r + t) / 2, t, x_half)
        x_tgt = jax.lax.stop_gradient(x_tgt)
        loss = weights * optax.squared_error(x_pred, x_tgt)
        return loss.mean()
    
@dataclass(frozen=True)
class LagrangianModel:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(x, tau, tau + dt)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: float, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def weighted_pb_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(x_t, t, t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()
    
    def weighted_pd_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        def p_map(model: MeanFlowModel, r: jax.Array, t: jax.Array, x_r: jax.Array):
            if t.ndim == 1 and x_r.ndim == 2:
                return x_r + (t - r)[:, None] * model(x_r, r, t)
            else:
                return x_r + (t - r) * model(x_r, r, t)
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_r = jax.vmap(self.q_sample)(r, x_start, noise)

        def f(tt):
            return p_map(model, r, tt, x_r)
        _, x_pred = jax.jvp(f, (t,), (jnp.ones_like(t),))

        x_rt = p_map(model, r, t, x_r)
        x_tgt = model(jax.lax.stop_gradient(x_rt), t, t)
        loss = weights * optax.squared_error(x_pred, x_tgt)
        return loss.mean()
    

@dataclass(frozen=True)
class MeanFlowModel:
    num_timesteps: int

    # We only consider one-step sampling in the experiments
    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = (self.num_timesteps - t) * dt
            drift = model(x, tau - dt, tau)
            x_next = x - drift
            # x_next = x - drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: float, x_start: jax.Array, noise: jax.Array):
        return (1 - t) * x_start + t * noise
    
    def weighted_pb_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(x_t, t, t)
        loss = weights * optax.squared_error(v_pred, (noise - x_start))
        return loss.mean()

    def weighted_pd_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert r.ndim == 1 and t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v = noise - x_start
        zero_r = jnp.zeros_like(r, dtype=jnp.float32)
        one_t  = jnp.ones_like(t, dtype=jnp.float32)
        u_pred, dudt = jax.jvp(model, (x_t, r, t), (v, zero_r, one_t))
        u_tgt = jax.lax.stop_gradient(v - (t - r)[:, None] * dudt)
        loss = weights * optax.squared_error(u_pred, u_tgt)
        return loss.mean()