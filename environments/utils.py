import flax
import jax

from brax.envs import env as brax_env
from brax import jumpy as jp
from jax import numpy as jnp

from typing import Dict


def quat2expmap(quat: jp.ndarray) -> jp.ndarray:
  """Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args:
    quat: 4-dim quaternion
  Returns:
    r: 3-dim exponential map
  Raises:
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  # assert jnp.abs(jnp.linalg.norm(quat) - 1) <= 1e-3, 'quat2expmap: input quaternion is not norm 1'

  sinhalftheta = jnp.linalg.norm(quat[1:])
  coshalftheta = quat[0]
  r0 = jnp.divide(quat[1:], (jnp.linalg.norm(quat[1:]) + jnp.finfo(jnp.float32).eps))
  theta = 2 * jnp.arctan2(sinhalftheta, coshalftheta)
  theta = jnp.mod(theta + 2 * jp.pi, 2 * jp.pi)
  r = jax.lax.cond(
    theta > jp.pi,
    lambda x: -r0 * (2 * jp.pi - x),
    lambda x: r0 * x,
    theta)
  return r


@flax.struct.dataclass
class GoalEvalMetrics:
  current_episode_metrics: Dict[str, jp.ndarray]
  completed_episodes_metrics: Dict[str, jp.ndarray]
  completed_episodes: jp.ndarray
  completed_episodes_steps: jp.ndarray
  success_episodes: jp.ndarray  # added
  final_distance: jp.ndarray  # added


class GoalEvalWrapper(brax_env.Wrapper):
  """Brax env with goal-based eval metrics."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = GoalEvalMetrics(
        current_episode_metrics=jax.tree_map(jp.zeros_like,
                                             reset_state.metrics),
        completed_episodes_metrics=jax.tree_map(
            lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics),
        completed_episodes=jp.zeros(()),
        completed_episodes_steps=jp.zeros(()),
        success_episodes=jp.zeros(()),
        final_distance=jp.zeros(()))
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, GoalEvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}')
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['reward'] = nstate.reward
    # steps stores the highest step reached when done = True, and then
    # the next steps becomes action_repeat
    completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
        nstate.info['steps'] * nstate.done)
    current_episode_metrics = jax.tree_multimap(
        lambda a, b: a + b, state_metrics.current_episode_metrics,
        nstate.metrics)
    completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
    # additional metrics
    success_episodes = state_metrics.success_episodes + jp.sum(nstate.done * (1.0 - nstate.info['truncation']))
    final_distance = state_metrics.final_distance + jp.sum(nstate.done * nstate.metrics['distance'])

    completed_episodes_metrics = jax.tree_multimap(
        lambda a, b: a + jp.sum(b * nstate.done),
        state_metrics.completed_episodes_metrics, current_episode_metrics)
    current_episode_metrics = jax.tree_multimap(
        lambda a, b: a * (1 - nstate.done) + b * nstate.done,
        current_episode_metrics, nstate.metrics)

    eval_metrics = GoalEvalMetrics(
        current_episode_metrics=current_episode_metrics,
        completed_episodes_metrics=completed_episodes_metrics,
        completed_episodes=completed_episodes,
        completed_episodes_steps=completed_episodes_steps,
        success_episodes=success_episodes,
        final_distance=final_distance,)
    nstate.info['eval_metrics'] = eval_metrics
    return nstate
