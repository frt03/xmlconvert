# from https://github.com/google/brax/
import functools
from typing import Callable, Optional, Union, overload

from brax.envs import wrappers
from brax.envs.env import Env
import gym

import environments.unimal_0
import environments.unimal_1
import environments.unimal_2
import environments.unimal_0_run
import environments.unimal_1_run
import environments.unimal_2_run

from environments.utils import GoalEvalWrapper


_envs = {
    'unimal_0': unimal_0.Unimal,
    'unimal_1': unimal_1.Unimal,
    'unimal_2': unimal_2.Unimal,
    'unimal_0_run': unimal_0_run.Unimal,
    'unimal_1_run': unimal_1_run.Unimal,
    'unimal_2_run': unimal_2_run.Unimal,
}


goal_envs_list = ('unimal_0', 'unimal_1', 'unimal_2')


def create(env_name: str,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           eval_metrics: bool = False,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = _envs[env_name](**kwargs)
  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)
  if eval_metrics and (env_name in goal_envs_list):
    env = GoalEvalWrapper(env)
  elif eval_metrics:
    env = wrappers.EvalWrapper(env)

  return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, **kwargs)


@overload
def create_gym_env(env_name: str,
                   batch_size: None = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.Env:
  ...


@overload
def create_gym_env(env_name: str,
                   batch_size: int,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.vector.VectorEnv:
  ...


def create_gym_env(env_name: str,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
  """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
  environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
  if batch_size is None:
    return wrappers.GymWrapper(environment, seed=seed, backend=backend)
  if batch_size <= 0:
    raise ValueError(
        '`batch_size` should either be None or a positive integer.')
  return wrappers.VectorGymWrapper(environment, seed=seed, backend=backend)
