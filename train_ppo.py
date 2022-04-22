import copy
import functools
import os

from absl import app
from brax import envs
from absl import flags
from brax.io import html
from brax.io import model
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import logger_utils
from datetime import datetime
import matplotlib.pyplot as plt
import jax

import ppo
import environments


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'unimal_0', 'Name of environment to train.')
flags.DEFINE_integer('total_env_steps', 30000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_frequency', 20, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 2048, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('reward_scaling', 10.0, 'Reward scale.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('entropy_cost', 1e-2, 'Entropy cost.')
flags.DEFINE_integer('unroll_length', 5, 'Unroll length.')
flags.DEFINE_float('discounting', 0.97, 'Discounting.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('num_minibatches', 32, 'Number')
flags.DEFINE_integer('num_update_epochs', 4,
                     'Number of times to reuse each transition for gradient '
                     'computation.')
flags.DEFINE_string('logdir', './results/', 'Logdir.')
flags.DEFINE_bool('normalize_observations', True,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_integer('num_save_html', 3, 'Number of Videos.')


def main(unused_argv):
  # save dir
  output_dir = os.path.join(
    FLAGS.logdir,
    f'ppo_{FLAGS.env}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  print(f'Saving outputs to {output_dir}')
  os.makedirs(output_dir, exist_ok=True)

  goal_env = (FLAGS.env in environments.goal_envs_list)

  train_job_params = {
    'action_repeat': FLAGS.action_repeat,
    'batch_size': FLAGS.batch_size,
    'discounting': FLAGS.discounting,
    'entropy_cost': FLAGS.entropy_cost,
    'episode_length': FLAGS.episode_length,
    'learning_rate': FLAGS.learning_rate,
    'log_frequency': FLAGS.eval_frequency,
    'normalize_observations': FLAGS.normalize_observations,
    'num_envs': FLAGS.num_envs,
    'num_minibatches': FLAGS.num_minibatches,
    'num_timesteps': FLAGS.total_env_steps,
    'num_update_epochs': FLAGS.num_update_epochs,
    'max_devices_per_host': FLAGS.max_devices_per_host,
    'reward_scaling': FLAGS.reward_scaling,
    'seed': FLAGS.seed,
    'unroll_length': FLAGS.unroll_length,
    'goal_env': goal_env}

  config = copy.deepcopy(train_job_params)
  config['env'] = FLAGS.env
  print(config)

  # logging
  logger_utils.save_config(
      f'{output_dir}/config.txt', config, verbose=True)
  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)

  times = [datetime.now()]
  plotpatterns = []

  progress, _, _, _ = experiments.get_progress_fn(
      plotpatterns,
      times,
      tab=tab,
      max_ncols=5,
      xlim=[0, train_job_params['num_timesteps']],
      post_plot_fn=functools.partial(plt.savefig, f'{output_dir}/progress.png'))

  env_fn = environments.create_fn(FLAGS.env)

  inference_fn, params, _ = ppo.train(
      environment_fn=env_fn,
      progress_fn=None,
      **train_job_params)

  # output an episode trajectory
  env = env_fn()
  rng = jax.random.PRNGKey(FLAGS.seed)
  for i, rng_i in enumerate(jax.random.split(rng, num=FLAGS.num_save_html)):
    qps = []
    rs = []
    jit_inference_fn = jax.jit(inference_fn)
    jit_step_fn = jax.jit(env.step)
    tmp_key, rng_i = jax.random.split(rng_i)
    state = env.reset(rng_i)
    while not state.done:
      qps.append(state.qp)
      act = jit_inference_fn(params, state.obs, tmp_key)
      state = jit_step_fn(state, act)
      rs.append(state.reward)
    print(len(rs))
    print(jax.numpy.sum(jax.numpy.array(rs)))

    html_path = os.path.join(output_dir, f'trajectory_{i}.html')
    html.save_html(html_path, env.sys, qps)


if __name__ == '__main__':
  app.run(main)
