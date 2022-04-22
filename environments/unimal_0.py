import brax
import jax
import brax.jumpy as jp
import tensorflow_probability as tfp
import jax.numpy as jnp
from brax.envs import env

from environments.utils import quat2expmap

tfp = tfp.substrates.jax
tfd = tfp.distributions

class Unimal(env.Env):
  """Trains a unimal100 (floor-5506-10-6-01-15-48-35) to locomote to the ball."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    b_scale_max = jnp.ones(1) * 5.0
    b_scale_min = jnp.ones(1) * 2.0
    self.ball_r_fn = lambda: tfd.Uniform(
        low=b_scale_min, high=b_scale_max)
    self.ball_theta_fn = lambda: tfd.Uniform(
        low=jnp.zeros(1), high=jnp.ones(1))
    self.eps = 0.25
    self.agent_idx = 0
    self.object_idx = 9

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3, rng4 = jp.random_split(rng, 5)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

    # sample ball position
    init_ball_r = self.ball_r_fn().sample(seed=rng3)
    init_ball_theta = self.ball_theta_fn().sample(seed=rng4) * 2 * jnp.pi
    init_ball_x = init_ball_r * jnp.cos(init_ball_theta)
    init_ball_y = init_ball_r * jnp.sin(init_ball_theta)
    self.init_ball_xy = jnp.concatenate([init_ball_x, init_ball_y])
    pos = jax.ops.index_update(qp.pos, jnp.index_exp[self.object_idx, :2], self.init_ball_xy)
    qp = qp.replace(pos=pos)

    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {'agent_ball_dist': zero, 'distance': zero, 'z': zero}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # dense reward: ball_xy - agent_xy
    agent_ball_dist = jnp.linalg.norm(qp.pos[self.object_idx, :2] - qp.pos[self.agent_idx, :2], axis=-1)
    # reward = -agent_ball_dist
    x_before = state.qp.pos[self.agent_idx, 0]
    x_after = qp.pos[self.agent_idx, 0]
    z = jp.where(qp.pos[self.agent_idx, 2] >= 0.3, x=jp.float32(1)*0.001, y=jp.float32(0))
    reward = (x_after - x_before) / self.sys.config.dt + z

    #done = jp.where(
    #  agent_ball_dist <= self.eps, x=jp.float32(1), y=jp.float32(0))
    done = jp.float32(0)

    state.metrics.update(
      agent_ball_dist=agent_ball_dist, distance=agent_ball_dist, z=z)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    # TODO: limb_type_vec doesn't fit to unimal.
    #self.sys.body.index = {
    #  'torso_0': 0,
    #  'limb_0': 1,
    #  'limb_4': 2,
    #  'limb_6': 3,
    #  'limb_8': 4,
    #  'limb_1': 5,
    #  'limb_5': 6,
    #  'limb_7': 7,
    #  'limb_9': 8,
    #  'Ball': 9,
    #  'Ground': 10}

    torso_x_pos = qp.pos[0]
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    full_obs = []

    # torso_0
    limb_type_vec = jp.array((1, 0, 0, 0))
    xpos = qp.pos[0] - torso_x_pos
    expmap = quat2expmap(qp.rot[0])
    angle = jp.array([0.0])
    joint_range = jp.array([0.0, 0.0])
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[0], -10., 10.),
        qp.ang[0],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_0
    limb_type_vec = jp.array((0, 1, 0, 0))
    xpos = qp.pos[1] - torso_x_pos
    expmap = quat2expmap(qp.rot[1])
    angle = jnp.degrees(jp.array([joint_angle[0]]))
    joint_range = jp.array([-30.0, 30.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[1], -10., 10.),
        qp.ang[1],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_4
    limb_type_vec = jp.array((0, 0, 1, 0))
    xpos = qp.pos[2] - torso_x_pos
    expmap = quat2expmap(qp.rot[2])
    angle = jnp.degrees(jp.array([joint_angle[1]]))
    joint_range = jp.array([-30.0, 60.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[2], -10., 10.),
        qp.ang[2],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

     # limb_6
    limb_type_vec = jp.array((0, 0, 0, 1))
    xpos = qp.pos[3] - torso_x_pos
    expmap = quat2expmap(qp.rot[3])
    angle = jnp.degrees(jp.array([joint_angle[2]]))
    joint_range = jp.array([-60.0, 0.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[3], -10., 10.),
        qp.ang[3],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_8
    limb_type_vec = jp.array((0, 0, 0, 1))
    xpos = qp.pos[4] - torso_x_pos
    expmap = quat2expmap(qp.rot[4])
    angle = jnp.degrees(jp.array([joint_angle[3]]))
    joint_range = jp.array([0.0, 45.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[4], -10., 10.),
        qp.ang[4],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_1
    limb_type_vec = jp.array((0, 1, 0, 0))
    xpos = qp.pos[5] - torso_x_pos
    expmap = quat2expmap(qp.rot[5])
    angle = jnp.degrees(jp.array([joint_angle[4]]))
    joint_range = jp.array([-30.0, 30.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[5], -10., 10.),
        qp.ang[5],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_5
    limb_type_vec = jp.array((0, 0, 1, 0))
    xpos = qp.pos[6] - torso_x_pos
    expmap = quat2expmap(qp.rot[6])
    angle = jnp.degrees(jp.array([joint_angle[5]]))
    joint_range = jp.array([-30.0, 60.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[6], -10., 10.),
        qp.ang[6],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_7
    limb_type_vec = jp.array((0, 0, 1, 0))
    xpos = qp.pos[7] - torso_x_pos
    expmap = quat2expmap(qp.rot[7])
    angle = jnp.degrees(jp.array([joint_angle[6]]))
    joint_range = jp.array([-60.0, 0.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[7], -10., 10.),
        qp.ang[7],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_9
    limb_type_vec = jp.array((0, 0, 0, 1))
    xpos = qp.pos[8] - torso_x_pos
    expmap = quat2expmap(qp.rot[8])
    angle = jnp.degrees(jp.array([joint_angle[7]]))
    joint_range = jp.array([0.0, 45.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[8], -10., 10.),
        qp.ang[8],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # Ball
    limb_type_vec = jp.array((0, 0, 0, 0))
    xpos = qp.pos[self.object_idx] - torso_x_pos
    expmap = quat2expmap(qp.rot[self.object_idx])
    angle = jp.array([0.0])
    joint_range = jp.array([0.0, 0.0])
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[self.object_idx], -10., 10.),
        qp.ang[self.object_idx],
        expmap,
        limb_type_vec,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # 19 * (9 + 1) dim
    full_obs = jp.concatenate(full_obs)
    return full_obs.ravel()

_SYSTEM_CONFIG = """"""
