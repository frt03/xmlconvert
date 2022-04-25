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
  """Trains a unimal100 (floor-1409-1-6-01-07-43-16) to locomote to the ball."""

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
    self.object_idx = 8

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
    metrics = {'agent_ball_dist': zero, 'distance': zero, 'height': zero}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # dense reward: ball_xy - agent_xy
    agent_ball_dist = jnp.linalg.norm(qp.pos[self.object_idx, :2] - qp.pos[self.agent_idx, :2], axis=-1)
    reward = -agent_ball_dist

    done = jp.where(agent_ball_dist <= self.eps, x=jp.float32(1), y=jp.float32(0))
    # done = jp.float32(0)

    state.metrics.update(
      agent_ball_dist=agent_ball_dist, distance=agent_ball_dist, height=qp.pos[self.agent_idx, 2])

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:

    torso_x_pos = qp.pos[0]
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    full_obs = []

    # torso_0
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
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_0
    xpos = qp.pos[1] - torso_x_pos
    expmap = quat2expmap(qp.rot[1])
    angle = jnp.degrees(jp.array([joint_angle[0]]))
    joint_range = jp.array([-60.0, 60.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[1], -10., 10.),
        qp.ang[1],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_1
    xpos = qp.pos[2] - torso_x_pos
    expmap = quat2expmap(qp.rot[2])
    angle = jnp.degrees(jp.array([joint_angle[1]]))
    joint_range = jp.array([0.0, 90.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[2], -10., 10.),
        qp.ang[2],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_4
    xpos = qp.pos[3] - torso_x_pos
    expmap = quat2expmap(qp.rot[3])
    angle = jnp.degrees(jp.array([joint_angle[2]]))
    joint_range = jp.array([0.0, 45.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[3], -10., 10.),
        qp.ang[3],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_5
    xpos = qp.pos[4] - torso_x_pos
    expmap = quat2expmap(qp.rot[4])
    angle = jnp.degrees(jp.array([joint_angle[3]]))
    joint_range = jp.array([-30.0, 0.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[4], -10., 10.),
        qp.ang[4],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_7
    xpos = qp.pos[5] - torso_x_pos
    expmap = quat2expmap(qp.rot[5])
    angle = jnp.degrees(jp.array([joint_angle[4]]))
    joint_range = jp.array([0.0, 90.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[5], -10., 10.),
        qp.ang[5],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_6
    xpos = qp.pos[6] - torso_x_pos
    expmap = quat2expmap(qp.rot[6])
    angle = jnp.degrees(jp.array([joint_angle[5]]))
    joint_range = jp.array([-30.0, 0.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[6], -10., 10.),
        qp.ang[6],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # limb_8
    xpos = qp.pos[7] - torso_x_pos
    expmap = quat2expmap(qp.rot[7])
    angle = jnp.degrees(jp.array([joint_angle[6]]))
    joint_range = jp.array([0.0, 90.0])
    angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
    joint_range = (180.0 + joint_range) / 360.0
    obs = jp.concatenate(
      [
        xpos,
        jp.clip(qp.vel[7], -10., 10.),
        qp.ang[7],
        expmap,
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # Ball
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
        angle,
        joint_range
      ])
    full_obs.append(obs)

    # 15 * (8 + 1) dim
    full_obs = jp.concatenate(full_obs)
    return full_obs.ravel()

_SYSTEM_CONFIG = """
bodies {
  name: "torso_0"
  colliders {
    position {
    }
    sphere {
      radius: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.1887903
}
bodies {
  name: "limb_0"
  colliders {
    position {
      z: -0.175
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.05
      length: 0.45
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.2907445
}
bodies {
  name: "limb_1"
  colliders {
    position {
      x: -0.175
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05
      length: 0.45
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.2907445
}
bodies {
  name: "limb_4"
  colliders {
    position {
      x: -0.225
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05
      length: 0.55
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.8405232
}
bodies {
  name: "limb_5"
  colliders {
    position {
      y: 0.225
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.05
      length: 0.55
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.8405232
}
bodies {
  name: "limb_7"
  colliders {
    position {
      x: -0.125
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05
      length: 0.35
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.740966
}
bodies {
  name: "limb_6"
  colliders {
    position {
      y: -0.225
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.05
      length: 0.55
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.8405232
}
bodies {
  name: "limb_8"
  colliders {
    position {
      x: -0.125
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05
      length: 0.35
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.740966
}
bodies {
  name: "Ball"
  colliders {
    sphere { radius: 0.25 }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1000.0
}
bodies {
  name: "Ground"
  colliders {
    plane {}
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
  frozen { all: true }
}
joints {
  name: "limbx_0"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_0"
  parent_offset {
    z: -0.05
  }
  child_offset {
    z: 0.05
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: -60.0
    max: 60.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_0"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_0"
  parent_offset {
    z: -0.05
  }
  child_offset {
    z: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -45.0
    max: 45.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_1"
  stiffness: 5000.0
  parent: "limb_0"
  child: "limb_1"
  parent_offset {
    z: -0.2
  }
  child_offset {
    x: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    max: 90.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_4"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_4"
  parent_offset {
    x: -0.05
  }
  child_offset {
    x: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    max: 45.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_5"
  stiffness: 5000.0
  parent: "limb_4"
  child: "limb_5"
  parent_offset {
    x: -0.45
  }
  child_offset {
    y: -0.05
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: -30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_7"
  stiffness: 5000.0
  parent: "limb_5"
  child: "limb_7"
  parent_offset {
    y: 0.45
  }
  child_offset {
    x: 0.05
  }
  rotation {
    x: -0.0
    y: 90.0
  }
  angle_limit {
    max: 90.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_7"
  stiffness: 5000.0
  parent: "limb_5"
  child: "limb_7"
  parent_offset {
    y: 0.45
  }
  child_offset {
    x: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -30.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_6"
  stiffness: 5000.0
  parent: "limb_4"
  child: "limb_6"
  parent_offset {
    x: -0.45
  }
  child_offset {
    y: 0.05
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: -30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_8"
  stiffness: 5000.0
  parent: "limb_6"
  child: "limb_8"
  parent_offset {
    y: -0.45
  }
  child_offset {
    x: 0.05
  }
  rotation {
    x: -0.0
    y: 90.0
  }
  angle_limit {
    max: 90.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_8"
  stiffness: 5000.0
  parent: "limb_6"
  child: "limb_8"
  parent_offset {
    y: -0.45
  }
  child_offset {
    x: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -30.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
actuators {
  name: "limbx_0"
  joint: "limbx_0"
  strength: 200.0
  torque {
  }
}
actuators {
  name: "limby_0"
  joint: "limby_0"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limby_1"
  joint: "limby_1"
  strength: 200.0
  torque {
  }
}
actuators {
  name: "limby_4"
  joint: "limby_4"
  strength: 200.0
  torque {
  }
}
actuators {
  name: "limbx_5"
  joint: "limbx_5"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_7"
  joint: "limbx_7"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "limby_7"
  joint: "limby_7"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_6"
  joint: "limbx_6"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_8"
  joint: "limbx_8"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "limby_8"
  joint: "limby_8"
  strength: 250.0
  torque {
  }
}
friction: 0.6
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "torso_0"
  second: "limb_1"
}
collide_include {
  first: "torso_0"
  second: "limb_5"
}
collide_include {
  first: "torso_0"
  second: "limb_7"
}
collide_include {
  first: "torso_0"
  second: "limb_6"
}
collide_include {
  first: "torso_0"
  second: "limb_8"
}
collide_include {
  first: "limb_0"
  second: "limb_5"
}
collide_include {
  first: "limb_0"
  second: "limb_7"
}
collide_include {
  first: "limb_0"
  second: "limb_6"
}
collide_include {
  first: "limb_0"
  second: "limb_8"
}
collide_include {
  first: "limb_1"
  second: "limb_4"
}
collide_include {
  first: "limb_1"
  second: "limb_5"
}
collide_include {
  first: "limb_1"
  second: "limb_7"
}
collide_include {
  first: "limb_1"
  second: "limb_6"
}
collide_include {
  first: "limb_1"
  second: "limb_8"
}
collide_include {
  first: "limb_4"
  second: "limb_7"
}
collide_include {
  first: "limb_4"
  second: "limb_8"
}
collide_include {
  first: "limb_5"
  second: "limb_8"
}
collide_include {
  first: "limb_7"
  second: "limb_6"
}
collide_include {
  first: "limb_7"
  second: "limb_8"
}
collide_include {
  first: "Ground"
  second: "torso_0"
}
collide_include {
  first: "Ground"
  second: "limb_0"
}
collide_include {
  first: "Ground"
  second: "limb_1"
}
collide_include {
  first: "Ground"
  second: "limb_4"
}
collide_include {
  first: "Ground"
  second: "limb_5"
}
collide_include {
  first: "Ground"
  second: "limb_6"
}
collide_include {
  first: "Ground"
  second: "limb_7"
}
collide_include {
  first: "Ground"
  second: "limb_8"
}
collide_include {
  first: "Ground"
  second: "Ball"
}
collide_include {
  first: "Ball"
  second: "torso_0"
}
collide_include {
  first: "Ball"
  second: "limb_0"
}
collide_include {
  first: "Ball"
  second: "limb_1"
}
collide_include {
  first: "Ball"
  second: "limb_4"
}
collide_include {
  first: "Ball"
  second: "limb_5"
}
collide_include {
  first: "Ball"
  second: "limb_6"
}
collide_include {
  first: "Ball"
  second: "limb_7"
}
collide_include {
  first: "Ball"
  second: "limb_8"
}
dt: 0.02
substeps: 4
"""
