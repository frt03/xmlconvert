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
  """Trains a unimal100 (floor-1409-9-9-01-14-34-07) to run."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.agent_idx = 0

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3, rng4 = jp.random_split(rng, 5)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {'height': zero}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[self.agent_idx, 0]
    x_after = qp.pos[self.agent_idx, 0]
    height_bonus = jp.where(qp.pos[self.agent_idx, 2] >= 0.4, x=jp.float32(1)*0.001, y=jp.float32(0))
    reward = (x_after - x_before) / self.sys.config.dt + height_bonus

    done = jp.where(qp.pos[self.agent_idx, 2] < 0.3, x=jp.float32(1), y=jp.float32(0))
    # done = jp.float32(0)

    state.metrics.update(height=qp.pos[self.agent_idx, 2])

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
    joint_range = jp.array([-30.0, 60.0])
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
    joint_range = jp.array([-30.0, 30.0])
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

    # 15 * 8 dim
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
  mass: 3.3510323
}
bodies {
  name: "limb_0"
  colliders {
    position {
      z: -0.125
    }
    rotation {
      y: -0.0
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
  mass: 2.4870942
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
  mass: 3.2724924
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
  mass: 4.0578904
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
  mass: 4.0578904
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
  mass: 2.4870942
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
  mass: 4.0578904
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
  mass: 2.4870942
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
    z: -0.15
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
  name: "limbx_4"
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
    x: -0.0
    y: 90.0
  }
  angle_limit {
    min: -30.0
    max: 60.0
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
  strength: 150.0
  angle {
  }
}
actuators {
  name: "limby_0"
  joint: "limby_0"
  strength: 300.0
  angle {
  }
}
actuators {
  name: "limby_1"
  joint: "limby_1"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "limbx_4"
  joint: "limbx_4"
  strength: 250.0
  angle {
  }
}
actuators {
  name: "limbx_5"
  joint: "limbx_5"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "limbx_7"
  joint: "limbx_7"
  strength: 250.0
  angle {
  }
}
actuators {
  name: "limby_7"
  joint: "limby_7"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "limbx_6"
  joint: "limbx_6"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "limbx_8"
  joint: "limbx_8"
  strength: 250.0
  angle {
  }
}
actuators {
  name: "limby_8"
  joint: "limby_8"
  strength: 150.0
  angle {
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
