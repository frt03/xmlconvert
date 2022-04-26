import brax
import jax
import jax.numpy as jnp
import brax.jumpy as jp
import tensorflow_probability as tfp
import jax.numpy as jnp
from brax.envs import env
from google.protobuf import text_format

from environments.utils import quat2expmap


class Unimal(env.Env):
  """Trains a unimal100 (floor-5506-10-1-01-12-44-00) to run."""

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
    metrics = {}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[self.agent_idx, 0]
    x_after = qp.pos[self.agent_idx, 0]
    height_bonus = jp.where(qp.pos[self.agent_idx, 2] >= 0.2, x=jp.float32(1), y=jp.float32(0))
    reward = (x_after - x_before) / self.sys.config.dt + height_bonus

    done = jp.where(qp.pos[self.agent_idx, 2] < 0.15, x=jp.float32(1), y=jp.float32(0))

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:

    torso_x_pos = qp.pos[0]
    # (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    # NOTE: brax does not like having more than one joint defined per pair of parents and children
    # your config had a couple of duplicate joints where the joints had the same parent and child.
    # the brax-y way of doing this is defining a single joint with multiple angle limits.  canonically,
    # brax uses the first angle limit for the local joint x-axis' limits, second angle limit for y, third for z.
    # doing this completely borked the observations in your get_obs function, so I regenerated them with a script a couple cells below
    # note that this includes some dummy obses now because every joint reports both its x-angle and y- angle, and some joints have a "frozen" y-angle
    # because they originally were only 1-dof joints.
  
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    joint_angle2, joint_vel2 = self.sys.joints[1].angle_vel(qp)

    full_obs = [joint_angle, joint_angle2[0], joint_angle2[1], joint_vel, joint_vel2[0], joint_vel2[1]]
    num_limb = len(self.sys.body.index) - 1
    for i in range(num_limb):
      full_obs.append(qp.pos[i] - torso_x_pos)
      full_obs.append(quat2expmap(qp.rot[i]))
      full_obs.append(jp.clip(qp.vel[i], -10., 10.))
      full_obs.append(qp.ang[i])
    # full_obs = [joint_angle, joint_vel]
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
      y: 0.175
    }
    rotation {
      x: -90.0
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
      x: 0.175
    }
    rotation {
      x: -0.0
      y: 90.0
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
  name: "limb_9"
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
  mass: 3.2724924
}
bodies {
  name: "limb_6"
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
  name: "limb_1"
  colliders {
    position {
      y: -0.175
    }
    rotation {
      x: 90.0
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
  mass: 3.2724924
}
bodies {
  name: "limb_5"
  colliders {
    position {
      x: 0.175
    }
    rotation {
      x: -0.0
      y: 90.0
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
  name: "limb_10"
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
  mass: 3.2724924
}
bodies {
  name: "limb_7"
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
  name: "limb_8"
  colliders {
    position {
      x: 0.175
    }
    rotation {
      x: -0.0
      y: 90.0
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
  name: "limb_11"
  colliders {
    position {
      x: -0.09
      z: -0.09
    }
    rotation {
      x: 180.0
      y: -45.0
      z: 180.0
    }
    capsule {
      radius: 0.05
      length: 0.35455844
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.522896
}
bodies {
  name: "Ground"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "limby_0"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_0"
  parent_offset {
    y: 0.05
  }
  child_offset {
    y: -0.05
  }
  rotation {
    y: -90.0
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
  name: "limbx_4"
  stiffness: 5000.0
  parent: "limb_0"
  child: "limb_4"
  parent_offset {
    y: 0.35
  }
  child_offset {
    x: -0.05
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -60.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_9"
  stiffness: 5000.0
  parent: "limb_4"
  child: "limb_9"
  parent_offset {
    x: 0.2
  }
  child_offset {
    z: 0.05
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
  name: "limby_9"
  stiffness: 5000.0
  parent: "limb_4"
  child: "limb_9"
  parent_offset {
    x: 0.2
  }
  child_offset {
    z: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -60.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_6"
  stiffness: 5000.0
  parent: "limb_0"
  child: "limb_6"
  parent_offset {
    y: 0.35
  }
  child_offset {
    x: 0.05
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
  parent: "torso_0"
  child: "limb_1"
  parent_offset {
    y: -0.05
  }
  child_offset {
    y: 0.05
  }
  rotation {
    x: -0.0
    y: 90.0
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
  name: "limbx_5"
  stiffness: 5000.0
  parent: "limb_1"
  child: "limb_5"
  parent_offset {
    y: -0.35
  }
  child_offset {
    x: -0.05
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -60.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_10"
  stiffness: 5000.0
  parent: "limb_5"
  child: "limb_10"
  parent_offset {
    x: 0.2
  }
  child_offset {
    z: 0.05
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
  name: "limby_10"
  stiffness: 5000.0
  parent: "limb_5"
  child: "limb_10"
  parent_offset {
    x: 0.2
  }
  child_offset {
    z: 0.05
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -60.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_7"
  stiffness: 5000.0
  parent: "limb_1"
  child: "limb_7"
  parent_offset {
    y: -0.35
  }
  child_offset {
    x: 0.05
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
  name: "limbx_8"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_8"
  parent_offset {
    x: 0.05
  }
  child_offset {
    x: -0.05
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -90.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_11"
  stiffness: 5000.0
  parent: "torso_0"
  child: "limb_11"
  parent_offset {
    x: -0.03
    z: -0.03
  }
  child_offset {
    x: 0.04
    z: 0.04
  }
  rotation {
    y: 45.0
  }
  angle_limit {
    max: 90.0
  }
  reference_rotation {
    y: -0.0
  }
}
actuators {
  name: "limby_0"
  joint: "limby_0"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_4"
  joint: "limbx_4"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_9"
  joint: "limbx_9"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_9"
  joint: "limby_9"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limby_6"
  joint: "limby_6"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_1"
  joint: "limby_1"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_5"
  joint: "limbx_5"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_10"
  joint: "limbx_10"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_10"
  joint: "limby_10"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limby_7"
  joint: "limby_7"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_8"
  joint: "limbx_8"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_11"
  joint: "limbx_11"
  strength: 150.0
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
  second: "limb_4"
}
collide_include {
  first: "torso_0"
  second: "limb_9"
}
collide_include {
  first: "torso_0"
  second: "limb_6"
}
collide_include {
  first: "torso_0"
  second: "limb_5"
}
collide_include {
  first: "torso_0"
  second: "limb_10"
}
collide_include {
  first: "torso_0"
  second: "limb_7"
}
collide_include {
  first: "limb_0"
  second: "limb_9"
}
collide_include {
  first: "limb_0"
  second: "limb_5"
}
collide_include {
  first: "limb_0"
  second: "limb_10"
}
collide_include {
  first: "limb_0"
  second: "limb_7"
}
collide_include {
  first: "limb_4"
  second: "limb_1"
}
collide_include {
  first: "limb_4"
  second: "limb_5"
}
collide_include {
  first: "limb_4"
  second: "limb_10"
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
  first: "limb_4"
  second: "limb_11"
}
collide_include {
  first: "limb_9"
  second: "limb_6"
}
collide_include {
  first: "limb_9"
  second: "limb_1"
}
collide_include {
  first: "limb_9"
  second: "limb_5"
}
collide_include {
  first: "limb_9"
  second: "limb_10"
}
collide_include {
  first: "limb_9"
  second: "limb_7"
}
collide_include {
  first: "limb_9"
  second: "limb_8"
}
collide_include {
  first: "limb_9"
  second: "limb_11"
}
collide_include {
  first: "limb_6"
  second: "limb_1"
}
collide_include {
  first: "limb_6"
  second: "limb_5"
}
collide_include {
  first: "limb_6"
  second: "limb_10"
}
collide_include {
  first: "limb_6"
  second: "limb_7"
}
collide_include {
  first: "limb_6"
  second: "limb_8"
}
collide_include {
  first: "limb_6"
  second: "limb_11"
}
collide_include {
  first: "limb_1"
  second: "limb_10"
}
collide_include {
  first: "limb_5"
  second: "limb_8"
}
collide_include {
  first: "limb_5"
  second: "limb_11"
}
collide_include {
  first: "limb_10"
  second: "limb_7"
}
collide_include {
  first: "limb_10"
  second: "limb_8"
}
collide_include {
  first: "limb_10"
  second: "limb_11"
}
collide_include {
  first: "limb_7"
  second: "limb_8"
}
collide_include {
  first: "limb_7"
  second: "limb_11"
}
"""

config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())

config.substeps = 10
config.dt = 0.05

paired = {}

# fix colliders
for c in config.collide_include:
  c.Clear()

parent_childs = {}
for j in config.joints:
  if j.parent not in parent_childs:
    parent_childs[j.parent] = []
  if j.child not in parent_childs:
    parent_childs[j.child] = []

  parent_childs[j.parent].append(j.child)
  parent_childs[j.child].append(j.parent)

parent_childs['Ground'] = []


paired = {}

for b1 in config.bodies:
  for b2 in config.bodies:
    if b2.name not in parent_childs[b1.name] and b1.name not in parent_childs[b2.name] and (b1.name, b2.name) not in paired:
      if b1.name != b2.name:
        c = config.collide_include.add()
        c.first = b1.name
        c.second = b2.name
        paired[(b1.name, b2.name)] = True

# deduplicate joints
joints_dict = {}
del_names = []
for i,j in enumerate(config.joints):

  if (j.parent, j.child) in joints_dict:
    thismin = j.angle_limit[0].min
    thismax = j.angle_limit[0].max
    a_l = joints_dict[(j.parent, j.child)].angle_limit.add()
    a_l.min=thismin
    a_l.max=thismax
    
    del_names.append(j.name)

    del config.joints[i]
    
  else:
    joints_dict[(j.parent, j.child)] = j

# remove actuators for duplicated joints
for i,a in enumerate(config.actuators):
  if a.name in del_names:
    del config.actuators[i]

# add a stiff joint so that everything can be sphericalized
for j in config.joints:
  if len(j.angle_limit)==1:
    j.angle_limit.add()

_SYSTEM_CONFIG = str(config)

# print(_SYSTEM_CONFIG)

# TODO:
#  - rename limb//0 --> limb_0
#  - remove "worldbody"
#  - add joints.angular_damping = actuators.strength / 10
#  - if the joint is y-axis one, fix the order of stiff joint
#  - find the "end-effector"
#  - "5506" series might be more stable than "1409"
CONFIGS = """
bodies {
  name: "torso_0"
  colliders {
    position {
    }
    sphere {
      radius: 0.10000000149011612
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.351032257080078
}
bodies {
  name: "limb_0"
  colliders {
    position {
      y: 0.17499999701976776
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_4"
  colliders {
    position {
      x: 0.17499999701976776
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_9"
  colliders {
    position {
      z: -0.17499999701976776
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_6"
  colliders {
    position {
      x: -0.22499999403953552
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.550000011920929
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.05789041519165
}
bodies {
  name: "limb_1"
  colliders {
    position {
      y: -0.17499999701976776
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_5"
  colliders {
    position {
      x: 0.17499999701976776
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_10"
  colliders {
    position {
      z: -0.17499999701976776
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_7"
  colliders {
    position {
      x: -0.22499999403953552
    }
    rotation {
      y: -90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.550000011920929
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.05789041519165
}
bodies {
  name: "limb_8"
  colliders {
    position {
      x: 0.17499999701976776
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.44999998807907104
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.2724924087524414
}
bodies {
  name: "limb_11"
  colliders {
    position {
      x: -0.09000000357627869
      z: -0.09000000357627869
    }
    rotation {
      x: 180.0
      y: -45.0
      z: 180.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.35455843806266785
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.5228960514068604
}
bodies {
  name: "Ground"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "limby_0"
  stiffness: 5000.0
  angular_damping: 15
  parent: "torso_0"
  child: "limb_0"
  parent_offset {
    y: 0.05000000074505806
  }
  child_offset {
    y: -0.05000000074505806
  }
  rotation {
    y: -90.0
  }
  angle_limit {
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
  name: "limbx_4"
  stiffness: 5000.0
  angular_damping: 25
  parent: "limb_0"
  child: "limb_4"
  parent_offset {
    y: 0.3499999940395355
  }
  child_offset {
    x: -0.05000000074505806
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -60.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_9"
  stiffness: 5000.0
  angular_damping: 15
  parent: "limb_4"
  child: "limb_9"
  parent_offset {
    x: 0.20000000298023224
  }
  child_offset {
    z: 0.05000000074505806
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: -30.0
  }
  angle_limit {
    min: -60.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_6"
  stiffness: 5000.0
  angular_damping: 15
  parent: "limb_0"
  child: "limb_6"
  parent_offset {
    y: 0.3499999940395355
  }
  child_offset {
    x: 0.05000000074505806
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
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
  angular_damping: 15
  parent: "torso_0"
  child: "limb_1"
  parent_offset {
    y: -0.05000000074505806
  }
  child_offset {
    y: 0.05000000074505806
  }
  rotation {
    x: -0.0
    y: 90.0
  }
  angle_limit {
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
  name: "limbx_5"
  stiffness: 5000.0
  angular_damping: 25
  parent: "limb_1"
  child: "limb_5"
  parent_offset {
    y: -0.3499999940395355
  }
  child_offset {
    x: -0.05000000074505806
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -60.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_10"
  stiffness: 5000.0
  angular_damping: 15
  parent: "limb_5"
  child: "limb_10"
  parent_offset {
    x: 0.20000000298023224
  }
  child_offset {
    z: 0.05000000074505806
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: -30.0
  }
  angle_limit {
    min: -60.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limby_7"
  stiffness: 5000.0
  angular_damping: 15
  parent: "limb_1"
  child: "limb_7"
  parent_offset {
    y: -0.3499999940395355
  }
  child_offset {
    x: 0.05000000074505806
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
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
  name: "limbx_8"
  stiffness: 5000.0
  angular_damping: 25
  parent: "torso_0"
  child: "limb_8"
  parent_offset {
    x: 0.05000000074505806
  }
  child_offset {
    x: -0.05000000074505806
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -90.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "limbx_11"
  stiffness: 5000.0
  angular_damping: 15
  parent: "torso_0"
  child: "limb_11"
  parent_offset {
    x: -0.029999999329447746
    z: -0.029999999329447746
  }
  child_offset {
    x: 0.03999999910593033
    z: 0.03999999910593033
  }
  rotation {
    y: 45.0
  }
  angle_limit {
    max: 90.0
  }
  reference_rotation {
    y: -0.0
  }
}
actuators {
  name: "limby_0"
  joint: "limby_0"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_4"
  joint: "limbx_4"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_9"
  joint: "limbx_9"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_6"
  joint: "limby_6"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_1"
  joint: "limby_1"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_5"
  joint: "limbx_5"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_10"
  joint: "limbx_10"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limby_7"
  joint: "limby_7"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "limbx_8"
  joint: "limbx_8"
  strength: 250.0
  torque {
  }
}
actuators {
  name: "limbx_11"
  joint: "limbx_11"
  strength: 150.0
  torque {
  }
}
friction: 0.6000000238418579
gravity {
  z: -9.8100004196167
}
angular_damping: -0.05000000074505806
baumgarte_erp: 0.10000000149011612
collide_include {
  first: "torso_0"
  second: "limb_4"
}
collide_include {
  first: "torso_0"
  second: "limb_9"
}
collide_include {
  first: "torso_0"
  second: "limb_6"
}
collide_include {
  first: "torso_0"
  second: "limb_5"
}
collide_include {
  first: "torso_0"
  second: "limb_10"
}
collide_include {
  first: "torso_0"
  second: "limb_7"
}
collide_include {
  first: "torso_0"
  second: "Ground"
}
collide_include {
  first: "limb_0"
  second: "limb_9"
}
collide_include {
  first: "limb_0"
  second: "limb_1"
}
collide_include {
  first: "limb_0"
  second: "limb_5"
}
collide_include {
  first: "limb_0"
  second: "limb_10"
}
collide_include {
  first: "limb_0"
  second: "limb_7"
}
collide_include {
  first: "limb_0"
  second: "limb_8"
}
collide_include {
  first: "limb_0"
  second: "limb_11"
}
collide_include {
  first: "limb_0"
  second: "Ground"
}
collide_include {
  first: "limb_4"
  second: "torso_0"
}
collide_include {
  first: "limb_4"
  second: "limb_6"
}
collide_include {
  first: "limb_4"
  second: "limb_1"
}
collide_include {
  first: "limb_4"
  second: "limb_5"
}
collide_include {
  first: "limb_4"
  second: "limb_10"
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
  first: "limb_4"
  second: "limb_11"
}
collide_include {
  first: "limb_4"
  second: "Ground"
}
collide_include {
  first: "limb_9"
  second: "torso_0"
}
collide_include {
  first: "limb_9"
  second: "limb_0"
}
collide_include {
  first: "limb_9"
  second: "limb_6"
}
collide_include {
  first: "limb_9"
  second: "limb_1"
}
collide_include {
  first: "limb_9"
  second: "limb_5"
}
collide_include {
  first: "limb_9"
  second: "limb_10"
}
collide_include {
  first: "limb_9"
  second: "limb_7"
}
collide_include {
  first: "limb_9"
  second: "limb_8"
}
collide_include {
  first: "limb_9"
  second: "limb_11"
}
collide_include {
  first: "limb_9"
  second: "Ground"
}
collide_include {
  first: "limb_6"
  second: "torso_0"
}
collide_include {
  first: "limb_6"
  second: "limb_4"
}
collide_include {
  first: "limb_6"
  second: "limb_9"
}
collide_include {
  first: "limb_6"
  second: "limb_1"
}
collide_include {
  first: "limb_6"
  second: "limb_5"
}
collide_include {
  first: "limb_6"
  second: "limb_10"
}
collide_include {
  first: "limb_6"
  second: "limb_7"
}
collide_include {
  first: "limb_6"
  second: "limb_8"
}
collide_include {
  first: "limb_6"
  second: "limb_11"
}
collide_include {
  first: "limb_6"
  second: "Ground"
}
collide_include {
  first: "limb_1"
  second: "limb_0"
}
collide_include {
  first: "limb_1"
  second: "limb_4"
}
collide_include {
  first: "limb_1"
  second: "limb_9"
}
collide_include {
  first: "limb_1"
  second: "limb_6"
}
collide_include {
  first: "limb_1"
  second: "limb_10"
}
collide_include {
  first: "limb_1"
  second: "limb_8"
}
collide_include {
  first: "limb_1"
  second: "limb_11"
}
collide_include {
  first: "limb_1"
  second: "Ground"
}
collide_include {
  first: "limb_5"
  second: "torso_0"
}
collide_include {
  first: "limb_5"
  second: "limb_0"
}
collide_include {
  first: "limb_5"
  second: "limb_4"
}
collide_include {
  first: "limb_5"
  second: "limb_9"
}
collide_include {
  first: "limb_5"
  second: "limb_6"
}
collide_include {
  first: "limb_5"
  second: "limb_7"
}
collide_include {
  first: "limb_5"
  second: "limb_8"
}
collide_include {
  first: "limb_5"
  second: "limb_11"
}
collide_include {
  first: "limb_5"
  second: "Ground"
}
collide_include {
  first: "limb_10"
  second: "torso_0"
}
collide_include {
  first: "limb_10"
  second: "limb_0"
}
collide_include {
  first: "limb_10"
  second: "limb_4"
}
collide_include {
  first: "limb_10"
  second: "limb_9"
}
collide_include {
  first: "limb_10"
  second: "limb_6"
}
collide_include {
  first: "limb_10"
  second: "limb_1"
}
collide_include {
  first: "limb_10"
  second: "limb_7"
}
collide_include {
  first: "limb_10"
  second: "limb_8"
}
collide_include {
  first: "limb_10"
  second: "limb_11"
}
collide_include {
  first: "limb_10"
  second: "Ground"
}
collide_include {
  first: "limb_7"
  second: "torso_0"
}
collide_include {
  first: "limb_7"
  second: "limb_0"
}
collide_include {
  first: "limb_7"
  second: "limb_4"
}
collide_include {
  first: "limb_7"
  second: "limb_9"
}
collide_include {
  first: "limb_7"
  second: "limb_6"
}
collide_include {
  first: "limb_7"
  second: "limb_5"
}
collide_include {
  first: "limb_7"
  second: "limb_10"
}
collide_include {
  first: "limb_7"
  second: "limb_8"
}
collide_include {
  first: "limb_7"
  second: "limb_11"
}
collide_include {
  first: "limb_7"
  second: "Ground"
}
collide_include {
  first: "limb_8"
  second: "limb_0"
}
collide_include {
  first: "limb_8"
  second: "limb_4"
}
collide_include {
  first: "limb_8"
  second: "limb_9"
}
collide_include {
  first: "limb_8"
  second: "limb_6"
}
collide_include {
  first: "limb_8"
  second: "limb_1"
}
collide_include {
  first: "limb_8"
  second: "limb_5"
}
collide_include {
  first: "limb_8"
  second: "limb_10"
}
collide_include {
  first: "limb_8"
  second: "limb_7"
}
collide_include {
  first: "limb_8"
  second: "limb_11"
}
collide_include {
  first: "limb_8"
  second: "Ground"
}
collide_include {
  first: "limb_11"
  second: "limb_0"
}
collide_include {
  first: "limb_11"
  second: "limb_4"
}
collide_include {
  first: "limb_11"
  second: "limb_9"
}
collide_include {
  first: "limb_11"
  second: "limb_6"
}
collide_include {
  first: "limb_11"
  second: "limb_1"
}
collide_include {
  first: "limb_11"
  second: "limb_5"
}
collide_include {
  first: "limb_11"
  second: "limb_10"
}
collide_include {
  first: "limb_11"
  second: "limb_7"
}
collide_include {
  first: "limb_11"
  second: "limb_8"
}
collide_include {
  first: "limb_11"
  second: "Ground"
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
  second: "limb_4"
}
collide_include {
  first: "Ground"
  second: "limb_9"
}
collide_include {
  first: "Ground"
  second: "limb_6"
}
collide_include {
  first: "Ground"
  second: "limb_1"
}
collide_include {
  first: "Ground"
  second: "limb_5"
}
collide_include {
  first: "Ground"
  second: "limb_10"
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
  second: "limb_11"
}
dt: 0.05000000074505806
substeps: 10
"""

_SYSTEM_CONFIG = CONFIGS
