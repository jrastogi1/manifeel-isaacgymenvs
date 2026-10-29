# Copyright (c) 2024, NVIDIA Corporation
# All rights reserved.
#
# Generator for the XELA-sensorized Allegro-on-KUKA URDF.
#
# It takes the existing ``kuka_allegro_touch_sensor.urdf`` (7-DOF KUKA iiwa7 arm
# + 16-DOF Allegro hand) and produces ``kuka_allegro_xela.urdf`` by:
#   1. Removing the four BioTac fingertip tip links/joints.
#   2. Adding the 18 XELA uSkin sensor-patch links (see xela_sensor_config.py)
#      as fixed children of the appropriate finger/palm links, each carrying a
#      thin box collision so PhysX attributes contact forces to the patch body.
#
# The 16 finger revolute joints and 7 arm revolute joints are left untouched, so
# the resulting model keeps the same 23-DOF structure (and the same joint/link
# names) as the original asset. Only extra fixed patch bodies are appended.
#
# Run from the repo root:
#   python isaacgymenvs/tasks/allegro_kuka/gen_xela_urdf.py

import os
import re
import sys

# Import the sensor-config module directly (by path) to avoid triggering the
# isaacgymenvs package __init__, which imports torch before isaacgym.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from xela_sensor_config import (  # noqa: E402
    XELA_FLATTEN_ORDER,
    XELA_PATCH_PARENT_LINK,
)


HERE = _HERE
SRC_URDF = os.path.join(HERE, "../../../assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf")
DST_URDF = os.path.join(HERE, "../../../assets/urdf/kuka_allegro_description/kuka_allegro_xela.urdf")

# Per-patch collision box half-extents (metres) and a placement offset expressed
# in the parent-link frame. Patches use a thin box approximating the uSkin patch
# footprint; the stand-off pushes the box just outside the link surface so that
# object contacts register on the patch body (whose net contact force we read).
PATCH_BOX = {
    "aftc": {"size": (0.030, 0.039, 0.008), "offset": (0.016, 0.0, 0.0)},
    "4x4": {"size": (0.026, 0.024, 0.006), "offset": (0.027, 0.0, -0.009)},
    "4x6": {"size": (0.052, 0.032, 0.006), "offset": (0.0, 0.0, 0.02)},
}

# Distinct in-plane offsets for the three palm patches so they do not perfectly
# overlap on palm_link (spread across the palm).
PALM_INPLANE_OFFSETS = {
    "ahr_palm_1_4x6_palm_link": (0.03, 0.02, 0.0),
    "ahr_palm_2_4x6_palm_link": (0.03, -0.02, 0.0),
    "ahr_palm_3_4x6_palm_link": (0.075, 0.0, 0.0),
}


def _patch_family(patch_name: str) -> str:
    if "aftc" in patch_name:
        return "aftc"
    if "4x4" in patch_name:
        return "4x4"
    if "4x6" in patch_name:
        return "4x6"
    raise ValueError(f"Unknown patch family for {patch_name!r}")


def _patch_link_xml(patch_name: str) -> str:
    """Build the <link> + fixed <joint> XML for one XELA patch body."""
    family = _patch_family(patch_name)
    box = PATCH_BOX[family]
    sx, sy, sz = box["size"]
    ox, oy, oz = box["offset"]
    if family == "4x6":
        px, py, pz = PALM_INPLANE_OFFSETS[patch_name]
        ox, oy, oz = ox + px, oy + py, oz + pz

    parent = XELA_PATCH_PARENT_LINK[patch_name]
    # small mass so the extra body does not perturb the hand dynamics
    mass = 0.002
    inertia = 1e-6

    return f'''  <link name="{patch_name}">
    <inertial>
      <mass value="{mass}"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <inertia ixx="{inertia}" ixy="0" ixz="0" iyy="{inertia}" iyz="0" izz="{inertia}"/>
    </inertial>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
    </collision>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
      <material name="Cyan">
        <color rgba="0.0 0.8 0.8 1.0"/>
      </material>
    </visual>
  </link>
  <joint name="{patch_name}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="{ox} {oy} {oz}"/>
    <parent link="{parent}"/>
    <child link="{patch_name}"/>
  </joint>
'''


def _remove_biotac(urdf: str) -> str:
    """Strip the four BioTac tip links and their fixed joints."""
    for finger in ("index", "middle", "ring", "thumb"):
        # remove the tip <link> ... </link>
        urdf = re.sub(
            rf'\s*<link name="{finger}_biotac_tip">.*?</link>',
            "",
            urdf,
            flags=re.DOTALL,
        )
        # remove the tip fixed <joint> ... </joint>
        urdf = re.sub(
            rf'\s*<joint name="{finger}_biotac_tip_joint".*?</joint>',
            "",
            urdf,
            flags=re.DOTALL,
        )
    return urdf


def _ensure_base_link_inertial(urdf: str) -> str:
    """Give the (inertia-less) iiwa7_base_link a tiny inertial.

    The base link is fixed and has no <inertial> in the source URDF. When
    ``collapse_fixed_joints`` is disabled (as this XELA task does, so the patch
    bodies stay distinct), PhysX requires every link to have an inertial, so we
    add a negligible one to the base link.
    """
    if '<link name="iiwa7_base_link"/>' in urdf:
        replacement = (
            '<link name="iiwa7_base_link">\n'
            '    <inertial>\n'
            '      <mass value="0.001"/>\n'
            '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
            '      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>\n'
            '    </inertial>\n'
            '  </link>'
        )
        urdf = urdf.replace('<link name="iiwa7_base_link"/>', replacement)
    return urdf


def generate() -> str:
    with open(SRC_URDF, "r") as f:
        urdf = f.read()

    urdf = _remove_biotac(urdf)
    urdf = _ensure_base_link_inertial(urdf)

    patch_blocks = "\n".join(_patch_link_xml(name) for name in XELA_FLATTEN_ORDER.keys())
    marker = "</robot>"
    assert marker in urdf, "Could not find closing </robot> tag in source URDF"
    urdf = urdf.replace(marker, "  <!-- ================= XELA uSkin sensor patches ================= -->\n" + patch_blocks + "\n" + marker)

    with open(DST_URDF, "w") as f:
        f.write(urdf)

    return DST_URDF


if __name__ == "__main__":
    out = generate()
    print(f"Wrote {out}")
