"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _parkour_amp_env_cfg(play: bool, shoe: bool = True):
    from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg

    return instinct_g1_parkour_amp_final_cfg(play=play, shoe=shoe)


def g1_parkour_amp_instinct_rl_cfg():
    from .rl_cfg import g1_parkour_amp_instinct_rl_cfg as _cfg

    return _cfg()


def _stairs_play_env_cfg():
    from ..stairs_play_cfg import configure_stairs_play

    return configure_stairs_play(_parkour_amp_env_cfg(play=True, shoe=True))


register_instinct_task(
    task_id="Instinct-Parkour-Stairs-G1-Play-v0",
    env_cfg_factory=_stairs_play_env_cfg,
    play_env_cfg_factory=_stairs_play_env_cfg,
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)
