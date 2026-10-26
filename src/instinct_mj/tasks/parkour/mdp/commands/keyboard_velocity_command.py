"""Keyboard velocity commands for standalone parkour playback."""

from __future__ import annotations

from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING

import torch
from mjlab.viewer.native.keys import KEY_KP_2, KEY_KP_4, KEY_KP_5, KEY_KP_6, KEY_KP_8

from instinct_mj.managers import CommandTerm

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands_cfg import KeyboardVelocityCommandCfg

KEYBOARD_HELP = (
    "Keypad 8/2: increase/decrease forward speed; keypad 4/6: turn left/right; "
    "keypad 5: stop; Enter: reset; Space: pause. Enable Num Lock. Commands persist until changed."
)

_KEY_ACTIONS = {
    KEY_KP_8: "speed_up",
    KEY_KP_2: "speed_down",
    KEY_KP_4: "turn_left",
    KEY_KP_6: "turn_right",
    KEY_KP_5: "stop",
}


class KeyboardVelocityCommand(CommandTerm):
    """Latch forward speed and yaw rate from native-viewer key presses."""

    cfg: KeyboardVelocityCommandCfg

    def __init__(self, cfg: KeyboardVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError("Keyboard playback requires exactly one environment.")
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._keys: SimpleQueue[str] = SimpleQueue()
        self._forward_speed = 0.0
        self._yaw_rate = 0.0

    @property
    def command(self) -> torch.Tensor:
        """Return [forward speed, lateral speed, yaw rate] in the base frame."""
        return self._command

    def handle_key(self, key: int) -> None:
        """Queue viewer input without touching simulation state on its thread."""
        action = _KEY_ACTIONS.get(key)
        if action is not None:
            self._keys.put(action)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Reset to a stopped command after falls and manual resets. The long
        # resampling interval prevents periodic changes during normal play.
        self._forward_speed = 0.0
        self._yaw_rate = 0.0
        self._command[env_ids] = 0.0
        while True:
            try:
                self._keys.get_nowait()
            except Empty:
                break

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        previous_command = (self._forward_speed, self._yaw_rate)
        while True:
            try:
                action = self._keys.get_nowait()
            except Empty:
                break
            if action == "speed_up":
                self._forward_speed += self.cfg.speed_step
            elif action == "speed_down":
                self._forward_speed -= self.cfg.speed_step
            elif action == "turn_left":
                self._yaw_rate += self.cfg.yaw_step
            elif action == "turn_right":
                self._yaw_rate -= self.cfg.yaw_step
            elif action == "stop":
                self._forward_speed = self._yaw_rate = 0.0
            self._forward_speed = min(max(self._forward_speed, 0.0), self.cfg.max_forward_speed)
            self._yaw_rate = min(max(self._yaw_rate, -self.cfg.max_yaw_rate), self.cfg.max_yaw_rate)
        if (self._forward_speed, self._yaw_rate) != previous_command:
            self._command[:, 0] = self._forward_speed
            self._command[:, 2] = self._yaw_rate
            print(f"[Keyboard] Forward: {self._forward_speed:.1f} m/s; turn: {self._yaw_rate:.1f} rad/s")
