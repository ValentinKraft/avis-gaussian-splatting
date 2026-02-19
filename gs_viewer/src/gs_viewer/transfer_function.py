"""1D transfer function (LUT) editor.

MVP UI:
- A small set of control points (value in [0,1] -> RGBA).
- LUT is baked to a 1D OpenGL texture.
"""

from __future__ import annotations

from dataclasses import dataclass

import imgui
import numpy as np
from OpenGL import GL


@dataclass
class TfPoint:
    value: float
    rgba: tuple[float, float, float, float]


class TransferFunction:
    """Transfer function with OpenGL 1D LUT texture."""

    def __init__(self, size: int = 256) -> None:
        self._size = int(size)
        self._points: list[TfPoint] = [
            TfPoint(0.0, (0.0, 0.0, 0.0, 0.0)),
            TfPoint(0.25, (0.2, 0.2, 0.2, 0.1)),
            TfPoint(0.5, (0.6, 0.6, 0.6, 0.4)),
            TfPoint(0.75, (0.9, 0.9, 0.9, 0.7)),
            TfPoint(1.0, (1.0, 1.0, 1.0, 1.0)),
        ]

        self._lut_rgba = self._bake_lut()

        # OpenGL texture is created lazily once an OpenGL context exists.
        self._tex_id: int = 0

    @property
    def lut_texture_id(self) -> int:
        return int(self._tex_id)

    def ensure_gl(self) -> None:
        """Ensure the 1D LUT OpenGL texture exists.

        Must be called after a valid OpenGL context is current.
        """

        if self._tex_id != 0:
            return
        self._tex_id = self._create_texture(self._lut_rgba)

    def draw_imgui(self) -> None:
        """Draw TF editor controls and update LUT if changed."""

        changed_any = False

        imgui.text("Control points")
        for i, p in enumerate(list(self._points)):
            imgui.push_id(f"tfpt_{i}")

            changed, value = imgui.slider_float("value", p.value, 0.0, 1.0)
            if changed:
                self._points[i].value = float(value)
                changed_any = True

            r, g, b, a = p.rgba
            changed, color = imgui.color_edit4("rgba", r, g, b, a)
            if changed:
                self._points[i].rgba = (float(color[0]), float(color[1]), float(color[2]), float(color[3]))
                changed_any = True

            imgui.separator()
            imgui.pop_id()

        # Keep endpoints pinned.
        self._points[0].value = 0.0
        self._points[-1].value = 1.0

        # Ensure strict sort by value.
        self._points.sort(key=lambda pt: pt.value)

        if changed_any:
            self._lut_rgba = self._bake_lut()
            if self._tex_id != 0:
                self._update_texture(self._lut_rgba)

    def _bake_lut(self) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, self._size, dtype=np.float32)
        lut = np.zeros((self._size, 4), dtype=np.float32)

        values = np.array([p.value for p in self._points], dtype=np.float32)
        colors = np.array([p.rgba for p in self._points], dtype=np.float32)

        for c in range(4):
            lut[:, c] = np.interp(xs, values, colors[:, c]).astype(np.float32)

        return np.clip(lut, 0.0, 1.0).astype(np.float32)

    def _create_texture(self, lut: np.ndarray) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, tex)
        GL.glTexImage1D(GL.GL_TEXTURE_1D, 0, GL.GL_RGBA8, lut.shape[0], 0, GL.GL_RGBA, GL.GL_FLOAT, lut)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
        return int(tex)

    def _update_texture(self, lut: np.ndarray) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_1D, self._tex_id)
        GL.glTexSubImage1D(GL.GL_TEXTURE_1D, 0, 0, lut.shape[0], GL.GL_RGBA, GL.GL_FLOAT, lut)
        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
