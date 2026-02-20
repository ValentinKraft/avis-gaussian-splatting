"""Main viewer loop, OpenGL renderer, and UI.

The rendering implementation is intentionally minimal:
- Splats are rendered as point sprites (GL_POINTS) with a Gaussian falloff.
- Transparency uses weighted blended OIT (no sorting).
- A 1D transfer-function LUT maps per-splat intensity to color/alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL

from gs_viewer.camera import OrbitCamera
from gs_viewer.ply_loader import GaussianModelPly, load_gaussian_model_ply
from gs_viewer.render import OitRenderer
from gs_viewer.transfer_function import TransferFunction


@dataclass
class _UiState:
    ply_path: str = ""
    error_text: str = ""
    splat_scale: float = 1.0


class Viewer:
    """Interactive OpenGL viewer."""

    def __init__(self, initial_ply_path: str = "") -> None:
        self._window: Any | None = None
        self._imgui: GlfwRenderer | None = None

        self._ui = _UiState(ply_path=initial_ply_path)

        self._camera = OrbitCamera()
        self._renderer: OitRenderer | None = None
        self._tf: TransferFunction | None = None

        self._model: GaussianModelPly | None = None
        self._drag_last_x: float | None = None
        self._drag_last_y: float | None = None
        self._drag_button: int | None = None

    def run(self) -> None:
        """Create window and enter the render loop."""

        self._init_window()
        self._init_gl()
        self._init_imgui()

        if self._ui.ply_path:
            self._try_load_ply(self._ui.ply_path)

        while not glfw.window_should_close(self._window):
            glfw.poll_events()

            width, height = glfw.get_framebuffer_size(self._window)
            if width <= 0 or height <= 0:
                continue

            self._renderer.ensure_size(width, height)

            self._process_camera_input()

            self._renderer.begin_frame()
            if self._model is not None:
                view = self._camera.view_matrix()
                proj = self._camera.proj_matrix(width / float(height))
                if self._tf is not None:
                    self._renderer.render_splats(
                        self._model,
                        view,
                        proj,
                        self._tf.lut_texture_id,
                        self._ui.splat_scale,
                    )
            self._renderer.composite_to_screen(width, height)

            self._render_ui(width, height)

            glfw.swap_buffers(self._window)

        self._shutdown()

    def _init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self._window = glfw.create_window(1280, 720, "GS Viewer", None, None)
        if self._window is None:
            raise RuntimeError("glfw.create_window() failed")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        glfw.set_scroll_callback(self._window, self._on_scroll)

    def _init_gl(self) -> None:
        GL.glEnable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)

        self._renderer = OitRenderer()
        self._tf = TransferFunction()
        self._tf.ensure_gl()

    def _init_imgui(self) -> None:
        imgui.create_context()
        self._imgui = GlfwRenderer(self._window)

    def _shutdown(self) -> None:
        if self._imgui is not None:
            self._imgui.shutdown()
        glfw.terminate()

    def _try_load_ply(self, ply_path: str) -> None:
        try:
            ply_path_obj = Path(ply_path)
            if not ply_path_obj.exists():
                raise FileNotFoundError(str(ply_path_obj))

            model = load_gaussian_model_ply(ply_path_obj)
            model = model.normalized_for_view()

            self._model = model
            self._camera.frame_bounds(model.bounds_center, model.bounds_radius)

            self._ui.error_text = ""
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._ui.error_text = f"Failed to load PLY: {exc}"

    def _process_camera_input(self) -> None:
        if imgui.get_io().want_capture_mouse:
            self._drag_last_x = None
            self._drag_last_y = None
            self._drag_button = None
            return

        x, y = glfw.get_cursor_pos(self._window)

        left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        if not left and not right:
            self._drag_last_x = None
            self._drag_last_y = None
            self._drag_button = None
            return

        button = glfw.MOUSE_BUTTON_LEFT if left else glfw.MOUSE_BUTTON_RIGHT
        if self._drag_button is None:
            self._drag_button = button

        if self._drag_last_x is None:
            self._drag_last_x, self._drag_last_y = x, y
            return

        dx = float(x - self._drag_last_x)
        dy = float(y - self._drag_last_y)
        self._drag_last_x, self._drag_last_y = x, y

        if button == glfw.MOUSE_BUTTON_LEFT:
            self._camera.orbit(dx, dy)
        else:
            self._camera.pan(dx, dy)

    def _on_scroll(self, _window: Any, _xoff: float, yoff: float) -> None:
        if imgui.get_io().want_capture_mouse:
            return
        self._camera.zoom(float(yoff))

    def _render_ui(self, width: int, height: int) -> None:
        assert self._imgui is not None

        self._imgui.process_inputs()
        imgui.new_frame()

        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(420, 520)
        imgui.begin("Controls", True)

        changed, self._ui.ply_path = imgui.input_text("PLY path", self._ui.ply_path, 1024)
        if imgui.button("Load"):
            self._try_load_ply(self._ui.ply_path)

        imgui.same_line()
        if imgui.button("Reset camera") and self._model is not None:
            self._camera.frame_bounds(self._model.bounds_center, self._model.bounds_radius)

        if self._ui.error_text:
            imgui.text_colored(self._ui.error_text, 1.0, 0.3, 0.3, 1.0)

        if self._model is not None:
            imgui.separator()
            imgui.text(f"Splats: {self._model.count}")
            imgui.text(f"Bounds radius: {self._model.bounds_radius:.4f}")
            _, self._ui.splat_scale = imgui.slider_float(
                "Splat scale",
                self._ui.splat_scale,
                0.1,
                50.0,
            )

        imgui.separator()
        imgui.text("Transfer Function")
        if self._tf is not None:
            self._tf.draw_imgui()

        imgui.end()

        imgui.render()
        GL.glViewport(0, 0, width, height)
        self._imgui.render(imgui.get_draw_data())
