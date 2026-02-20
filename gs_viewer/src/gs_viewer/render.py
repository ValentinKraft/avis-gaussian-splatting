"""OpenGL rendering utilities.

Uses weighted blended order-independent transparency (OIT):
- Accum buffer: sum(color * alpha) and sum(alpha)
- Revealage: multiplicative (1 - alpha)

Composite:
  out_rgb = accum_rgb / max(accum_a, eps)
  out_a   = 1 - revealage
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from gs_viewer.ply_loader import GaussianModelPly


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if status != GL.GL_TRUE:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader compile failed: {log}")
    return shader


def _link_program(vs: int, fs: int) -> int:
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if status != GL.GL_TRUE:
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"Program link failed: {log}")
    return program


_SPLAT_VS = """#version 330 core

layout(location=0) in vec3 a_pos;
layout(location=1) in float a_opacity;
layout(location=2) in float a_intensity;
layout(location=3) in float a_sigma;
layout(location=4) in float a_ao;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_focal_px;
uniform float u_max_point_size;
uniform float u_splat_scale;

out float v_opacity;
out float v_intensity;
out float v_sigma;
out float v_ao;

void main() {
    vec4 viewPos = u_view * vec4(a_pos, 1.0);
    float z = max(-viewPos.z, 1e-3);

    // Approximate: point size covers ~3 sigma radius.
    float sigma_px = (a_sigma * u_focal_px * u_splat_scale) / z;
    float pointSize = clamp(6.0 * sigma_px, 1.0, u_max_point_size);

    gl_Position = u_proj * viewPos;
    gl_PointSize = pointSize;

    v_opacity = a_opacity;
    v_intensity = a_intensity;
    v_sigma = a_sigma;
    v_ao = a_ao;
}
"""


_SPLAT_FS = """#version 330 core

layout(location=0) out vec4 out_accum;
layout(location=1) out vec4 out_reveal;

in float v_opacity;
in float v_intensity;
in float v_ao;

uniform sampler1D u_lut;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5, 0.5);
    float r = length(d);

    // r is in [0, ~0.707]. Treat the sprite as a 3-sigma radius.
    float t = 6.0 * r;
    float alpha_gauss = exp(-0.5 * t * t);

    vec4 lut = texture(u_lut, clamp(v_intensity, 0.0, 1.0));

    float alpha = clamp(v_opacity, 0.0, 1.0) * lut.a * alpha_gauss;
    vec3 rgb = lut.rgb;

    // Optional AO modulation if provided (a_ao defaults to 1).
    rgb *= clamp(v_ao, 0.0, 1.0);

    // Weighted blended OIT (simple weight).
    vec3 premul = rgb * alpha;
    out_accum = vec4(premul, alpha);

    // Revealage starts at 1 and is multiplied by (1 - alpha).
    out_reveal = vec4(1.0, 1.0, 1.0, alpha);
}
"""


_COMPOSE_VS = """#version 330 core

const vec2 verts[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

out vec2 v_uv;

void main() {
    vec2 p = verts[gl_VertexID];
    v_uv = 0.5 * (p + vec2(1.0, 1.0));
    gl_Position = vec4(p, 0.0, 1.0);
}
"""


_COMPOSE_FS = """#version 330 core

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_accum;
uniform sampler2D u_reveal;

void main() {
    vec4 accum = texture(u_accum, v_uv);
    float reveal = texture(u_reveal, v_uv).r;

    float a = clamp(accum.a, 0.0, 1e9);
    vec3 rgb = (a > 1e-6) ? (accum.rgb / a) : vec3(0.0);
    float outA = 1.0 - clamp(reveal, 0.0, 1.0);

    out_color = vec4(rgb, outA);
}
"""


@dataclass
class _GpuBuffers:
    vao: int
    vbo_pos: int
    vbo_opacity: int
    vbo_intensity: int
    vbo_sigma: int
    vbo_ao: int
    count: int


class OitRenderer:
    """Weighted blended OIT renderer."""

    def __init__(self) -> None:
        self._width: int = 0
        self._height: int = 0

        self._fbo: int = 0
        self._tex_accum: int = 0
        self._tex_reveal: int = 0

        self._splat_prog: int = 0
        self._compose_prog: int = 0

        self._fs_vao: int = 0

        self._gpu: _GpuBuffers | None = None
        self._current_model_id: int | None = None

        self._init_programs()
        self._fs_vao = GL.glGenVertexArrays(1)

    def ensure_size(self, width: int, height: int) -> None:
        if width == self._width and height == self._height:
            return
        self._width, self._height = width, height
        self._create_or_resize_targets(width, height)

    def begin_frame(self) -> None:
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self._width, self._height)

        # Clear accum to 0 and revealage to 1.
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        GL.glClearBufferfv(GL.GL_COLOR, 0, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        GL.glClearBufferfv(GL.GL_COLOR, 1, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    def render_splats(
        self,
        model: GaussianModelPly,
        view: np.ndarray,
        proj: np.ndarray,
        lut_tex_id: int,
        splat_scale: float = 1.0,
    ) -> None:
        self._ensure_model_uploaded(model)
        assert self._gpu is not None

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])

        GL.glUseProgram(self._splat_prog)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

        loc_view = GL.glGetUniformLocation(self._splat_prog, "u_view")
        loc_proj = GL.glGetUniformLocation(self._splat_prog, "u_proj")
        GL.glUniformMatrix4fv(loc_view, 1, GL.GL_TRUE, view.astype(np.float32))
        GL.glUniformMatrix4fv(loc_proj, 1, GL.GL_TRUE, proj.astype(np.float32))

        # Approximate focal length in pixels.
        focal_px = 0.5 * float(self._height) / float(np.tan(np.deg2rad(45.0) / 2.0))
        GL.glUniform1f(GL.glGetUniformLocation(self._splat_prog, "u_focal_px"), focal_px)
        GL.glUniform1f(GL.glGetUniformLocation(self._splat_prog, "u_max_point_size"), 256.0)
        GL.glUniform1f(
            GL.glGetUniformLocation(self._splat_prog, "u_splat_scale"),
            max(0.001, float(splat_scale)),
        )

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_1D, lut_tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._splat_prog, "u_lut"), 0)

        # Blending per attachment.
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_FUNC_ADD)

        # Accum: additive.
        GL.glBlendFunci(0, GL.GL_ONE, GL.GL_ONE)

        # Revealage: dst = dst * (1 - src_alpha)
        GL.glBlendFunci(1, GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glBindVertexArray(self._gpu.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self._gpu.count)
        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)

    def composite_to_screen(self, width: int, height: int) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.05, 0.05, 0.05, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self._compose_prog)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_accum)
        GL.glUniform1i(GL.glGetUniformLocation(self._compose_prog, "u_accum"), 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_reveal)
        GL.glUniform1i(GL.glGetUniformLocation(self._compose_prog, "u_reveal"), 1)

        # Core profile requires a VAO bound for glDrawArrays.
        GL.glBindVertexArray(self._fs_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

    def _init_programs(self) -> None:
        vs = _compile_shader(_SPLAT_VS, GL.GL_VERTEX_SHADER)
        fs = _compile_shader(_SPLAT_FS, GL.GL_FRAGMENT_SHADER)
        self._splat_prog = _link_program(vs, fs)

        cvs = _compile_shader(_COMPOSE_VS, GL.GL_VERTEX_SHADER)
        cfs = _compile_shader(_COMPOSE_FS, GL.GL_FRAGMENT_SHADER)
        self._compose_prog = _link_program(cvs, cfs)

    def _create_or_resize_targets(self, width: int, height: int) -> None:
        if self._fbo == 0:
            self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        if self._tex_accum == 0:
            self._tex_accum = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_accum)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, width, height, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._tex_accum, 0)

        if self._tex_reveal == 0:
            self._tex_reveal = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_reveal)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16F, width, height, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self._tex_reveal, 0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _ensure_model_uploaded(self, model: GaussianModelPly) -> None:
        model_id = id(model.positions)
        if self._current_model_id == model_id and self._gpu is not None:
            return

        # Prepare attributes.
        positions = model.positions.astype(np.float32)
        opacity = model.opacity.astype(np.float32)
        intensity = model.intensity01.astype(np.float32)

        sigma = np.exp(model.log_scale).mean(axis=1).astype(np.float32)

        if model.ao is None:
            ao = np.ones((model.count,), dtype=np.float32)
        else:
            ao = model.ao.astype(np.float32)

        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        vbo_pos = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, positions.nbytes, positions, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        vbo_op = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_op)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, opacity.nbytes, opacity, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 1, GL.GL_FLOAT, False, 0, None)

        vbo_int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_int)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, intensity.nbytes, intensity, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 1, GL.GL_FLOAT, False, 0, None)

        vbo_sigma = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_sigma)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, sigma.nbytes, sigma, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, False, 0, None)

        vbo_ao = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_ao)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, ao.nbytes, ao, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribPointer(4, 1, GL.GL_FLOAT, False, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        self._gpu = _GpuBuffers(
            vao=vao,
            vbo_pos=vbo_pos,
            vbo_opacity=vbo_op,
            vbo_intensity=vbo_int,
            vbo_sigma=vbo_sigma,
            vbo_ao=vbo_ao,
            count=model.count,
        )
        self._current_model_id = model_id
