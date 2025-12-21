import sys
import numpy as np
import pygltflib
import random
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import queue
import threading
import time
from pathlib import Path
import ctypes
import collections

DEBUG = False
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TARGET_FPS = 60.0
WATER_UPDATE_HZ = 30.0
LIGHT_RAIN_ENABLED = True
NOTE_ACTIVITY_WINDOW = 0.35
NOTE_ACTIVITY_REF_DENSITY = 12.0
PIANO_SCALE = 4.0
PIANO_MODEL_SCALE = 2.0
STRING_GLOW_COLOR = (1.0, 0.9, 0.2)
STRING_BASE_COLOR = (0.12, 0.18, 0.28)
STRING_GLOW_INTENSITY = 4.0
STRING_LINE_WIDTH = 4.5
STRING_GLOW_THICKNESS = 3.0
STRING_FRONT_RATIO = 0.22
STRING_BACK_RATIO = 0.45
KEY_VISUAL_LIFT = 0.02

UI_PADDING = 20
UI_BUTTON_SIZE = 44
UI_BUTTON_GAP = 10

model = None
piano_display_list = None
last_x, last_y = 0, 0
rot_x, rot_y = 25.0, -45.0
mouse_down = False
window_width = WINDOW_WIDTH
window_height = WINDOW_HEIGHT

cam_pan_x, cam_pan_y = 0.0, 0.0
cam_pan_z = 0.0
pan_speed = 0.5
cam_distance = 50.0
zoom_speed = 0.5

# MIDI事件隊列（用於接收外部事件）
midi_event_queue = queue.Queue()
skybox = None
ui_state = {"playing": False, "track": "No track"}
ui_callbacks = {"play_pause": None, "next": None}
string_display_list = None
PIANO_OFFSET_X = 0.0
PIANO_OFFSET_Z = 0.0


def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def set_ui_state(playing=None, track_name=None):
    if playing is not None:
        ui_state["playing"] = playing
    if track_name is not None:
        ui_state["track"] = track_name


def set_ui_callbacks(play_pause=None, next_track=None):
    if play_pause is not None:
        ui_callbacks["play_pause"] = play_pause
    if next_track is not None:
        ui_callbacks["next"] = next_track

# ============================================================
# 琴鍵類別 - 使用調整後的參數
# ============================================================
class PianoKeys:
    def __init__(self):
        # 白鍵參數 - 使用調整後的數值
        self.white_key_width = 0.100
        self.white_key_length = 0.730
        self.white_key_height = 0.150
        self.white_key_y_offset = 2.350
        self.white_key_z_offset = -0.020
        self.white_key_spacing = 0.110
        self.white_key_start_x = 0.220
        self.white_key_press_depth = 0.05

        # 黑鍵參數 - 使用調整後的數值
        self.black_key_width = 0.100
        self.black_key_length = 0.470
        self.black_key_height = 0.200
        self.black_key_y_offset = 2.350
        self.black_key_z_offset = -0.230
        self.black_key_press_depth = 0.07

        self.align_scale_x = 1.0
        self.align_offset_x = 0.0
        self.align_offset_y = 0.0
        self.align_offset_z = 0.0

        # 白鍵數量（88鍵鋼琴有52個白鍵）
        self.num_white_keys = 52
        
        # 黑鍵位置模式（從A開始的白鍵序列：A B C D E F G）
        self.black_key_pattern = [1, 0, 1, 1, 0, 1, 1]
        
        # 琴鍵狀態（press_depth: 0.0~1.0）
        self.key_states = {}
        
        # 琴鍵映射：MIDI note (0-87) -> 琴鍵類型和索引
        self.build_key_mapping()
        self.white_states = np.zeros(self.num_white_keys, dtype=np.float32)
        self.black_states = np.zeros(self.num_black_keys, dtype=np.float32)
        
    def build_key_mapping(self):
        """建立MIDI note到琴鍵的映射
        88鍵鋼琴從A0(MIDI 21)開始到C8(MIDI 108)
        """
        self.note_to_key = {}  # MIDI note -> (key_type, key_index)
        self.note_to_x = {}  # MIDI note -> world x
        
        # 88鍵的音符模式（從A0開始）
        # A0, A#0, B0, 然後是7個完整的八度（C1-B7），最後C8
        note_pattern = ['A', 'A#', 'B']  # 前3個音
        
        # 7個完整八度
        for _ in range(7):
            note_pattern.extend(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        
        # 最後一個C8
        note_pattern.append('C')
        
        white_idx = 0
        black_idx = 0
        
        for i, note_name in enumerate(note_pattern):
            midi_note = i  # 相對於A0的索引
            
            if '#' in note_name:  # 黑鍵
                self.note_to_key[midi_note] = ('black', black_idx)
                if white_idx > 0:
                    x_pos = self.white_key_start_x + (white_idx - 1) * self.white_key_spacing
                    x_pos += self.white_key_spacing * 0.5
                else:
                    x_pos = self.white_key_start_x
                self.note_to_x[midi_note] = x_pos
                black_idx += 1
            else:  # 白鍵
                self.note_to_key[midi_note] = ('white', white_idx)
                x_pos = self.white_key_start_x + white_idx * self.white_key_spacing
                self.note_to_x[midi_note] = x_pos
                white_idx += 1
        self.num_black_keys = black_idx
        key_span = (self.num_white_keys - 1) * self.white_key_spacing
        self.center_x = self.white_key_start_x + key_span * 0.5
        self.center_z = self.white_key_z_offset

    def set_alignment(self, scale_x=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0):
        self.align_scale_x = float(scale_x)
        self.align_offset_x = float(offset_x)
        self.align_offset_y = float(offset_y)
        self.align_offset_z = float(offset_z)

    def key_span_x(self):
        min_x = self.white_key_start_x
        max_x = self.white_key_start_x + (self.num_white_keys - 1) * self.white_key_spacing
        return min_x, max_x

    def note_world_x(self, note):
        """Return world x position for a MIDI note (21-108)."""
        relative_note = note - 21
        x = self.note_to_x.get(relative_note)
        if x is None:
            return None
        return self.align_offset_x + self.align_scale_x * x

    def note_is_black(self, note):
        relative_note = note - 21
        mapping = self.note_to_key.get(relative_note)
        if not mapping:
            return False
        return mapping[0] == "black"
    
    def press_key(self, note, velocity):
        """按下琴鍵
        note: MIDI note number (21-108)
        velocity: 力度 (0-127)
        """
        # 轉換為相對索引 (0-87)
        relative_note = note - 21
        if 0 <= relative_note < 88:
            if relative_note in self.note_to_key:
                # 計算按壓深度（根據velocity）
                press_depth = min(velocity / 127.0, 1.0)
                key_type, key_idx = self.note_to_key[relative_note]
                if key_type == 'white':
                    self.white_states[key_idx] = press_depth
                else:
                    self.black_states[key_idx] = press_depth
                self.key_states[relative_note] = press_depth
                return True
        return False
    
    def release_key(self, note):
        """釋放琴鍵
        note: MIDI note number (21-108)
        """
        relative_note = note - 21
        if relative_note in self.note_to_key:
            key_type, key_idx = self.note_to_key[relative_note]
            if key_type == 'white':
                self.white_states[key_idx] = 0.0
            else:
                self.black_states[key_idx] = 0.0
            self.key_states.pop(relative_note, None)
            return True
        return False
    
    def update_key_animation(self, dt=0.016):
        """更新琴鍵動畫（平滑回彈效果）
        dt: 時間增量（秒）
        """
        # 目前使用即時按下/釋放，不需要額外動畫
        pass
    
    def draw_box(self, width, height, length):
        """繪製一個中心在原點的長方體"""
        w, h, l = width/2, height/2, length/2
        
        glBegin(GL_QUADS)
        # 前面
        glNormal3f(0, 0, 1)
        glVertex3f(-w, -h, l)
        glVertex3f(w, -h, l)
        glVertex3f(w, h, l)
        glVertex3f(-w, h, l)
        
        # 後面
        glNormal3f(0, 0, -1)
        glVertex3f(-w, -h, -l)
        glVertex3f(-w, h, -l)
        glVertex3f(w, h, -l)
        glVertex3f(w, -h, -l)
        
        # 上面
        glNormal3f(0, 1, 0)
        glVertex3f(-w, h, -l)
        glVertex3f(-w, h, l)
        glVertex3f(w, h, l)
        glVertex3f(w, h, -l)
        
        # 下面
        glNormal3f(0, -1, 0)
        glVertex3f(-w, -h, -l)
        glVertex3f(w, -h, -l)
        glVertex3f(w, -h, l)
        glVertex3f(-w, -h, l)
        
        # 左面
        glNormal3f(-1, 0, 0)
        glVertex3f(-w, -h, -l)
        glVertex3f(-w, -h, l)
        glVertex3f(-w, h, l)
        glVertex3f(-w, h, -l)
        
        # 右面
        glNormal3f(1, 0, 0)
        glVertex3f(w, -h, -l)
        glVertex3f(w, h, -l)
        glVertex3f(w, h, l)
        glVertex3f(w, -h, l)
        glEnd()
    
    def draw(self):
        """繪製所有琴鍵"""
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glPushMatrix()
        if (
            self.align_scale_x != 1.0
            or self.align_offset_x != 0.0
            or self.align_offset_y != 0.0
            or self.align_offset_z != 0.0
        ):
            glTranslatef(self.align_offset_x, self.align_offset_y, self.align_offset_z)
            glScalef(self.align_scale_x, 1.0, 1.0)
        depth_scale = 1.0 / max(PIANO_SCALE, 1e-6)
        
        # 繪製白鍵
        for i in range(self.num_white_keys):
            glPushMatrix()
            x_pos = self.white_key_start_x + i * self.white_key_spacing
            glTranslatef(x_pos, self.white_key_y_offset, self.white_key_z_offset)
            
            # 檢查是否被按下
            press_depth = float(self.white_states[i])
            
            # 按下時下移
            if press_depth > 0:
                glTranslatef(0, -self.white_key_press_depth * press_depth * depth_scale, 0)
                # 根據按壓力度調整顏色
                brightness = 1.0 - press_depth * 0.2
                glColor3f(brightness, brightness, brightness + 0.05)
            else:
                glColor3f(1.0, 1.0, 1.0)
            
            self.draw_box(self.white_key_width, self.white_key_height, self.white_key_length)
            glPopMatrix()
        
        # 繪製黑鍵
        black_key_count = 0
        for i in range(self.num_white_keys - 1):
            pattern_index = i % len(self.black_key_pattern)
            if self.black_key_pattern[pattern_index] == 1:
                if black_key_count >= self.num_black_keys:
                    break
                glPushMatrix()
                x_pos = self.white_key_start_x + i * self.white_key_spacing + self.white_key_spacing / 2
                glTranslatef(x_pos, self.black_key_y_offset, self.black_key_z_offset)
                
                # 檢查是否被按下
                press_depth = float(self.black_states[black_key_count])
                
                if press_depth > 0:
                    glTranslatef(0, -self.black_key_press_depth * press_depth * depth_scale, 0)
                    brightness = 0.15 + press_depth * 0.45
                    glColor3f(brightness, brightness, brightness + 0.1)
                else:
                    glColor3f(0.1, 0.1, 0.1)
                
                self.draw_box(self.black_key_width, self.black_key_height, self.black_key_length)
                glPopMatrix()
                
                black_key_count += 1
        
        glPopMatrix()
        glColor3f(1, 1, 1)

piano_keys = PianoKeys()
PIANO_OFFSET_X = -piano_keys.center_x
PIANO_OFFSET_Z = -piano_keys.center_z

# ============================================================
# 水面模擬類別 - 優化版本
# ============================================================
class WaterRipple:
    def __init__(self, cols=240, rows=240, size=50.0):
        # 適中網格密度，搭配向量化更新提升穩定性與幀率
        self.cols = cols
        self.rows = rows
        self.size = size
        self.spacing = size / cols

        self.buffer1 = np.zeros((cols, rows), dtype=np.float32)
        self.buffer2 = np.zeros((cols, rows), dtype=np.float32)
        self.damping = 0.99

        self.scale_h = 0.07
        self._dy = self.spacing * 2.0
        self._base_blue = np.array([0.04, 0.1, 0.2], dtype=np.float32)
        self._peak_color = np.array([0.22, 0.35, 0.5], dtype=np.float32)
        self._specular_color = np.array([0.1, 0.14, 0.2], dtype=np.float32)
        self._alpha_base = 0.82
        self._alpha_scale = 0.14
        self._alpha_max = 0.92
        self._abs_heights = np.zeros((self.cols, self.rows), dtype=np.float32)
        self._specular = np.zeros((self.cols, self.rows), dtype=np.float32)
        self._texel = np.array(
            [1.0 / (self.cols - 1), 1.0 / (self.rows - 1)], dtype=np.float32
        )
        self._kernel_cache = {}
        self._default_kernel_radius = 4

        self._use_gpu = True
        self._gpu_ready = False
        self._program = None
        self._vbo = None
        self._ebo = None
        self._height_tex = None
        self._attrib_pos = None
        self._attrib_uv = None
        self._uni_height = None
        self._uni_height_scale = None
        self._uni_texel = None
        self._uni_normal_y = None
        self._uni_base_color = None
        self._uni_peak_color = None
        self._uni_spec_color = None
        self._uni_alpha_base = None
        self._uni_alpha_scale = None
        self._uni_alpha_max = None
        self._height_format = GL_RED

        self._build_mesh()

    def _build_mesh(self):
        xs = np.arange(self.cols, dtype=np.float32) * self.spacing
        zs = np.arange(self.rows, dtype=np.float32) * self.spacing
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
        denom_x = self.spacing * (self.cols - 1)
        denom_z = self.spacing * (self.rows - 1)
        grid_u = grid_x / denom_x if denom_x > 0 else grid_x
        grid_v = grid_z / denom_z if denom_z > 0 else grid_z

        self._vertices = np.zeros((self.cols, self.rows, 3), dtype=np.float32)
        self._vertices[..., 0] = grid_x
        self._vertices[..., 2] = grid_z
        self._vertices_flat = self._vertices.reshape(-1, 3)

        self._normals = np.zeros_like(self._vertices)
        self._normals[..., 1] = 1.0
        self._normals_flat = self._normals.reshape(-1, 3)

        self._colors = np.zeros((self.cols, self.rows, 4), dtype=np.float32)
        self._colors[..., 3] = self._alpha_base
        self._colors_flat = self._colors.reshape(-1, 4)
        self._blend = np.zeros((self.cols, self.rows), dtype=np.float32)
        self._inv_blend = np.zeros((self.cols, self.rows), dtype=np.float32)
        self._uvs = np.zeros((self.cols, self.rows, 2), dtype=np.float32)
        self._uvs[..., 0] = grid_u
        self._uvs[..., 1] = grid_v
        self._uvs_flat = self._uvs.reshape(-1, 2)

        i = np.arange(self.cols - 1, dtype=np.uint32)[:, None]
        j = np.arange(self.rows - 1, dtype=np.uint32)[None, :]
        tl = i * self.rows + j
        tr = (i + 1) * self.rows + j
        br = (i + 1) * self.rows + (j + 1)
        bl = i * self.rows + (j + 1)
        indices = np.stack((tl, tr, br, tl, br, bl), axis=-1).reshape(-1)
        self._indices = indices.astype(np.uint32)
        self._index_count = int(self._indices.size)
        self._vertex_data = np.zeros((self.cols * self.rows, 4), dtype=np.float32)
        self._vertex_data[:, 0] = self._vertices_flat[:, 0]
        self._vertex_data[:, 1] = self._vertices_flat[:, 2]
        self._vertex_data[:, 2] = self._uvs_flat[:, 0]
        self._vertex_data[:, 3] = self._uvs_flat[:, 1]
        self._vertex_stride = int(self._vertex_data.strides[0])

    def init_gl(self):
        if not self._use_gpu or self._gpu_ready:
            return
        try:
            from OpenGL.GL import shaders
        except Exception:
            self._use_gpu = False
            return

        vertex_src = """
        #version 120
        attribute vec2 a_pos;
        attribute vec2 a_uv;
        uniform sampler2D u_height;
        uniform float u_height_scale;
        uniform vec2 u_texel;
        uniform float u_normal_y;
        uniform vec3 u_base_color;
        uniform vec3 u_peak_color;
        uniform vec3 u_spec_color;
        uniform float u_alpha_base;
        uniform float u_alpha_scale;
        uniform float u_alpha_max;
        varying vec3 v_normal;
        varying vec3 v_color;
        varying float v_alpha;
        void main() {
            float h = texture2D(u_height, a_uv).r;
            float hL = texture2D(u_height, a_uv + vec2(-u_texel.x, 0.0)).r;
            float hR = texture2D(u_height, a_uv + vec2(u_texel.x, 0.0)).r;
            float hD = texture2D(u_height, a_uv + vec2(0.0, -u_texel.y)).r;
            float hU = texture2D(u_height, a_uv + vec2(0.0, u_texel.y)).r;
            float y = h * u_height_scale;
            vec3 normal = normalize(vec3(hL - hR, u_normal_y, hD - hU));
            v_normal = normal;
            float blend = clamp(y + 0.5, 0.0, 1.0);
            vec3 color = mix(u_base_color, u_peak_color, blend);
            float spec = pow(clamp(normal.y, 0.0, 1.0), 2.0);
            color += u_spec_color * spec;
            v_color = clamp(color, 0.0, 1.0);
            float alpha = clamp(u_alpha_base + abs(h) * u_alpha_scale, 0.0, u_alpha_max);
            v_alpha = alpha;
            gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos.x, y, a_pos.y, 1.0);
        }
        """

        fragment_src = """
        #version 120
        varying vec3 v_normal;
        varying vec3 v_color;
        varying float v_alpha;
        void main() {
            gl_FragColor = vec4(v_color, v_alpha);
        }
        """

        try:
            self._program = shaders.compileProgram(
                shaders.compileShader(vertex_src, GL_VERTEX_SHADER),
                shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER),
            )
        except Exception:
            self._use_gpu = False
            return

        self._attrib_pos = glGetAttribLocation(self._program, "a_pos")
        self._attrib_uv = glGetAttribLocation(self._program, "a_uv")
        if self._attrib_pos < 0 or self._attrib_uv < 0:
            self._use_gpu = False
            return
        self._uni_height = glGetUniformLocation(self._program, "u_height")
        self._uni_height_scale = glGetUniformLocation(self._program, "u_height_scale")
        self._uni_texel = glGetUniformLocation(self._program, "u_texel")
        self._uni_normal_y = glGetUniformLocation(self._program, "u_normal_y")
        self._uni_base_color = glGetUniformLocation(self._program, "u_base_color")
        self._uni_peak_color = glGetUniformLocation(self._program, "u_peak_color")
        self._uni_spec_color = glGetUniformLocation(self._program, "u_spec_color")
        self._uni_alpha_base = glGetUniformLocation(self._program, "u_alpha_base")
        self._uni_alpha_scale = glGetUniformLocation(self._program, "u_alpha_scale")
        self._uni_alpha_max = glGetUniformLocation(self._program, "u_alpha_max")

        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_data.nbytes, self._vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes, self._indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self._height_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._height_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R32F,
            self.cols,
            self.rows,
            0,
            GL_RED,
            GL_FLOAT,
            self.buffer1,
        )
        err = glGetError()
        if err != GL_NO_ERROR:
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_LUMINANCE,
                self.cols,
                self.rows,
                0,
                GL_LUMINANCE,
                GL_FLOAT,
                self.buffer1,
            )
            err = glGetError()
            if err != GL_NO_ERROR:
                glBindTexture(GL_TEXTURE_2D, 0)
                self._use_gpu = False
                return
            self._height_format = GL_LUMINANCE
        glBindTexture(GL_TEXTURE_2D, 0)

        self._gpu_ready = True

    def _upload_height_texture(self):
        glBindTexture(GL_TEXTURE_2D, self._height_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.cols,
            self.rows,
            self._height_format,
            GL_FLOAT,
            self.buffer1,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self):
        """向量化的波動更新算法"""
        b1 = self.buffer1
        b2 = self.buffer2

        b2[1:-1, 1:-1] = (
            (b1[0:-2, 1:-1] + b1[2:, 1:-1] + b1[1:-1, 0:-2] + b1[1:-1, 2:]) * 0.5
            - b2[1:-1, 1:-1]
        ) * self.damping

        self.buffer1, self.buffer2 = self.buffer2, self.buffer1
        if self._gpu_ready:
            self._upload_height_texture()
        else:
            self._update_render_cache()

    def _update_render_cache(self):
        heights = self.buffer1
        self._vertices[..., 1] = heights * self.scale_h

        dx = heights[:-2, 1:-1] - heights[2:, 1:-1]
        dz = heights[1:-1, :-2] - heights[1:-1, 2:]
        length = np.sqrt(dx * dx + dz * dz + self._dy * self._dy)
        length = np.where(length == 0.0, 1.0, length)
        self._normals[1:-1, 1:-1, 0] = dx / length
        self._normals[1:-1, 1:-1, 1] = self._dy / length
        self._normals[1:-1, 1:-1, 2] = dz / length

        np.multiply(heights, self.scale_h, out=self._blend)
        self._blend += 0.5
        np.clip(self._blend, 0.0, 1.0, out=self._blend)
        np.subtract(1.0, self._blend, out=self._inv_blend)
        self._colors[..., 0:3] = (
            self._base_blue * self._inv_blend[..., None]
            + self._peak_color * self._blend[..., None]
        )

        np.clip(self._normals[..., 1], 0.0, 1.0, out=self._specular)
        self._specular **= 2
        self._colors[..., 0:3] += self._specular_color * self._specular[..., None]
        np.clip(self._colors[..., 0:3], 0.0, 1.0, out=self._colors[..., 0:3])

        np.absolute(heights, out=self._abs_heights)
        np.multiply(self._abs_heights, self._alpha_scale, out=self._abs_heights)
        self._abs_heights += self._alpha_base
        np.clip(self._abs_heights, 0.0, self._alpha_max, out=self._colors[..., 3])

    def _get_kernel(self, radius):
        kernel = self._kernel_cache.get(radius)
        if kernel is not None:
            return kernel
        axis = np.linspace(-radius, radius, radius * 2 + 1, dtype=np.float32)
        dx, dy = np.meshgrid(axis, axis, indexing="ij")
        dist = np.sqrt(dx * dx + dy * dy)
        mask = dist <= radius
        falloff = np.zeros_like(dist, dtype=np.float32)
        falloff[mask] = np.cos((dist[mask] / radius) * (math.pi / 2.0))
        if falloff.max() > 0:
            falloff /= falloff.max()
        self._kernel_cache[radius] = falloff
        return falloff

    def drop_rain_at_position(self, x, y, strength=3.0, radius=None):
        """在指定位置產生雨滴"""
        if radius is None:
            radius = self._default_kernel_radius
        kernel = self._get_kernel(radius)
        x0 = max(1, x - radius)
        x1 = min(self.cols - 1, x + radius + 1)
        y0 = max(1, y - radius)
        y1 = min(self.rows - 1, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kx0 = x0 - (x - radius)
        ky0 = y0 - (y - radius)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)
        self.buffer1[x0:x1, y0:y1] -= strength * kernel[kx0:kx1, ky0:ky1]

    def drop_rain_random(self):
        """隨機位置產生雨滴"""
        x = random.randint(4, self.cols - 5)
        y = random.randint(4, self.rows - 5)
        strength = random.uniform(0.8, 1.6)
        self.drop_rain_at_position(x, y, strength)

    def drop_rain_for_note(self, note, velocity=64, activity=0.5):
        """根據音符產生雨滴"""
        relative_note = note - 21
        x_ratio = min(max(relative_note / 87.0, 0.0), 1.0)
        x = int(x_ratio * (self.cols - 8)) + 4
        vel_ratio = min(max(velocity / 127.0, 0.0), 1.0)
        activity = min(max(activity, 0.0), 1.0)
        y_center = int(self.rows * 0.5)
        y_spread = int(self.rows * (0.06 + 0.05 * vel_ratio))
        y = y_center + random.randint(-y_spread, y_spread)
        strength = 0.7 + (vel_ratio ** 1.3) * 1.9
        strength *= 0.6 + 0.5 * activity
        radius = self._default_kernel_radius + int(activity * 2.0)
        self.drop_rain_at_position(x, y, strength, radius=radius)

    def drop_rain_world(self, world_x, world_z, strength=1.0, radius=None):
        origin = -self.size / 2.0
        gx = int((world_x - origin) / self.spacing)
        gz = int((world_z - origin) / self.spacing)
        if 1 <= gx < self.cols - 1 and 1 <= gz < self.rows - 1:
            self.drop_rain_at_position(gx, gz, strength, radius=radius)

    def drop_light_rain(self):
        x = random.randint(4, self.cols - 5)
        y = random.randint(4, self.rows - 5)
        strength = random.uniform(0.3, 0.8)
        self.drop_rain_at_position(x, y, strength, radius=2)

    def _draw_cpu(self):
        """繪製水面 - 半透明與波光反射效果"""
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, 128)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.1, 0.15, 0.3, 0.6])

        glPushMatrix()
        glTranslatef(-self.size/2, -1.8, -self.size/2)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._vertices_flat)
        glNormalPointer(GL_FLOAT, 0, self._normals_flat)
        glColorPointer(4, GL_FLOAT, 0, self._colors_flat)
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, self._indices)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glPopMatrix()
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glColor4f(1, 1, 1, 1)

    def _draw_gpu(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glUseProgram(self._program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._height_tex)
        glUniform1i(self._uni_height, 0)
        glUniform1f(self._uni_height_scale, self.scale_h)
        glUniform2f(self._uni_texel, float(self._texel[0]), float(self._texel[1]))
        glUniform1f(self._uni_normal_y, self._dy)
        glUniform3f(self._uni_base_color, *self._base_blue)
        glUniform3f(self._uni_peak_color, *self._peak_color)
        glUniform3f(self._uni_spec_color, *self._specular_color)
        glUniform1f(self._uni_alpha_base, self._alpha_base)
        glUniform1f(self._uni_alpha_scale, self._alpha_scale)
        glUniform1f(self._uni_alpha_max, self._alpha_max)

        glPushMatrix()
        glTranslatef(-self.size/2, -1.8, -self.size/2)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glEnableVertexAttribArray(self._attrib_pos)
        glEnableVertexAttribArray(self._attrib_uv)
        glVertexAttribPointer(
            self._attrib_pos,
            2,
            GL_FLOAT,
            GL_FALSE,
            self._vertex_stride,
            ctypes.c_void_p(0),
        )
        glVertexAttribPointer(
            self._attrib_uv,
            2,
            GL_FLOAT,
            GL_FALSE,
            self._vertex_stride,
            ctypes.c_void_p(8),
        )
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)
        glDisableVertexAttribArray(self._attrib_uv)
        glDisableVertexAttribArray(self._attrib_pos)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glPopMatrix()

        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)

    def draw(self):
        if self._gpu_ready:
            self._draw_gpu()
        else:
            self._draw_cpu()

class PianoStrings:
    def __init__(self, keys):
        self._keys = keys
        self._intensity = np.zeros(88, dtype=np.float32)
        self._target = np.zeros(88, dtype=np.float32)
        self._attack = 18.0
        self._decay = 10.0
        self._y = (keys.white_key_y_offset + 0.32) * PIANO_SCALE
        self._z_start = (keys.white_key_z_offset - 0.25) * PIANO_SCALE
        self._z_end = (keys.white_key_z_offset - 1.7) * PIANO_SCALE
        self._x_min_model = None
        self._x_max_model = None
        self._key_min_model = None
        self._key_max_model = None
        self._y_mid_model = None
        self._z_min_model = None
        self._z_max_model = None

    def _idx(self, note):
        idx = note - 21
        if 0 <= idx < 88:
            return idx
        return None

    def configure_from_bounds(self, string_bounds=None, key_bounds=None):
        if string_bounds:
            bounds_min, bounds_max = string_bounds
            self._x_min_model = float(bounds_min[0])
            self._x_max_model = float(bounds_max[0])
            self._y_mid_model = float((bounds_min[1] + bounds_max[1]) * 0.5)
            self._z_min_model = float(bounds_min[2])
            self._z_max_model = float(bounds_max[2])
        if key_bounds:
            key_min, key_max = key_bounds
            self._key_min_model = float(key_min[0])
            self._key_max_model = float(key_max[0])

    def _note_x_model(self, note):
        idx = self._idx(note)
        if idx is None:
            return None
        if self._key_min_model is not None and self._key_max_model is not None:
            span = self._key_max_model - self._key_min_model
            if span != 0:
                return self._key_min_model + span * (idx / 87.0)
        if self._x_min_model is not None and self._x_max_model is not None:
            span = self._x_max_model - self._x_min_model
            if span != 0:
                return self._x_min_model + span * (idx / 87.0)
        x_pos = self._keys.note_world_x(note)
        if x_pos is None:
            return None
        return x_pos / max(PIANO_MODEL_SCALE, 1e-6)

    def press(self, note, velocity):
        idx = self._idx(note)
        if idx is None:
            return
        vel_ratio = _clamp(velocity / 127.0, 0.0, 1.0)
        self._target[idx] = max(self._target[idx], vel_ratio)

    def release(self, note):
        idx = self._idx(note)
        if idx is None:
            return
        self._target[idx] = 0.0

    def update(self, dt):
        if dt <= 0:
            return
        rise = 1.0 - np.exp(-self._attack * dt)
        fall = 1.0 - np.exp(-self._decay * dt)
        rising = self._intensity < self._target
        self._intensity[rising] += (self._target[rising] - self._intensity[rising]) * rise
        falling = ~rising
        self._intensity[falling] += (self._target[falling] - self._intensity[falling]) * fall

    def get_light_state(self):
        idx = int(np.argmax(self._intensity))
        peak = float(self._intensity[idx])
        if peak <= 0.02:
            return None
        note = idx + 21
        x_pos = self._note_x_model(note)
        if x_pos is None:
            return None
        # Transform from model space to world space (match display transforms)
        x = (x_pos * PIANO_MODEL_SCALE + PIANO_OFFSET_X) * PIANO_SCALE
        if self._y_mid_model is not None:
            y = (self._y_mid_model * PIANO_MODEL_SCALE + 2.2) * PIANO_SCALE
        else:
            y = self._y
        if self._z_min_model is not None and self._z_max_model is not None:
            z_mid = (self._z_min_model + self._z_max_model) * 0.5
            z = (z_mid * PIANO_MODEL_SCALE + PIANO_OFFSET_Z) * PIANO_SCALE
        else:
            z = (self._z_start + self._z_end) * 0.5
        intensity = 0.3 + peak * STRING_GLOW_INTENSITY
        return (x, y, z, STRING_GLOW_COLOR, intensity)

    def draw(self):
        if string_display_list:
            active = [
                (idx, float(val))
                for idx, val in enumerate(self._intensity)
                if val > 0.03
            ]
            if not active:
                return
            active.sort(key=lambda item: item[1], reverse=True)
            active = active[:12]

            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glEnable(GL_CLIP_PLANE0)
            glEnable(GL_CLIP_PLANE1)
            glEnable(GL_CLIP_PLANE2)
            glEnable(GL_CLIP_PLANE3)

            glPushMatrix()
            glScalef(PIANO_SCALE, PIANO_SCALE, PIANO_SCALE)
            glTranslatef(PIANO_OFFSET_X, 0.0, PIANO_OFFSET_Z)
            glTranslatef(0.0, 2.2, 0.0)
            glScalef(PIANO_MODEL_SCALE, PIANO_MODEL_SCALE, PIANO_MODEL_SCALE)

            if self._x_min_model is not None and self._x_max_model is not None:
                base_width = (self._x_max_model - self._x_min_model) / 87.0
            else:
                base_width = self._keys.white_key_spacing
            clip_width = base_width * STRING_GLOW_THICKNESS
            z_front_min = None
            z_front_max = None
            z_back_min = None
            z_back_max = None
            if self._z_min_model is not None and self._z_max_model is not None:
                z_span = self._z_max_model - self._z_min_model
                z_front_max = self._z_max_model
                z_front_min = z_front_max - z_span * STRING_FRONT_RATIO
                z_back_min = self._z_min_model
                z_back_max = z_back_min + z_span * STRING_BACK_RATIO
            for idx, intensity in active:
                note = idx + 21
                x_pos = self._note_x_model(note)
                if x_pos is None:
                    continue
                x = x_pos
                x_min = x - clip_width * 0.5
                x_max = x + clip_width * 0.5
                glClipPlane(GL_CLIP_PLANE0, (1.0, 0.0, 0.0, -x_min))
                glClipPlane(GL_CLIP_PLANE1, (-1.0, 0.0, 0.0, x_max))
                r = STRING_BASE_COLOR[0] + STRING_GLOW_COLOR[0] * intensity
                g = STRING_BASE_COLOR[1] + STRING_GLOW_COLOR[1] * intensity
                b = STRING_BASE_COLOR[2] + STRING_GLOW_COLOR[2] * intensity
                glColor4f(r, g, b, 0.55 + 0.35 * intensity)
                if z_front_min is not None and z_front_max is not None:
                    glClipPlane(GL_CLIP_PLANE2, (0.0, 0.0, 1.0, -z_front_min))
                    glClipPlane(GL_CLIP_PLANE3, (0.0, 0.0, -1.0, z_front_max))
                    glCallList(string_display_list)
                if z_back_min is not None and z_back_max is not None:
                    glClipPlane(GL_CLIP_PLANE2, (0.0, 0.0, 1.0, -z_back_min))
                    glClipPlane(GL_CLIP_PLANE3, (0.0, 0.0, -1.0, z_back_max))
                    glCallList(string_display_list)
                if z_front_min is None and z_back_min is None:
                    glCallList(string_display_list)

            glPopMatrix()
            glDisable(GL_CLIP_PLANE0)
            glDisable(GL_CLIP_PLANE1)
            glDisable(GL_CLIP_PLANE2)
            glDisable(GL_CLIP_PLANE3)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glLineWidth(STRING_LINE_WIDTH)
            glBegin(GL_LINES)
            for idx, intensity in enumerate(self._intensity):
                if intensity <= 0.01:
                    continue
                note = idx + 21
                x_pos = self._note_x_model(note)
                if x_pos is None:
                    continue
                x = x_pos * PIANO_SCALE
                r = STRING_BASE_COLOR[0] + STRING_GLOW_COLOR[0] * intensity
                g = STRING_BASE_COLOR[1] + STRING_GLOW_COLOR[1] * intensity
                b = STRING_BASE_COLOR[2] + STRING_GLOW_COLOR[2] * intensity
                glColor4f(r, g, b, 0.55 + 0.4 * intensity)
                glVertex3f(x, self._y, self._z_start)
                glVertex3f(x, self._y, self._z_end)
            glEnd()
            glLineWidth(1.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)


class LightRain:
    def __init__(self, water, count=200, area=40.0, top=60.0, bottom=-2.0):
        self._water = water
        self._count = count
        self._area = area
        self._top = top
        self._bottom = bottom
        self._drops = []
        self._last_time = time.perf_counter()
        for _ in range(count):
            self._drops.append(self._new_drop(random_y=True))

    def _new_drop(self, random_y=False):
        x = random.uniform(-self._area * 0.5, self._area * 0.5)
        z = random.uniform(-self._area * 0.5, self._area * 0.5)
        y = random.uniform(self._bottom, self._top) if random_y else self._top
        speed = random.uniform(12.0, 18.0)
        length = random.uniform(0.5, 1.1)
        return [x, y, z, speed, length]

    def update(self, now):
        dt = now - self._last_time
        if dt <= 0:
            return
        self._last_time = now
        for idx, drop in enumerate(self._drops):
            drop[1] -= drop[3] * dt
            if drop[1] <= self._bottom:
                self._water.drop_rain_world(drop[0], drop[2], strength=0.45, radius=2)
                self._drops[idx] = self._new_drop()

    def draw(self):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.7, 0.85, 1.0, 0.45)
        glBegin(GL_LINES)
        for x, y, z, _, length in self._drops:
            glVertex3f(x, y, z)
            glVertex3f(x, y + length, z)
        glEnd()
        glEnable(GL_LIGHTING)


water_sim = WaterRipple()
piano_strings = PianoStrings(piano_keys)
light_rain = LightRain(water_sim)
_recent_note_events = collections.deque()
_recent_velocity_sum = 0.0


def _update_note_activity(now, velocity):
    global _recent_velocity_sum
    _recent_note_events.append((now, velocity))
    _recent_velocity_sum += velocity
    while _recent_note_events and now - _recent_note_events[0][0] > NOTE_ACTIVITY_WINDOW:
        _, old_velocity = _recent_note_events.popleft()
        _recent_velocity_sum -= old_velocity
    count = len(_recent_note_events)
    if count:
        avg_vel = (_recent_velocity_sum / count) / 127.0
    else:
        avg_vel = 0.0
    density = count / NOTE_ACTIVITY_WINDOW
    density_norm = min(density / NOTE_ACTIVITY_REF_DENSITY, 1.0)
    activity = 0.2 + 0.8 * (0.6 * avg_vel + 0.4 * density_norm)
    return min(max(activity, 0.0), 1.0)

# ============================================================
# MIDI事件處理
# ============================================================
def handle_midi_event(event):
    """處理MIDI事件
    event = {
        "type": "on" | "off",
        "note": 60,
        "velocity": 96,
        "channel": 0,
        "time_ms": 1234.0,
        "playback_time_ms": 1238.4
    }
    """
    event_type = event.get("type")
    note = event.get("note")
    velocity = event.get("velocity", 64)
    
    if event_type == "on":
        now = time.perf_counter()
        activity = _update_note_activity(now, velocity)
        success = piano_keys.press_key(note, velocity)
        if success:
            piano_strings.press(note, velocity)
            water_sim.drop_rain_for_note(note, velocity=velocity, activity=activity)
        if DEBUG:
            print(f"按下琴鍵 Note={note}, Velocity={velocity}")
    
    elif event_type == "off":
        now = time.perf_counter()
        success = piano_keys.release_key(note)
        if success:
            piano_strings.release(note)
        if DEBUG and success:
            print(f"釋放琴鍵 Note={note}")

def process_midi_events():
    """處理隊列中的所有MIDI事件"""
    while not midi_event_queue.empty():
        try:
            event = midi_event_queue.get_nowait()
            handle_midi_event(event)
        except queue.Empty:
            break

# 測試用的MIDI事件生成器
def test_midi_sequence():
    """測試用：生成一段簡單的旋律"""
    # C大調音階
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 到 C5
    
    def play_sequence():
        for i, note in enumerate(notes):
            import time
            time.sleep(0.3)
            midi_event_queue.put({
                "type": "on",
                "note": note,
                "velocity": 80,
                "channel": 0,
                "time_ms": i * 300
            })
            time.sleep(0.2)
            midi_event_queue.put({
                "type": "off",
                "note": note,
                "velocity": 0,
                "channel": 0,
                "time_ms": i * 300 + 200
            })
    
    # 在新線程中播放
    thread = threading.Thread(target=play_sequence, daemon=True)
    thread.start()

# ============================================================
# 材質與模型加載
# ============================================================

def apply_material(gltf, material_index):
    if material_index is None or material_index < 0: return
    material = gltf.materials[material_index]
    pbr = material.pbrMetallicRoughness
    if pbr and pbr.baseColorFactor:
        r, g, b, a = pbr.baseColorFactor
    else:
        r, g, b, a = (1.0, 1.0, 1.0, 1.0)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 32)
    glDisable(GL_BLEND)
    glDepthMask(GL_TRUE)

def load_glb_model(file_path):
    print("Loading GLB file...")
    try:
        gltf = pygltflib.GLTF2().load(file_path)
        return gltf
    except:
        return None

def get_buffer_data(gltf, accessor_idx):
    if accessor_idx is None: return np.array([])
    accessor = gltf.accessors[accessor_idx]
    if accessor.bufferView is None: return np.array([])
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    byte_offset = (accessor.byteOffset or 0) + (view.byteOffset or 0)
    byte_length = view.byteLength
    if accessor.componentType == 5126: dtype = np.float32
    elif accessor.componentType == 5125: dtype = np.uint32
    elif accessor.componentType == 5123: dtype = np.uint16
    else: dtype = np.float32
    type_dim = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}.get(accessor.type,1)
    count = accessor.count
    data = blob[byte_offset:byte_offset+byte_length]
    arr = np.frombuffer(data, dtype=dtype)
    if len(arr) < count * type_dim: return np.array([])
    arr = arr[:count * type_dim]
    return arr.reshape(count, type_dim)

def draw_primitive_geometry(gltf, primitive, use_material=True):
    if use_material:
        apply_material(gltf, primitive.material)
    positions = get_buffer_data(gltf, primitive.attributes.POSITION)
    if positions.size == 0: return
    normals = None
    if primitive.attributes.NORMAL is not None:
        normals = get_buffer_data(gltf, primitive.attributes.NORMAL)
        if normals.size == 0: normals = None
    if primitive.indices is not None:
        indices = get_buffer_data(gltf, primitive.indices).flatten()
    else:
        indices = range(len(positions))
    glBegin(GL_TRIANGLES)
    for i in indices:
        if normals is not None and i < len(normals):
            glNormal3fv(normals[i])
        glVertex3fv(positions[i])
    glEnd()

def draw_node(gltf, node_index, use_material=True):
    node = gltf.nodes[node_index]
    glPushMatrix()
    if node.matrix:
        glMultMatrixf(np.array(node.matrix, dtype=np.float32))
    else:
        if node.translation: glTranslatef(*node.translation)
        if node.scale: glScalef(*node.scale)
    if node.mesh is not None:
        mesh = gltf.meshes[node.mesh]
        for p in mesh.primitives:
            draw_primitive_geometry(gltf, p, use_material=use_material)
    if node.children:
        for c in node.children:
            draw_node(gltf, c, use_material=use_material)
    glPopMatrix()

def create_display_list(gltf):
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    scene = gltf.scenes[gltf.scene]
    for node_index in scene.nodes:
        draw_node(gltf, node_index)
    glEndList()
    return list_id


def _collect_nodes(gltf, indices):
    collected = []
    stack = list(indices)
    seen = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        collected.append(idx)
        node = gltf.nodes[idx]
        if node.children:
            stack.extend(node.children)
    return collected


def find_nodes_by_name(gltf, keyword):
    result = []
    key = keyword.lower()
    for idx, node in enumerate(gltf.nodes):
        name = getattr(node, "name", "") or ""
        if key in name.lower():
            result.append(idx)
    return result


def create_display_list_for_nodes(gltf, node_indices, use_material=True):
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    for node_index in node_indices:
        draw_node(gltf, node_index, use_material=use_material)
    glEndList()
    return list_id


def _node_local_matrix(node):
    if node.matrix:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4)
    mat = np.eye(4, dtype=np.float32)
    if node.translation:
        t = np.array(node.translation, dtype=np.float32)
        mat[:3, 3] = t
    if node.rotation:
        x, y, z, w = node.rotation
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rot = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0.0],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0.0],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        mat = mat @ rot
    if node.scale:
        s = np.array(node.scale, dtype=np.float32)
        mat = mat @ np.diag([s[0], s[1], s[2], 1.0])
    return mat


def _parent_map(gltf):
    parents = {idx: None for idx in range(len(gltf.nodes))}
    for idx, node in enumerate(gltf.nodes):
        if node.children:
            for child in node.children:
                parents[child] = idx
    return parents


def _world_matrix(gltf, idx, parents):
    chain = []
    cur = idx
    while cur is not None:
        chain.append(_node_local_matrix(gltf.nodes[cur]))
        cur = parents[cur]
    mat = np.eye(4, dtype=np.float32)
    for m in reversed(chain):
        mat = mat @ m
    return mat


def compute_bounds_for_nodes(gltf, node_indices):
    parents = _parent_map(gltf)
    mins = []
    maxs = []
    for idx in node_indices:
        node = gltf.nodes[idx]
        if node.mesh is None:
            continue
        world = _world_matrix(gltf, idx, parents)
        mesh = gltf.meshes[node.mesh]
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue
            pos = get_buffer_data(gltf, prim.attributes.POSITION)
            if pos.size == 0:
                continue
            ones = np.ones((pos.shape[0], 1), dtype=np.float32)
            v = np.hstack([pos, ones])
            v = (world @ v.T).T
            mins.append(v[:, :3].min(axis=0))
            maxs.append(v[:, :3].max(axis=0))
    if not mins:
        return None
    return np.min(mins, axis=0), np.max(maxs, axis=0)


class Skybox:
    def __init__(self, folder):
        self._folder = Path(folder)
        self._textures = []
        self._available = False

    def load(self):
        try:
            import pygame
        except Exception:
            return False
        if not pygame.get_init():
            pygame.init()
        if hasattr(pygame.image, "init"):
            pygame.image.init()

        # Face order: +X, -X, +Y (top), -Y, +Z, -Z
        faces = [
            ("skybox4", 0),   # +X
            ("skybox5", 0),   # -X
            ("skybox3", 0),   # +Y (top)
            ("skybox2", 0),   # -Y (bottom)
            ("skybox1", 0),   # +Z (front)
            ("skybox0", 0),   # -Z (back)
        ]
        textures = []
        for base, rot in faces:
            path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = self._folder / f"{base}{ext}"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                return False
            try:
                surface = pygame.image.load(str(path))
            except Exception:
                return False
            if rot:
                surface = pygame.transform.rotate(surface, rot)
            surface = pygame.transform.flip(surface, False, True)
            width, height = surface.get_size()
            if width != height:
                size = min(width, height)
                x0 = (width - size) // 2
                y0 = (height - size) // 2
                surface = surface.subsurface((x0, y0, size, size)).copy()
                width = height = size
            image_data = pygame.image.tostring(surface, "RGB", True)
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            if (
                "GL_TEXTURE_MAX_ANISOTROPY_EXT" in globals()
                and "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT" in globals()
            ):
                try:
                    max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
                except Exception:
                    pass
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                width,
                height,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                image_data,
            )
            textures.append(tex_id)
        self._textures = textures
        self._available = True
        return True

    def draw(self, size=60.0):
        if not self._available:
            return False
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glColor3f(1.0, 1.0, 1.0)

        # Face order: +X, -X, +Y, -Y, +Z, -Z
        faces = [
            ((size, -size, -size), (size, -size, size), (size, size, size), (size, size, -size)),
            ((-size, -size, size), (-size, -size, -size), (-size, size, -size), (-size, size, size)),
            ((-size, size, -size), (size, size, -size), (size, size, size), (-size, size, size)),
            ((-size, -size, size), (size, -size, size), (size, -size, -size), (-size, -size, -size)),
            ((-size, -size, size), (-size, size, size), (size, size, size), (size, -size, size)),
            ((size, -size, -size), (size, size, -size), (-size, size, -size), (-size, -size, -size)),
        ]

        for tex_id, verts in zip(self._textures, faces):
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(*verts[0])
            glTexCoord2f(1.0, 0.0)
            glVertex3f(*verts[1])
            glTexCoord2f(1.0, 1.0)
            glVertex3f(*verts[2])
            glTexCoord2f(0.0, 1.0)
            glVertex3f(*verts[3])
            glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        return True


def draw_sky():
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    if skybox and skybox.draw():
        glPopMatrix()
        glColor3f(1.0, 1.0, 1.0)
        return

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    size = 80.0
    top = (0.03, 0.05, 0.12)
    horizon = (0.09, 0.11, 0.18)
    bottom = (0.02, 0.03, 0.06)

    glBegin(GL_QUADS)
    # +Y (top)
    glColor3f(*top)
    glVertex3f(-size, size, -size)
    glVertex3f(size, size, -size)
    glVertex3f(size, size, size)
    glVertex3f(-size, size, size)

    # -Y (bottom)
    glColor3f(*bottom)
    glVertex3f(-size, -size, -size)
    glVertex3f(-size, -size, size)
    glVertex3f(size, -size, size)
    glVertex3f(size, -size, -size)

    # +Z (front)
    glColor3f(*horizon)
    glVertex3f(-size, -size, size)
    glColor3f(*top)
    glVertex3f(-size, size, size)
    glVertex3f(size, size, size)
    glColor3f(*horizon)
    glVertex3f(size, -size, size)

    # -Z (back)
    glColor3f(*horizon)
    glVertex3f(-size, -size, -size)
    glColor3f(*top)
    glVertex3f(size, size, -size)
    glVertex3f(-size, size, -size)
    glColor3f(*horizon)
    glVertex3f(size, -size, -size)

    # +X (right)
    glColor3f(*horizon)
    glVertex3f(size, -size, -size)
    glColor3f(*top)
    glVertex3f(size, size, -size)
    glVertex3f(size, size, size)
    glColor3f(*horizon)
    glVertex3f(size, -size, size)

    # -X (left)
    glColor3f(*horizon)
    glVertex3f(-size, -size, -size)
    glColor3f(*top)
    glVertex3f(-size, size, size)
    glVertex3f(-size, size, -size)
    glColor3f(*horizon)
    glVertex3f(-size, -size, size)
    glEnd()

    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glColor3f(1.0, 1.0, 1.0)


def _ui_button_rects():
    x = UI_PADDING
    y = UI_PADDING
    size = UI_BUTTON_SIZE
    play_rect = (x, y, size, size)
    next_rect = (x + size + UI_BUTTON_GAP, y, size, size)
    return play_rect, next_rect


def _ui_hit_test(x, y):
    play_rect, next_rect = _ui_button_rects()
    for name, rect in (("play_pause", play_rect), ("next", next_rect)):
        rx, ry, rw, rh = rect
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return name
    return None


def draw_ui():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, window_width, window_height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_CULL_FACE)

    play_rect, next_rect = _ui_button_rects()
    panel_w = UI_BUTTON_SIZE * 2 + UI_BUTTON_GAP + 220
    panel_h = UI_BUTTON_SIZE + 16
    glColor4f(0.0, 0.0, 0.0, 0.75)
    glBegin(GL_QUADS)
    glVertex2f(UI_PADDING - 8, UI_PADDING - 8)
    glVertex2f(UI_PADDING - 8 + panel_w, UI_PADDING - 8)
    glVertex2f(UI_PADDING - 8 + panel_w, UI_PADDING - 8 + panel_h)
    glVertex2f(UI_PADDING - 8, UI_PADDING - 8 + panel_h)
    glEnd()

    # Play/Pause button with outline
    px, py, ps, _ = play_rect
    glColor4f(0.18, 0.18, 0.22, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(px, py)
    glVertex2f(px + ps, py)
    glVertex2f(px + ps, py + ps)
    glVertex2f(px, py + ps)
    glEnd()
    glColor4f(0.95, 0.95, 1.0, 0.95)
    if ui_state["playing"]:
        bar_w = ps * 0.22
        gap = ps * 0.12
        glBegin(GL_QUADS)
        glVertex2f(px + ps * 0.28, py + ps * 0.2)
        glVertex2f(px + ps * 0.28 + bar_w, py + ps * 0.2)
        glVertex2f(px + ps * 0.28 + bar_w, py + ps * 0.8)
        glVertex2f(px + ps * 0.28, py + ps * 0.8)
        glVertex2f(px + ps * 0.28 + bar_w + gap, py + ps * 0.2)
        glVertex2f(px + ps * 0.28 + bar_w + gap + bar_w, py + ps * 0.2)
        glVertex2f(px + ps * 0.28 + bar_w + gap + bar_w, py + ps * 0.8)
        glVertex2f(px + ps * 0.28 + bar_w + gap, py + ps * 0.8)
        glEnd()
    else:
        glBegin(GL_TRIANGLES)
        glVertex2f(px + ps * 0.28, py + ps * 0.2)
        glVertex2f(px + ps * 0.28, py + ps * 0.8)
        glVertex2f(px + ps * 0.78, py + ps * 0.5)
        glEnd()

    # Next button with outline
    nx, ny, ns, _ = next_rect
    glColor4f(0.18, 0.18, 0.22, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(nx, ny)
    glVertex2f(nx + ns, ny)
    glVertex2f(nx + ns, ny + ns)
    glVertex2f(nx, ny + ns)
    glEnd()
    glColor4f(0.95, 0.95, 1.0, 0.95)
    glBegin(GL_TRIANGLES)
    glVertex2f(nx + ns * 0.2, ny + ns * 0.2)
    glVertex2f(nx + ns * 0.2, ny + ns * 0.8)
    glVertex2f(nx + ns * 0.68, ny + ns * 0.5)
    glEnd()
    glBegin(GL_QUADS)
    glVertex2f(nx + ns * 0.72, ny + ns * 0.2)
    glVertex2f(nx + ns * 0.86, ny + ns * 0.2)
    glVertex2f(nx + ns * 0.86, ny + ns * 0.8)
    glVertex2f(nx + ns * 0.72, ny + ns * 0.8)
    glEnd()

    # Track label
    glColor4f(0.9, 0.9, 0.95, 0.95)
    label = ui_state.get("track") or "No track"
    text_x = next_rect[0] + next_rect[2] + 14
    text_y = UI_PADDING + UI_BUTTON_SIZE * 0.72
    glRasterPos2f(text_x, text_y)
    for ch in label[:48]:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# ============================================================
# OpenGL setup
# ============================================================

def init():
    global piano_display_list, model, skybox, string_display_list, PIANO_OFFSET_X, PIANO_OFFSET_Z
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_MULTISAMPLE)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    
    # 主光源（模擬陽光）
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [10, 20, 10, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.05, 0.05, 0.06, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.04, 0.04, 0.04, 1])

    # 側面補光
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-10, 8, -5, 1])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.04, 0.04, 0.05, 1])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.03, 0.03, 0.04, 1])

    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_POSITION, [0, 8, 0, 1])
    glLightfv(GL_LIGHT2, GL_DIFFUSE, [0, 0, 0, 1])
    glLightfv(GL_LIGHT2, GL_SPECULAR, [0, 0, 0, 1])

    # 環境光
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.03, 0.03, 0.04, 1])
    
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    glEnable(GL_NORMALIZE)
    
    # 深藍色背景
    glClearColor(0.03, 0.05, 0.12, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

    if model:
        piano_display_list = create_display_list(model)
        key_bounds = None
        key_nodes = find_nodes_by_name(model, "keys")
        if key_nodes:
            key_nodes = _collect_nodes(model, key_nodes)
            key_bounds = compute_bounds_for_nodes(model, key_nodes)
        string_nodes = find_nodes_by_name(model, "string")
        string_bounds = None
        if string_nodes:
            string_nodes = _collect_nodes(model, string_nodes)
            string_display_list = create_display_list_for_nodes(
                model, string_nodes, use_material=False
            )
            string_bounds = compute_bounds_for_nodes(model, string_nodes)
        else:
            string_display_list = None
        piano_strings.configure_from_bounds(string_bounds, key_bounds)
        model_bounds = compute_bounds_for_nodes(model, list(range(len(model.nodes))))
        if model_bounds:
            min_b, max_b = model_bounds
            center_x = float((min_b[0] + max_b[0]) * 0.5)
            center_z = float((min_b[2] + max_b[2]) * 0.5)
        else:
            center_x = piano_keys.center_x
            center_z = piano_keys.center_z
        PIANO_OFFSET_X = -center_x * PIANO_MODEL_SCALE
        PIANO_OFFSET_Z = -center_z * PIANO_MODEL_SCALE
        if key_bounds:
            model_key_min_x = float(key_bounds[0][0]) * PIANO_MODEL_SCALE
            model_key_max_x = float(key_bounds[1][0]) * PIANO_MODEL_SCALE
            local_key_min_x, local_key_max_x = piano_keys.key_span_x()
            local_span = local_key_max_x - local_key_min_x
            model_span = model_key_max_x - model_key_min_x
            if local_span > 0 and model_span > 0:
                local_center = (local_key_min_x + local_key_max_x) * 0.5
                model_center = (model_key_min_x + model_key_max_x) * 0.5
                offset_x = model_center - local_center
                piano_keys.set_alignment(1.0, offset_x)
    skybox_path = Path(__file__).with_name("skybox")
    if skybox_path.exists():
        skybox = Skybox(skybox_path)
        if not skybox.load():
            skybox = None
    water_sim.init_gl()
    
    print("\n" + "="*60)
    print("88鍵鋼琴演奏模擬器")
    print("="*60)
    print("\n控制說明:")
    print("W/S: 朝向鏡頭方向前進/後退")
    print("A/D: 左右移動")
    print("Q/E: 相機距離縮放（拉近/拉遠）")
    print("鼠標拖拽: 旋轉視角")
    print("T: 播放測試音階")
    print("R: 隨機雨滴")
    print("L: 切換細雨效果")
    print("ESC: 退出")
    print("\nMIDI事件格式:")
    print('{"type": "on/off", "note": 21-108, "velocity": 0-127}')
    print("="*60 + "\n")

# ============================================================
# Interaction Loop
# ============================================================

def keyboard(key, x, y):
    global cam_pan_x, cam_pan_y, cam_pan_z, cam_distance, rot_x, rot_y
    
    try:
        k = key.decode('utf-8').lower()
    except:
        k = key
    
    if k == '\x1b':
        sys.exit(0)
    
    # 朝向鏡頭方向移動
    if k == 'w' or k == 's':
        rot_y_rad = math.radians(rot_y)
        rot_x_rad = math.radians(rot_x)
        forward_x = math.sin(rot_y_rad) * math.cos(rot_x_rad)
        forward_y = -math.sin(rot_x_rad)
        forward_z = -math.cos(rot_y_rad) * math.cos(rot_x_rad)
        direction = 1.0 if k == 'w' else -1.0
        cam_pan_x += forward_x * pan_speed * direction
        cam_pan_y += forward_y * pan_speed * direction
        cam_pan_z += forward_z * pan_speed * direction
    
    if k == 'a' or k == 'd':
        rot_y_rad = math.radians(rot_y)
        right_x = math.cos(rot_y_rad)
        right_z = math.sin(rot_y_rad)
        direction = -1.0 if k == 'a' else 1.0
        cam_pan_x += right_x * pan_speed * direction
        cam_pan_z += right_z * pan_speed * direction
    
    # 相機縮放
    if k == 'q':
        cam_distance = max(5.0, cam_distance - zoom_speed)
    elif k == 'e':
        cam_distance = min(50.0, cam_distance + zoom_speed)
    
    # 測試音階
    if k == 't':
        test_midi_sequence()
        print("播放測試音階...")
    
    # 隨機雨滴
    if k == 'r':
        water_sim.drop_rain_random()

    # 切換細雨
    if k == 'l':
        global LIGHT_RAIN_ENABLED
        LIGHT_RAIN_ENABLED = not LIGHT_RAIN_ENABLED

    glutPostRedisplay()

def mouse(button, state, x, y):
    global mouse_down, last_x, last_y
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            hit = _ui_hit_test(x, y)
            if hit:
                callback = ui_callbacks.get(hit)
                if callback:
                    callback()
                return
        mouse_down = (state == GLUT_DOWN)
        last_x, last_y = x, y

def motion(x, y):
    global rot_x, rot_y, last_x, last_y
    if mouse_down:
        rot_y += (x - last_x) * 0.5
        rot_x += (y - last_y) * 0.5
        last_x, last_y = x, y
        glutPostRedisplay()

_last_frame_time = time.perf_counter()
_last_water_time = _last_frame_time
_last_string_time = _last_frame_time


def idle():
    # 處理MIDI事件
    process_midi_events()

    global _last_frame_time, _last_water_time
    now = time.perf_counter()
    if now - _last_water_time >= 1.0 / WATER_UPDATE_HZ:
        water_sim.update()
        _last_water_time = now

    if LIGHT_RAIN_ENABLED:
        light_rain.update(now)

    global _last_string_time
    dt = now - _last_string_time
    piano_strings.update(dt)
    _last_string_time = now

    if now - _last_frame_time >= 1.0 / TARGET_FPS:
        piano_keys.update_key_animation()
        glutPostRedisplay()
        _last_frame_time = now

def reshape(w, h):
    global window_width, window_height
    if h == 0: h = 1
    window_width = w
    window_height = h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w/h, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

def display():
    global rot_x, rot_y, piano_display_list, cam_distance
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 6, cam_distance, 0, 0, 0, 0, 1, 0)

    draw_sky()

    now = time.perf_counter()
    light_state = piano_strings.get_light_state()
    if light_state:
        lx, ly, lz, color, intensity = light_state
        glLightfv(GL_LIGHT2, GL_POSITION, [lx, ly, lz, 1])
        glLightfv(
            GL_LIGHT2,
            GL_DIFFUSE,
            [color[0] * intensity, color[1] * intensity, color[2] * intensity, 1],
        )
        glLightfv(
            GL_LIGHT2,
            GL_SPECULAR,
            [color[0] * intensity, color[1] * intensity, color[2] * intensity, 1],
        )
    else:
        glLightfv(GL_LIGHT2, GL_DIFFUSE, [0, 0, 0, 1])
        glLightfv(GL_LIGHT2, GL_SPECULAR, [0, 0, 0, 1])

    glPushMatrix()
    glTranslatef(-cam_pan_x, -cam_pan_y, -cam_pan_z)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)

    glPushMatrix()
    glScalef(PIANO_SCALE, PIANO_SCALE, PIANO_SCALE)
    glTranslatef(PIANO_OFFSET_X, 0.0, PIANO_OFFSET_Z)

    # 繪製鋼琴模型
    if piano_display_list:
        glPushMatrix()
        glTranslatef(0.0, 2.2, 0.0)
        glScalef(PIANO_MODEL_SCALE, PIANO_MODEL_SCALE, PIANO_MODEL_SCALE)
        glColor4f(1, 1, 1, 1)
        glCallList(piano_display_list)
        glPopMatrix()

    # 繪製琴鍵
    piano_keys.draw()
    glPopMatrix()

    piano_strings.draw()
    if LIGHT_RAIN_ENABLED:
        light_rain.draw()

    # 繪製水面（最後繪製以正確處理透明度）
    water_sim.draw()

    glPopMatrix()
    draw_ui()
    glutSwapBuffers()

# ============================================================
# 對外API - 供外部程序調用
# ============================================================

def send_midi_event(event):
    """供外部調用的MIDI事件接口
    event: dict with keys "type", "note", "velocity", etc.
    """
    midi_event_queue.put(event)


def run(model_path=None, window_title=b"Piano Visualization with Water", debug=False, on_ready=None):
    """啟動視覺化主循環（供外部調用）"""
    global model, DEBUG
    DEBUG = debug
    if model_path is None:
        model_path = Path(__file__).with_name("glass_piano.glb")
    model = load_glb_model(str(model_path))
    if model:
        print("成功載入鋼琴模型:", model_path)
    else:
        print("未找到鋼琴模型，僅顯示琴鍵和水面")

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutCreateWindow(window_title)

    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)

    init()
    if on_ready is not None:
        on_ready()
    glutMainLoop()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run()
