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

try:
    from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_8_BY_13
except Exception:
    GLUT_BITMAP_HELVETICA_18 = None
    GLUT_BITMAP_8_BY_13 = None

GL_TEXTURE_MAX_ANISOTROPY_EXT = globals().get("GL_TEXTURE_MAX_ANISOTROPY_EXT")
GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = globals().get("GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT")

DEBUG = False
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TARGET_FPS = 60.0
WATER_UPDATE_HZ = 60.0
WATER_GRID_COLS = 384
WATER_GRID_ROWS = 384
WATER_REFLECTION_STRENGTH = 0.6
WATER_HEIGHT_SCALE = 0.13
WATER_RIPPLE_STRENGTH_SCALE = 1.35
WATER_PLANE_Y = -1.8
WATER_SCENE_REFLECTION_STRENGTH = 0.45
WATER_REFLECTION_DISTORT = 0.02
REFLECTION_TEXTURE_SCALE = 0.5
REFLECTION_TEXTURE_MIN = 256
REFLECTION_TEXTURE_MAX = 512
REFLECTION_UPDATE_HZ = 20.0
WATER_ALPHA_BASE = 0.96
WATER_ALPHA_SCALE = 0.03
WATER_ALPHA_MAX = 1.0
LIGHT_RAIN_ENABLED = True
LIGHT_RAIN_COUNT = 1200
NOTE_ACTIVITY_WINDOW = 0.22
NOTE_ACTIVITY_REF_DENSITY = 12.0
PIANO_SCALE = 4.0
PIANO_MODEL_SCALE = 2.0
PIANO_MATERIAL_BRIGHTNESS = 1.45
PIANO_SPECULAR_INTENSITY = 0.6
PIANO_SHININESS = 48.0
STRING_GLOW_COLOR = (1.0, 0.35, 0.05)
STRING_BASE_COLOR = (0.6, 0.05, 0.03)
STRING_GLOW_INTENSITY = 4.0
STRING_LINE_WIDTH = 6.0
STRING_GLOW_THICKNESS = 4.6
STRING_MESH_WIDTH_SCALE = 5.2
STRING_FRONT_RATIO = 0.22
STRING_BACK_RATIO = 0.45
KEY_VISUAL_LIFT = 0.02
STRING_RENDER_SCALE = 0.58
STRING_LIGHT_CONSTANT = 1.0
STRING_LIGHT_LINEAR = 0.6
STRING_LIGHT_QUADRATIC = 0.35
SKYBOX_FAR = 120.0
SKYBOX_DRAW_MODE = "sphere"
SKYBOX_TEXTURE_BASENAME = "skybox12"
SKYBOX_CUBE_FACE_BASES = (SKYBOX_TEXTURE_BASENAME,) * 6
SKYBOX_SPHERE_SEGMENTS = 96
SKYBOX_VERTICAL_FLIP = True

WATER_IDLE_DAMPING = 0.9
WATER_IDLE_THRESHOLD = 0.06
WATER_IDLE_EPS = 0.00015
RAIN_ACTIVITY_RISE = 3.0
RAIN_ACTIVITY_FALL = 4.5
RAIN_ACTIVITY_FLOOR = 0.18
RAIN_ACTIVITY_GAMMA = 1.6
RAIN_COUNT_GAMMA = 3.0
RAIN_COUNT_RESPONSE_RISE = 2.2
RAIN_COUNT_RESPONSE_FALL = 3.2
RAIN_DROP_STRENGTH_MIN = 0.06
RAIN_DROP_STRENGTH_MAX = 0.9
RAIN_SPEED_MIN = 0.4
RAIN_SPEED_MAX = 1.6
RAIN_TOP_HEIGHT = 24.0
RAIN_AIR_FADE_OUT_RATE = 2.5
RAIN_AIR_FADE_IN_RATE = 6.0
RAIN_PIANO_PADDING_X = 1.2
RAIN_PIANO_PADDING_Z = 2.2

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
_buffer_width = WINDOW_WIDTH
_buffer_height = WINDOW_HEIGHT

cam_pan_x, cam_pan_y = 0.0, 0.0
cam_pan_z = 0.0
pan_speed = 0.5
cam_distance = 50.0
zoom_speed = 0.5

# MIDI事件隊列（用於接收外部事件）
midi_event_queue = queue.Queue()
skybox = None
spherical_skybox = None
ui_state = {"playing": False, "track": "No track"}
ui_callbacks = {"play_pause": None, "next": None}
string_display_list = None
PIANO_OFFSET_X = 0.0
PIANO_OFFSET_Z = 0.0
_reflection_fbo = None
_reflection_tex = None
_reflection_depth = None
_reflection_size = 0
_reflection_ready = False
_last_reflection_time = 0.0


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


def _load_image_data(path, target_ratio=None):
    image_data = None
    width = None
    height = None
    gl_format = GL_RGB
    internal_format = GL_RGB
    try:
        import pygame
    except Exception:
        pygame = None
    if pygame:
        try:
            if not pygame.get_init():
                pygame.init()
            if hasattr(pygame.image, "init"):
                pygame.image.init()
            surface = pygame.image.load(str(path))
            width, height = surface.get_size()
            if target_ratio:
                current = width / height if height else 0
                if current > target_ratio and height:
                    target_w = int(height * target_ratio)
                    if 0 < target_w < width:
                        x0 = (width - target_w) // 2
                        surface = surface.subsurface((x0, 0, target_w, height)).copy()
                        width = target_w
                elif current < target_ratio and width:
                    target_h = int(width / target_ratio)
                    if 0 < target_h < height:
                        y0 = (height - target_h) // 2
                        surface = surface.subsurface((0, y0, width, target_h)).copy()
                        height = target_h
            surface = pygame.transform.flip(surface, False, True)
            image_data = pygame.image.tostring(surface, "RGB", True)
        except Exception:
            image_data = None
    if image_data is None and sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return None
        gdiplus = ctypes.windll.gdiplus
        class GdiplusStartupInput(ctypes.Structure):
            _fields_ = [
                ("GdiplusVersion", ctypes.c_uint32),
                ("DebugEventCallback", ctypes.c_void_p),
                ("SuppressBackgroundThread", wintypes.BOOL),
                ("SuppressExternalCodecs", wintypes.BOOL),
            ]
        class Rect(ctypes.Structure):
            _fields_ = [
                ("X", ctypes.c_int),
                ("Y", ctypes.c_int),
                ("Width", ctypes.c_int),
                ("Height", ctypes.c_int),
            ]
        class BitmapData(ctypes.Structure):
            _fields_ = [
                ("Width", ctypes.c_uint32),
                ("Height", ctypes.c_uint32),
                ("Stride", ctypes.c_int),
                ("PixelFormat", ctypes.c_uint32),
                ("Scan0", ctypes.c_void_p),
                ("Reserved", ctypes.c_void_p),
            ]
        gdiplus.GdiplusStartup.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(GdiplusStartupInput), ctypes.c_void_p]
        gdiplus.GdiplusStartup.restype = ctypes.c_int
        gdiplus.GdiplusShutdown.argtypes = [ctypes.c_void_p]
        gdiplus.GdipCreateBitmapFromFile.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
        gdiplus.GdipCreateBitmapFromFile.restype = ctypes.c_int
        gdiplus.GdipGetImageWidth.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        gdiplus.GdipGetImageHeight.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        gdiplus.GdipBitmapLockBits.argtypes = [ctypes.c_void_p, ctypes.POINTER(Rect), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(BitmapData)]
        gdiplus.GdipBitmapLockBits.restype = ctypes.c_int
        gdiplus.GdipBitmapUnlockBits.argtypes = [ctypes.c_void_p, ctypes.POINTER(BitmapData)]
        gdiplus.GdipDisposeImage.argtypes = [ctypes.c_void_p]

        token = ctypes.c_void_p()
        startup = GdiplusStartupInput(1, None, False, False)
        if gdiplus.GdiplusStartup(ctypes.byref(token), ctypes.byref(startup), None) != 0:
            return None
        bitmap = ctypes.c_void_p()
        if gdiplus.GdipCreateBitmapFromFile(wintypes.LPCWSTR(str(path)), ctypes.byref(bitmap)) != 0:
            gdiplus.GdiplusShutdown(token)
            return None
        width_c = ctypes.c_uint32()
        height_c = ctypes.c_uint32()
        gdiplus.GdipGetImageWidth(bitmap, ctypes.byref(width_c))
        gdiplus.GdipGetImageHeight(bitmap, ctypes.byref(height_c))
        width = int(width_c.value)
        height = int(height_c.value)
        crop_x = 0
        crop_y = 0
        crop_w = width
        crop_h = height
        if target_ratio and height:
            current = width / height
            if current > target_ratio:
                crop_w = int(height * target_ratio)
                crop_x = max((width - crop_w) // 2, 0)
            elif current < target_ratio:
                crop_h = int(width / target_ratio)
                crop_y = max((height - crop_h) // 2, 0)
        rect = Rect(crop_x, crop_y, crop_w, crop_h)
        bmp_data = BitmapData()
        PixelFormat32bppARGB = 0x26200A
        ImageLockModeRead = 1
        if gdiplus.GdipBitmapLockBits(bitmap, ctypes.byref(rect), ImageLockModeRead, PixelFormat32bppARGB, ctypes.byref(bmp_data)) != 0:
            gdiplus.GdipDisposeImage(bitmap)
            gdiplus.GdiplusShutdown(token)
            return None
        stride = abs(bmp_data.Stride)
        buf = ctypes.string_at(bmp_data.Scan0, stride * crop_h)
        rows = [buf[i * stride:(i + 1) * stride] for i in range(crop_h)]
        rows.reverse()
        image_data = b"".join(rows)
        gdiplus.GdipBitmapUnlockBits(bitmap, ctypes.byref(bmp_data))
        gdiplus.GdipDisposeImage(bitmap)
        gdiplus.GdiplusShutdown(token)
        width = crop_w
        height = crop_h
        gl_format = GL_BGRA
        internal_format = GL_RGBA
    if image_data is None:
        return None
    return image_data, int(width), int(height), gl_format, internal_format


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
    def __init__(self, cols=WATER_GRID_COLS, rows=WATER_GRID_ROWS, size=500.0):
        # 適中網格密度，搭配向量化更新提升穩定性與幀率
        self.cols = cols
        self.rows = rows
        self.size = size
        self.spacing = size / cols

        self.buffer1 = np.zeros((cols, rows), dtype=np.float32)
        self.buffer2 = np.zeros((cols, rows), dtype=np.float32)
        self.damping = 0.99

        self.scale_h = WATER_HEIGHT_SCALE
        self._dy = self.spacing * 2.0
        self._base_blue = np.array([0.02, 0.05, 0.15], dtype=np.float32)
        self._peak_color = np.array([0.22, 0.2, 0.4], dtype=np.float32)
        self._specular_color = np.array([0.1, 0.14, 0.2], dtype=np.float32)
        self._alpha_base = WATER_ALPHA_BASE
        self._alpha_scale = WATER_ALPHA_SCALE
        self._alpha_max = WATER_ALPHA_MAX
        self._strength_scale = WATER_RIPPLE_STRENGTH_SCALE
        self._world_origin = np.array(
            [-self.size * 0.5, WATER_PLANE_Y, -self.size * 0.5], dtype=np.float32
        )
        self._scene_tex = None
        self._scene_mvp = None
        self._scene_mvp_gl = None
        self._scene_strength = 0.0
        self._scene_distort = WATER_REFLECTION_DISTORT
        self._identity_mvp = np.identity(4, dtype=np.float32)
        self._sky_tex = None
        self._reflect_strength = WATER_REFLECTION_STRENGTH
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
        self._uni_world_origin = None
        self._uni_sky = None
        self._uni_reflect = None
        self._uni_scene_tex = None
        self._uni_scene_strength = None
        self._uni_scene_distort = None
        self._uni_scene_mvp = None
        self._height_internal_format = GL_R32F
        self._height_format = GL_RED
        self._height_tex_alt = None
        self._height_rg = None
        self._sim_program = None
        self._sim_fbo = None
        self._sim_ready = False
        self._sim_uni_state = None
        self._sim_uni_texel = None
        self._sim_uni_damping = None
        self._splat_program = None
        self._splat_uni_state = None
        self._splat_uni_center = None
        self._splat_uni_radius = None
        self._splat_uni_strength = None
        self._splat_uni_texel = None
        self._use_gpu_sim = True

        self._build_mesh()

    def set_reflection_texture(self, tex_id, strength=WATER_REFLECTION_STRENGTH):
        self._sky_tex = tex_id
        self._reflect_strength = strength

    def set_scene_reflection(
        self,
        tex_id=None,
        mvp=None,
        strength=WATER_SCENE_REFLECTION_STRENGTH,
        distortion=WATER_REFLECTION_DISTORT,
    ):
        self._scene_tex = tex_id
        self._scene_mvp = mvp
        if tex_id is None or mvp is None:
            self._scene_strength = 0.0
            self._scene_mvp_gl = None
        else:
            self._scene_strength = _clamp(float(strength), 0.0, 1.0)
            self._scene_mvp_gl = np.array(mvp.T, dtype=np.float32)
        self._scene_distort = max(0.0, float(distortion))

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
        uniform vec3 u_world_origin;
        uniform mat4 u_scene_mvp;
        varying vec3 v_normal;
        varying vec3 v_color;
        varying float v_alpha;
        varying vec4 v_scene_proj;
        void main() {
            float h = texture2D(u_height, a_uv).r;
            float hL = texture2D(u_height, a_uv + vec2(-u_texel.x, 0.0)).r;
            float hR = texture2D(u_height, a_uv + vec2(u_texel.x, 0.0)).r;
            float hD = texture2D(u_height, a_uv + vec2(0.0, -u_texel.y)).r;
            float hU = texture2D(u_height, a_uv + vec2(0.0, u_texel.y)).r;
            float y = h * u_height_scale;
            vec3 normal = normalize(vec3((hL - hR) * u_height_scale, u_normal_y, (hD - hU) * u_height_scale));
            v_normal = normal;
            float blend = clamp(y + 0.5, 0.0, 1.0);
            vec3 color = mix(u_base_color, u_peak_color, blend);
            float spec = pow(clamp(normal.y, 0.0, 1.0), 2.0);
            vec3 light_dir = normalize(vec3(10.0, 12.0, 8.0));
            float diffuse = max(dot(normal, light_dir), 0.0);
            color = color * (0.5 + 0.5 * diffuse);
            color += u_spec_color * spec;
            v_color = clamp(color, 0.0, 1.0);
            float alpha = clamp(u_alpha_base + abs(h) * u_alpha_scale, 0.0, u_alpha_max);
            v_alpha = alpha;
            vec3 world_pos = vec3(
                a_pos.x + u_world_origin.x,
                y + u_world_origin.y,
                a_pos.y + u_world_origin.z
            );
            v_scene_proj = u_scene_mvp * vec4(world_pos, 1.0);
            gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos.x, y, a_pos.y, 1.0);
        }
        """

        fragment_src = """
        #version 120
        varying vec3 v_normal;
        varying vec3 v_color;
        varying float v_alpha;
        varying vec4 v_scene_proj;
        uniform sampler2D u_sky;
        uniform float u_reflect_strength;
        uniform sampler2D u_scene;
        uniform float u_scene_strength;
        uniform float u_scene_distort;
        const float PI = 3.14159265;
        void main() {
            vec3 n = normalize(v_normal);
            float u = atan(n.z, n.x) / (2.0 * PI) + 0.5;
            float v = asin(clamp(n.y, -1.0, 1.0)) / PI + 0.5;
            vec3 sky = texture2D(u_sky, vec2(u, v)).rgb;
            vec3 color = mix(v_color, sky, u_reflect_strength);
            vec2 scene_uv = v_scene_proj.xy / max(v_scene_proj.w, 0.0001);
            scene_uv = scene_uv * 0.5 + 0.5;
            scene_uv += n.xz * u_scene_distort;
            float inside = step(0.0, scene_uv.x) * step(0.0, scene_uv.y)
                * step(scene_uv.x, 1.0) * step(scene_uv.y, 1.0);
            float front = step(0.0, v_scene_proj.w);
            float fresnel = pow(1.0 - clamp(n.y, 0.0, 1.0), 3.0);
            float scene_mix = u_scene_strength * inside * front * (0.2 + 0.8 * fresnel);
            vec3 scene = texture2D(u_scene, scene_uv).rgb;
            color = mix(color, scene, scene_mix);
            gl_FragColor = vec4(color, v_alpha);
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
        self._uni_world_origin = glGetUniformLocation(self._program, "u_world_origin")
        self._uni_sky = glGetUniformLocation(self._program, "u_sky")
        self._uni_reflect = glGetUniformLocation(self._program, "u_reflect_strength")
        self._uni_scene_tex = glGetUniformLocation(self._program, "u_scene")
        self._uni_scene_strength = glGetUniformLocation(self._program, "u_scene_strength")
        self._uni_scene_distort = glGetUniformLocation(self._program, "u_scene_distort")
        self._uni_scene_mvp = glGetUniformLocation(self._program, "u_scene_mvp")

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
        err = glGetError()
        glBindTexture(GL_TEXTURE_2D, 0)
        if err != GL_NO_ERROR:
            self._gpu_ready = False
            self._use_gpu = False
            self._update_render_cache()

    def update(self, activity=1.0):
        """向量化的波動更新算法"""
        b1 = self.buffer1
        b2 = self.buffer2

        b2[1:-1, 1:-1] = (
            (b1[0:-2, 1:-1] + b1[2:, 1:-1] + b1[1:-1, 0:-2] + b1[1:-1, 2:]) * 0.5
            - b2[1:-1, 1:-1]
        ) * self.damping

        self.buffer1, self.buffer2 = self.buffer2, self.buffer1
        activity = _clamp(activity, 0.0, 1.0)
        if activity < 1.0:
            idle_damp = WATER_IDLE_DAMPING + (1.0 - WATER_IDLE_DAMPING) * activity
            if idle_damp < 1.0:
                self.buffer1 *= idle_damp
                self.buffer2 *= idle_damp
            if activity <= WATER_IDLE_THRESHOLD:
                mean = float(self.buffer1.mean())
                if abs(mean) > WATER_IDLE_EPS:
                    self.buffer1 -= mean
                    self.buffer2 -= mean
                np.absolute(self.buffer1, out=self._abs_heights)
                mask = self._abs_heights < WATER_IDLE_EPS
                self.buffer1[mask] = 0.0
                self.buffer2[mask] = 0.0
        if self._gpu_ready:
            self._upload_height_texture()
        else:
            self._update_render_cache()

    def _update_render_cache(self):
        heights = self.buffer1
        self._vertices[..., 1] = heights * self.scale_h

        dx = heights[:-2, 1:-1] - heights[2:, 1:-1]
        dz = heights[1:-1, :-2] - heights[1:-1, 2:]
        scaled_dx = dx * self.scale_h
        scaled_dz = dz * self.scale_h
        length = np.sqrt(scaled_dx * scaled_dx + scaled_dz * scaled_dz + self._dy * self._dy)
        length = np.where(length == 0.0, 1.0, length)
        self._normals[1:-1, 1:-1, 0] = scaled_dx / length
        self._normals[1:-1, 1:-1, 1] = self._dy / length
        self._normals[1:-1, 1:-1, 2] = scaled_dz / length

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
        scaled = strength * self._strength_scale
        self.buffer1[x0:x1, y0:y1] -= scaled * kernel[kx0:kx1, ky0:ky1]

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
        use_reflection = self._sky_tex is not None
        if use_reflection:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._sky_tex)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)
            glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_INTERPOLATE)
            glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_TEXTURE)
            glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_PRIMARY_COLOR)
            glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE2_RGB, GL_CONSTANT)
            glTexEnvfv(
                GL_TEXTURE_ENV,
                GL_TEXTURE_ENV_COLOR,
                [self._reflect_strength, self._reflect_strength, self._reflect_strength, 1.0],
            )
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glEnable(GL_TEXTURE_GEN_S)
            glEnable(GL_TEXTURE_GEN_T)
        else:
            glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, 128)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.1, 0.15, 0.3, 0.6])

        glPushMatrix()
        glTranslatef(
            float(self._world_origin[0]),
            float(self._world_origin[1]),
            float(self._world_origin[2]),
        )

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
        if use_reflection:
            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
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
        if self._uni_world_origin is not None:
            glUniform3f(self._uni_world_origin, *self._world_origin)
        if self._sky_tex and self._uni_sky is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._sky_tex)
            glUniform1i(self._uni_sky, 1)
            if self._uni_reflect is not None:
                glUniform1f(self._uni_reflect, self._reflect_strength)
        else:
            if self._uni_reflect is not None:
                glUniform1f(self._uni_reflect, 0.0)
        scene_ready = (
            self._scene_tex is not None
            and self._scene_mvp_gl is not None
            and self._scene_strength > 0.0
        )
        if self._uni_scene_mvp is not None:
            mvp_gl = self._scene_mvp_gl if scene_ready else self._identity_mvp
            glUniformMatrix4fv(self._uni_scene_mvp, 1, GL_FALSE, mvp_gl)
        if self._uni_scene_strength is not None:
            glUniform1f(self._uni_scene_strength, self._scene_strength if scene_ready else 0.0)
        if self._uni_scene_distort is not None:
            glUniform1f(self._uni_scene_distort, self._scene_distort)
        if scene_ready and self._uni_scene_tex is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self._scene_tex)
            glUniform1i(self._uni_scene_tex, 2)
        elif self._uni_scene_tex is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(self._uni_scene_tex, 2)

        glPushMatrix()
        glTranslatef(
            float(self._world_origin[0]),
            float(self._world_origin[1]),
            float(self._world_origin[2]),
        )

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

        if self._sky_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
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


class PianoStrings1:
    def __init__(self):
        self.string_states = {}
        self.num_strings = 88
        self.base_color = np.array([0.8, 0.6, 0.3])
        self.active_color = np.array([0.95, 0.2, 0.08])
        self.fade_speed = 2.0

    def activate_string(self, note, velocity):
        string_index = note - 21
        if 0 <= string_index < self.num_strings:
            intensity = velocity / 127.0
            self.string_states[string_index] = {
                "active": True,
                "intensity": intensity,
                "time": 0.0,
            }

    def deactivate_string(self, note):
        string_index = note - 21
        if string_index in self.string_states:
            self.string_states[string_index]["active"] = False

    def update(self, dt=0.016):
        to_remove = []
        for string_idx, state in self.string_states.items():
            if not state["active"]:
                state["intensity"] -= self.fade_speed * dt
                if state["intensity"] <= 0:
                    to_remove.append(string_idx)
            else:
                state["time"] += dt
        for idx in to_remove:
            del self.string_states[idx]

    def get_string_color(self, string_index):
        if string_index not in self.string_states:
            return None
        state = self.string_states[string_index]
        intensity = max(0, min(1, state["intensity"]))
        color = self.base_color * (1.0 - intensity) + self.active_color * intensity
        if state["active"]:
            pulse = 0.5 + 0.5 * math.sin(state["time"] * 10)
            color = color * (0.8 + 0.2 * pulse)
        return np.clip(color, 0, 1)


class StringRenderer:
    def __init__(self):
        self.string_node = None
        self.parent_node = None
        self.sorted_strings = []
        self.num_strings = 0
        self.is_ready = False

    def find_mesh_in_children(self, gltf, node_idx):
        node = gltf.nodes[node_idx]
        if node.mesh is not None:
            return node_idx
        if node.children:
            for child_idx in node.children:
                found_idx = self.find_mesh_in_children(gltf, child_idx)
                if found_idx is not None:
                    return found_idx
        return None

    def initialize(self, gltf):
        if not gltf or not gltf.nodes:
            return False
        target_node_idx = -1
        for idx, node in enumerate(gltf.nodes):
            if node.name and "string" in node.name.lower():
                target_node_idx = idx
                self.parent_node = node
                break
        if target_node_idx == -1:
            return False
        mesh_node_idx = self.find_mesh_in_children(gltf, target_node_idx)
        if mesh_node_idx is None:
            return False
        actual_node = gltf.nodes[mesh_node_idx]
        self.string_node = {"index": mesh_node_idx, "node": actual_node}
        mesh = gltf.meshes[actual_node.mesh]
        prim = mesh.primitives[0]
        raw_positions = get_buffer_data(gltf, prim.attributes.POSITION)
        if prim.indices is not None:
            raw_indices = get_buffer_data(gltf, prim.indices).flatten()
        else:
            raw_indices = np.arange(len(raw_positions))
        if raw_positions is None or raw_positions.size == 0 or len(raw_indices) < 3:
            return False

        clusters = []
        current_cluster_indices = []
        last_centroid = None
        threshold = 2.0
        for i in range(0, len(raw_indices), 3):
            idx0, idx1, idx2 = raw_indices[i], raw_indices[i + 1], raw_indices[i + 2]
            p0, p1, p2 = raw_positions[idx0], raw_positions[idx1], raw_positions[idx2]
            centroid = (p0 + p1 + p2) / 3.0
            is_new_string = False
            if last_centroid is not None:
                dist = np.linalg.norm(centroid - last_centroid)
                if dist > threshold:
                    is_new_string = True
            if is_new_string and current_cluster_indices:
                clusters.append(current_cluster_indices)
                current_cluster_indices = []
            current_cluster_indices.extend([idx0, idx1, idx2])
            last_centroid = centroid
        if current_cluster_indices:
            clusters.append(current_cluster_indices)

        temp_string_data = []
        for indices in clusters:
            unique_indices = list(set(indices))
            verts = raw_positions[unique_indices]
            sort_key = np.mean(verts[:, 0])
            temp_string_data.append({"indices": indices, "sort_key": sort_key})
        temp_string_data.sort(key=lambda x: x["sort_key"], reverse=False)

        self.sorted_strings = []
        for item in temp_string_data:
            self.sorted_strings.append({"indices": item["indices"], "positions": raw_positions})
        self.num_strings = len(self.sorted_strings)
        self.is_ready = True
        return True

    def apply_node_transform(self, node):
        if node.matrix:
            glMultMatrixf(np.array(node.matrix, dtype=np.float32))
        else:
            if node.translation:
                glTranslatef(*node.translation)
            if node.rotation:
                x, y, z, w = node.rotation
                angle = 2 * math.acos(max(-1, min(1, w)))
                s = math.sqrt(max(0, 1 - w * w))
                axis = [x / s, y / s, z / s] if s > 0.001 else [0, 1, 0]
                glRotatef(math.degrees(angle), *axis)
            if node.scale:
                glScalef(*node.scale)

    def draw(self, gltf):
        if not self.is_ready or not self.string_node or self.num_strings <= 0:
            return
        glPushMatrix()
        if self.parent_node is not None:
            self.apply_node_transform(self.parent_node)
        self.apply_node_transform(self.string_node["node"])
        for i in range(self.num_strings):
            string_data = self.sorted_strings[i]
            note_idx = int((i / self.num_strings) * 87)
            note_idx = max(0, min(87, note_idx))
            active_color = piano_strings1.get_string_color(note_idx)
            if active_color is None:
                continue
            glDisable(GL_LIGHTING)
            glColor4f(active_color[0], active_color[1], active_color[2], 1.0)
            self._render_primitive(string_data["positions"], string_data["indices"])
            glEnable(GL_LIGHTING)
        glPopMatrix()

    def _render_primitive(self, all_positions, indices):
        glBegin(GL_TRIANGLES)
        for t in range(0, len(indices), 3):
            i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
            p0 = all_positions[i0]
            p1 = all_positions[i1]
            p2 = all_positions[i2]
            d01 = np.linalg.norm(p0 - p1)
            d12 = np.linalg.norm(p1 - p2)
            d20 = np.linalg.norm(p2 - p0)
            if d01 <= d12 and d01 <= d20:
                near_a, near_b, far = p0, p1, p2
            elif d12 <= d01 and d12 <= d20:
                near_a, near_b, far = p1, p2, p0
            else:
                near_a, near_b, far = p2, p0, p1
            width_factor = STRING_MESH_WIDTH_SCALE
            orig_a = near_a.copy()
            orig_b = near_b.copy()
            mid = (orig_a + orig_b) * 0.5
            scale = 1.0 + width_factor
            near_a = mid + (orig_a - mid) * scale
            near_b = mid + (orig_b - mid) * scale
            glVertex3fv(near_a)
            glVertex3fv(near_b)
            glVertex3fv(far)
        glEnd()

def _piano_rain_area_world():
    min_x, max_x = piano_keys.key_span_x()
    min_x = min_x * piano_keys.align_scale_x + piano_keys.align_offset_x
    max_x = max_x * piano_keys.align_scale_x + piano_keys.align_offset_x
    white_min_z = piano_keys.white_key_z_offset - piano_keys.white_key_length * 0.5
    white_max_z = piano_keys.white_key_z_offset + piano_keys.white_key_length * 0.5
    black_min_z = piano_keys.black_key_z_offset - piano_keys.black_key_length * 0.5
    black_max_z = piano_keys.black_key_z_offset + piano_keys.black_key_length * 0.5
    min_z = min(white_min_z, black_min_z) + piano_keys.align_offset_z
    max_z = max(white_max_z, black_max_z) + piano_keys.align_offset_z
    min_x -= RAIN_PIANO_PADDING_X
    max_x += RAIN_PIANO_PADDING_X
    min_z -= RAIN_PIANO_PADDING_Z
    max_z += RAIN_PIANO_PADDING_Z
    center_x = (min_x + max_x) * 0.5 + PIANO_OFFSET_X
    center_z = (min_z + max_z) * 0.5 + PIANO_OFFSET_Z
    width = (max_x - min_x) * PIANO_SCALE
    depth = (max_z - min_z) * PIANO_SCALE
    return center_x * PIANO_SCALE, center_z * PIANO_SCALE, width, depth

class LightRain:
    def __init__(self, water, count=200, area=None, top=None, bottom=-2.0):
        self._water = water
        self._count = count
        if area is None:
            area = getattr(water, "size", 40.0)
        if isinstance(area, (tuple, list)) and len(area) == 2:
            area_x, area_z = area
        else:
            area_x = area
            area_z = area
        if top is None:
            top = RAIN_TOP_HEIGHT
        self._center_x = 0.0
        self._center_z = 0.0
        self._half_x = float(area_x) * 0.5
        self._half_z = float(area_z) * 0.5
        self._top = float(top)
        self._bottom = float(bottom)
        self._drops = []
        self._last_time = time.perf_counter()
        self._activity = 0.0
        self._air_fade = 1.0
        self._count_level = 0.0
        self._active_count = 0
        self._target_count = 0
        for _ in range(count):
            self._drops.append(self._new_drop(random_y=True))

    def set_area(self, center_x, center_z, width, depth):
        self._center_x = float(center_x)
        self._center_z = float(center_z)
        self._half_x = max(0.5, float(width) * 0.5)
        self._half_z = max(0.5, float(depth) * 0.5)
        for idx in range(self._count):
            self._drops[idx] = self._new_drop(random_y=True)

    def _new_drop(self, random_y=False):
        x = random.uniform(self._center_x - self._half_x, self._center_x + self._half_x)
        z = random.uniform(self._center_z - self._half_z, self._center_z + self._half_z)
        y = random.uniform(self._bottom, self._top) if random_y else self._top
        speed = random.uniform(12.0, 18.0)
        length = random.uniform(0.5, 1.1)
        return [x, y, z, speed, length]

    def update(self, now):
        dt = now - self._last_time
        if dt <= 0:
            return
        self._last_time = now
        target = _note_activity_base(now)
        if target <= RAIN_ACTIVITY_FLOOR:
            target = 0.0
        else:
            target = (target - RAIN_ACTIVITY_FLOOR) / (1.0 - RAIN_ACTIVITY_FLOOR)
            target = target * target * (3.0 - 2.0 * target)
            target = target ** RAIN_ACTIVITY_GAMMA
        if target > 0.0:
            self._air_fade = min(1.0, self._air_fade + dt * RAIN_AIR_FADE_IN_RATE)
        else:
            self._air_fade = max(0.0, self._air_fade - dt * RAIN_AIR_FADE_OUT_RATE)
        response_rate = RAIN_ACTIVITY_RISE if target > self._activity else RAIN_ACTIVITY_FALL
        response = min(1.0, dt * response_rate)
        self._activity += (target - self._activity) * response
        count_target = self._activity ** RAIN_COUNT_GAMMA
        count_rate = (
            RAIN_COUNT_RESPONSE_RISE
            if count_target > self._count_level
            else RAIN_COUNT_RESPONSE_FALL
        )
        count_response = min(1.0, dt * count_rate)
        self._count_level += (count_target - self._count_level) * count_response
        target_count = int(round(self._count * self._count_level))
        if target <= 0.0:
            target_count = 0
        target_count = max(0, min(self._count, target_count))
        self._target_count = target_count
        if target_count > self._active_count:
            for idx in range(self._active_count, target_count):
                self._drops[idx] = self._new_drop(random_y=True)
            self._active_count = target_count
        if target <= 0.0 and self._air_fade <= 0.0:
            self._active_count = 0
            return
        if self._active_count <= 0:
            return
        strength = (
            RAIN_DROP_STRENGTH_MIN
            + (RAIN_DROP_STRENGTH_MAX - RAIN_DROP_STRENGTH_MIN) * self._activity
        ) * self._air_fade
        speed_scale = RAIN_SPEED_MIN + (RAIN_SPEED_MAX - RAIN_SPEED_MIN) * self._activity
        idx = 0
        while idx < self._active_count:
            drop = self._drops[idx]
            drop[1] -= drop[3] * dt * speed_scale
            if drop[1] <= self._bottom:
                self._water.drop_rain_world(drop[0], drop[2], strength=strength, radius=2)
                if self._active_count > self._target_count:
                    last_idx = self._active_count - 1
                    self._drops[idx] = self._drops[last_idx]
                    self._drops[last_idx] = drop
                    self._active_count -= 1
                    continue
                self._drops[idx] = self._new_drop()
            idx += 1

    def draw(self):
        if self._active_count <= 0 or self._air_fade <= 0.0:
            return
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        base_alpha = (0.01 + 0.18 * self._activity) * self._air_fade
        glColor4f(0.47, 0.68, 1.0, base_alpha)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        for x, y, z, _, length in self._drops[: self._active_count]:
            glVertex3f(x, y, z)
            glVertex3f(x, y + length, z)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for x, y, z, speed, length in self._drops[: self._active_count]:
            speed_ratio = _clamp((speed - 12.0) / 6.0)
            glint_alpha = (
                (0.01 + 0.02 * speed_ratio) * (0.25 + 0.2 * self._activity)
            ) * self._air_fade
            glColor4f(0.62, 0.81, 1.0, glint_alpha)
            glint_start = y + length * 0.55
            glint_end = y + length * 0.95
            glVertex3f(x, glint_start, z)
            glVertex3f(x, glint_end, z)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    @property
    def activity(self):
        return self._activity


water_sim = WaterRipple()
piano_strings = PianoStrings(piano_keys)
piano_strings1 = PianoStrings1()
string_renderer = StringRenderer()
light_rain = LightRain(water_sim, count=LIGHT_RAIN_COUNT)
_recent_note_events = collections.deque()
_recent_velocity_sum = 0.0


def _note_activity_base(now):
    global _recent_velocity_sum
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
    base_mix = 0.6 * avg_vel + 0.4 * density_norm
    return min(max(base_mix, 0.0), 1.0)


def _update_note_activity(now, velocity):
    global _recent_velocity_sum
    _recent_note_events.append((now, velocity))
    _recent_velocity_sum += velocity
    base_mix = _note_activity_base(now)
    activity = 0.2 + 0.8 * base_mix
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
    if note is None:
        return
    try:
        note = int(note)
    except (TypeError, ValueError):
        return
    velocity = event.get("velocity", 64)
    try:
        velocity = int(velocity)
    except (TypeError, ValueError):
        velocity = 0
    velocity = max(0, min(127, velocity))
    
    if event_type == "on" and velocity > 0:
        now = time.perf_counter()
        activity = _update_note_activity(now, velocity)
        success = piano_keys.press_key(note, velocity)
        if success:
            piano_strings.press(note, velocity)
            piano_strings1.activate_string(note, velocity)
            water_sim.drop_rain_for_note(note, velocity=velocity, activity=activity)
        if DEBUG:
            print(f"按下琴鍵 Note={note}, Velocity={velocity}")
    
    elif event_type in ("off", "on"):
        now = time.perf_counter()
        success = piano_keys.release_key(note)
        piano_strings1.deactivate_string(note)
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
    r = g = b = 1.0
    if material_index is not None and material_index >= 0:
        material = gltf.materials[material_index]
        pbr = material.pbrMetallicRoughness
        if pbr and pbr.baseColorFactor:
            r, g, b, _ = pbr.baseColorFactor
    boost = PIANO_MATERIAL_BRIGHTNESS
    r = _clamp(r * boost)
    g = _clamp(g * boost)
    b = _clamp(b * boost)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glMaterialfv(
        GL_FRONT_AND_BACK,
        GL_SPECULAR,
        [PIANO_SPECULAR_INTENSITY, PIANO_SPECULAR_INTENSITY, PIANO_SPECULAR_INTENSITY, 1.0],
    )
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, PIANO_SHININESS)
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
    component_type_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    dtype = component_type_map.get(accessor.componentType, np.float32)
    type_dim = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}.get(accessor.type,1)
    count = accessor.count
    if count <= 0 or type_dim <= 0:
        return np.array([])
    component_size = np.dtype(dtype).itemsize
    element_size = component_size * type_dim
    stride = view.byteStride or element_size
    buffer = memoryview(blob)[byte_offset:byte_offset + byte_length]
    if stride == element_size:
        arr = np.frombuffer(buffer, dtype=dtype, count=count * type_dim)
        if arr.size < count * type_dim:
            return np.array([])
        return arr.reshape(count, type_dim)
    needed = (count - 1) * stride + element_size
    if len(buffer) < needed:
        return np.array([])
    arr = np.ndarray(
        shape=(count, type_dim),
        dtype=dtype,
        buffer=buffer,
        offset=0,
        strides=(stride, component_size),
    )
    return np.array(arr, copy=True)

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
        if node.rotation:
            x, y, z, w = node.rotation
            angle = 2 * math.acos(max(-1, min(1, w)))
            s = math.sqrt(max(0, 1 - w * w))
            axis = [x / s, y / s, z / s] if s > 0.001 else [0, 1, 0]
            glRotatef(math.degrees(angle), *axis)
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
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4, order="F")
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
    def __init__(self, folder, face_bases=None):
        self._folder = Path(folder)
        self._face_bases = tuple(face_bases) if face_bases else None
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
        if self._face_bases:
            bases = self._face_bases
        else:
            bases = ("skybox0", "skybox1", "skybox2", "skybox3", "skybox4", "skybox5")
        faces = [(base, 0) for base in bases]
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
            loaded = _load_image_data(path, target_ratio=1.0)
            if not loaded:
                return False
            image_data, width, height, gl_format, internal_format = loaded
            if rot:
                # Fallback: ignore rotation for cube faces when using raw bytes
                pass
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            if (
                GL_TEXTURE_MAX_ANISOTROPY_EXT is not None
                and GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT is not None
            ):
                try:
                    max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
                except Exception:
                    pass
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                internal_format,
                width,
                height,
                0,
                gl_format,
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


class SphericalSkybox:
    def __init__(self, radius=200.0, segments=48):
        self.radius = radius
        self.segments = segments
        self.texture_id = None
        self._ready = False
        self._gl_format = GL_RGB
        self._internal_format = GL_RGB
        self._projection = "equirect"
        self._width = 0
        self._height = 0

    def _select_projection(self, width, height):
        if height <= 0:
            return "equirect"
        ratio = width / height
        if 0.9 <= ratio <= 1.1:
            return "angular"
        if 1.9 <= ratio <= 2.1:
            return "equirect"
        return "equirect"

    def _angular_uv(self, x, y, z):
        r = math.sqrt(x * x + y * y + z * z)
        if r <= 0.0:
            return 0.5, 0.5
        dx = x / r
        dy = y / r
        dz = z / r
        phi = math.acos(_clamp(dy, -1.0, 1.0))
        theta = math.atan2(dz, dx)
        radius = phi / math.pi
        u = 0.5 + 0.5 * radius * math.cos(theta)
        v = 0.5 + 0.5 * radius * math.sin(theta)
        v = 1.0 - v
        if SKYBOX_VERTICAL_FLIP:
            v = 1.0 - v
        return _clamp(u, 0.0, 1.0), _clamp(v, 0.0, 1.0)

    def load_texture(self, image_path):
        loaded = _load_image_data(image_path)
        if not loaded:
            return False
        image_data, width, height, gl_format, internal_format = loaded
        self._width = int(width)
        self._height = int(height)
        self._projection = self._select_projection(self._width, self._height)
        if self._projection == "angular" and self._width > 0 and self._height > 0:
            max_dim = max(self._width, self._height)
            target = max(self.segments, min(180, max_dim // 32))
            self.segments = int(target)
        self._gl_format = gl_format
        self._internal_format = internal_format
        if self.texture_id is not None:
            try:
                glDeleteTextures([self.texture_id])
            except Exception:
                pass
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        wrap_s = GL_REPEAT if self._projection == "equirect" else GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        if (
            GL_TEXTURE_MAX_ANISOTROPY_EXT is not None
            and GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT is not None
        ):
            try:
                max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
            except Exception:
                pass
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            self._internal_format,
            width,
            height,
            0,
            self._gl_format,
            GL_UNSIGNED_BYTE,
            image_data,
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        self._ready = True
        return True

    def draw(self):
        if not self._ready or self.texture_id is None:
            return False
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        drawn = False
        if self._projection == "equirect" and not SKYBOX_VERTICAL_FLIP:
            try:
                quad = gluNewQuadric()
                gluQuadricTexture(quad, GL_TRUE)
                gluQuadricOrientation(quad, GLU_INSIDE)
                gluSphere(quad, self.radius, self.segments, self.segments)
                gluDeleteQuadric(quad)
                drawn = True
            except Exception:
                drawn = False
        if not drawn:
            self._draw_sphere_fallback(projection=self._projection)
        glPopMatrix()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        return True

    def _draw_sphere_fallback(self, projection="equirect"):
        lat_steps = max(8, int(self.segments))
        lon_steps = max(8, int(self.segments) * 2)
        radius = float(self.radius)
        u_eps = 0.0
        v_eps = 0.0
        if projection == "equirect" and self._width > 0 and self._height > 0:
            u_eps = 0.5 / self._width
            v_eps = 0.5 / self._height
        for i in range(lat_steps):
            lat0 = math.pi * (-0.5 + (i / lat_steps))
            lat1 = math.pi * (-0.5 + ((i + 1) / lat_steps))
            y0 = math.sin(lat0) * radius
            y1 = math.sin(lat1) * radius
            r0 = math.cos(lat0) * radius
            r1 = math.cos(lat1) * radius
            v0 = 1.0 - (i / lat_steps)
            v1 = 1.0 - ((i + 1) / lat_steps)
            if projection != "angular":
                v0 = 1.0 - v0
                v1 = 1.0 - v1
            if SKYBOX_VERTICAL_FLIP:
                v0 = 1.0 - v0
                v1 = 1.0 - v1
            glBegin(GL_QUAD_STRIP)
            for j in range(lon_steps + 1):
                lon = (j / lon_steps) * math.tau
                x = math.cos(lon)
                z = math.sin(lon)
                if projection == "angular":
                    u0, vv0 = self._angular_uv(r0 * x, y0, r0 * z)
                    u1, vv1 = self._angular_uv(r1 * x, y1, r1 * z)
                else:
                    u = j / lon_steps
                    u = u_eps + (1.0 - 2.0 * u_eps) * u
                    v0_adj = v_eps + (1.0 - 2.0 * v_eps) * v0
                    v1_adj = v_eps + (1.0 - 2.0 * v_eps) * v1
                    u0, vv0 = u, v0_adj
                    u1, vv1 = u, v1_adj
                glTexCoord2f(u0, vv0)
                glVertex3f(r0 * x, y0, r0 * z)
                glTexCoord2f(u1, vv1)
                glVertex3f(r1 * x, y1, r1 * z)
            glEnd()


def draw_sky():
    if SKYBOX_DRAW_MODE == "sphere":
        if spherical_skybox and spherical_skybox.draw():
            return
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
    font = GLUT_BITMAP_HELVETICA_18 or GLUT_BITMAP_8_BY_13
    if font:
        for ch in label[:48]:
            glutBitmapCharacter(font, ord(ch))

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
# Window sizing helpers
# ============================================================

def _get_drawable_size():
    w = window_width
    h = window_height
    try:
        w = glutGet(GLUT_WINDOW_WIDTH)
        h = glutGet(GLUT_WINDOW_HEIGHT)
    except Exception:
        pass
    buf_w = w
    buf_h = h
    buffer_w_const = globals().get("GLUT_WINDOW_BUFFER_WIDTH")
    buffer_h_const = globals().get("GLUT_WINDOW_BUFFER_HEIGHT")
    if buffer_w_const is not None and buffer_h_const is not None:
        try:
            buf_w = glutGet(buffer_w_const)
            buf_h = glutGet(buffer_h_const)
        except Exception:
            buf_w, buf_h = w, h
    return w, h, buf_w, buf_h


def _update_viewport():
    global window_width, window_height, _buffer_width, _buffer_height
    w, h, buf_w, buf_h = _get_drawable_size()
    if w > 0 and h > 0:
        window_width = w
        window_height = h
    if buf_w <= 0 or buf_h <= 0:
        buf_w = max(1, w)
        buf_h = max(1, h)
    if buf_w != _buffer_width or buf_h != _buffer_height:
        _buffer_width = buf_w
        _buffer_height = buf_h
        glViewport(0, 0, buf_w, buf_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, buf_w / buf_h, 0.1, SKYBOX_FAR)
        glMatrixMode(GL_MODELVIEW)


def _compute_camera():
    cam_x = cam_pan_x
    cam_y = 6.0 + cam_pan_y
    cam_z = cam_distance + cam_pan_z
    rot_y_rad = math.radians(rot_y)
    rot_x_rad = math.radians(rot_x)
    forward_x = math.sin(rot_y_rad) * math.cos(rot_x_rad)
    forward_y = -math.sin(rot_x_rad)
    forward_z = -math.cos(rot_y_rad) * math.cos(rot_x_rad)
    return cam_x, cam_y, cam_z, forward_x, forward_y, forward_z


def _apply_string_light():
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


def _draw_scene(include_water=True, include_rain=True):
    glPushMatrix()

    glPushMatrix()
    glScalef(PIANO_SCALE, PIANO_SCALE, PIANO_SCALE)
    glTranslatef(PIANO_OFFSET_X, 0.0, PIANO_OFFSET_Z)

    if piano_display_list:
        glPushMatrix()
        glTranslatef(0.0, 2.2, 0.0)
        glScalef(PIANO_MODEL_SCALE, PIANO_MODEL_SCALE, PIANO_MODEL_SCALE)
        glColor4f(1, 1, 1, 1)
        glCallList(piano_display_list)
        glScalef(STRING_RENDER_SCALE, STRING_RENDER_SCALE, STRING_RENDER_SCALE)
        if model and string_renderer.string_node:
            string_renderer.draw(model)
        glPopMatrix()

    piano_keys.draw()
    glPopMatrix()

    piano_strings.draw()
    if include_rain and LIGHT_RAIN_ENABLED:
        light_rain.draw()
    if include_water:
        water_sim.draw()

    glPopMatrix()


def _current_mvp():
    proj = np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32).reshape(4, 4).T
    model = np.array(glGetFloatv(GL_MODELVIEW_MATRIX), dtype=np.float32).reshape(4, 4).T
    return proj @ model


def _reflection_target_size():
    base = max(_buffer_width, _buffer_height)
    size = int(base * REFLECTION_TEXTURE_SCALE)
    size = max(REFLECTION_TEXTURE_MIN, min(REFLECTION_TEXTURE_MAX, size))
    return size


def _release_reflection_target():
    global _reflection_fbo, _reflection_tex, _reflection_depth, _reflection_ready, _reflection_size
    if _reflection_fbo:
        glDeleteFramebuffers(1, [_reflection_fbo])
        _reflection_fbo = None
    if _reflection_tex:
        glDeleteTextures(1, [_reflection_tex])
        _reflection_tex = None
    if _reflection_depth:
        glDeleteRenderbuffers(1, [_reflection_depth])
        _reflection_depth = None
    _reflection_ready = False
    _reflection_size = 0


def _ensure_reflection_target():
    global _reflection_fbo, _reflection_tex, _reflection_depth, _reflection_ready, _reflection_size
    size = _reflection_target_size()
    if size <= 0:
        _reflection_ready = False
        return False
    if _reflection_ready and _reflection_size == size:
        return True
    _release_reflection_target()
    try:
        _reflection_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, _reflection_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            size,
            size,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            None,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        _reflection_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, _reflection_depth)
        try:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size)
        except Exception:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, size, size)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        _reflection_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, _reflection_fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _reflection_tex, 0
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _reflection_depth
        )
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            _release_reflection_target()
            return False
    except Exception:
        _release_reflection_target()
        return False
    _reflection_ready = True
    _reflection_size = size
    return True


def _render_reflection_pass(cam_x, cam_y, cam_z, forward_x, forward_y, forward_z):
    if not _ensure_reflection_target():
        water_sim.set_scene_reflection(None, None)
        return
    plane_y = WATER_PLANE_Y
    refl_cam_y = 2.0 * plane_y - cam_y
    refl_target_y = 2.0 * plane_y - (cam_y + forward_y)

    glBindFramebuffer(GL_FRAMEBUFFER, _reflection_fbo)
    glViewport(0, 0, _reflection_size, _reflection_size)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    aspect = _buffer_width / _buffer_height if _buffer_height else 1.0
    gluPerspective(45, aspect, 0.1, SKYBOX_FAR)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    gluLookAt(
        cam_x,
        refl_cam_y,
        cam_z,
        cam_x + forward_x,
        refl_target_y,
        cam_z + forward_z,
        0,
        1,
        0,
    )

    glDisable(GL_CULL_FACE)
    draw_sky()

    glClipPlane(GL_CLIP_PLANE4, (0.0, 1.0, 0.0, -plane_y))
    glEnable(GL_CLIP_PLANE4)

    _apply_string_light()
    _draw_scene(include_water=False, include_rain=False)

    glDisable(GL_CLIP_PLANE4)
    mvp = _current_mvp()
    water_sim.set_scene_reflection(_reflection_tex, mvp)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, _buffer_width, _buffer_height)
    glEnable(GL_CULL_FACE)

# ============================================================
# OpenGL setup
# ============================================================

def init():
    global piano_display_list, model, skybox, spherical_skybox, string_display_list, PIANO_OFFSET_X, PIANO_OFFSET_Z
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_MULTISAMPLE)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    
    # 主光源（模擬陽光）
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [10, 20, 10, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.04, 0.04, 0.05, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.03, 0.03, 0.04, 1])

    # 側面補光
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-10, 8, -5, 1])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.03, 0.03, 0.04, 1])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.02, 0.02, 0.03, 1])

    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_POSITION, [0, 8, 0, 1])
    glLightfv(GL_LIGHT2, GL_DIFFUSE, [0, 0, 0, 1])
    glLightfv(GL_LIGHT2, GL_SPECULAR, [0, 0, 0, 1])
    glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, STRING_LIGHT_CONSTANT)
    glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, STRING_LIGHT_LINEAR)
    glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, STRING_LIGHT_QUADRATIC)

    # 環境光
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1])
    
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    glEnable(GL_NORMALIZE)
    
    # 深藍色背景
    glClearColor(0.03, 0.05, 0.12, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, SKYBOX_FAR)
    glMatrixMode(GL_MODELVIEW)

    if model:
        piano_display_list = create_display_list(model)
        if string_renderer.initialize(model):
            pass
        key_bounds = None
        key_nodes = find_nodes_by_name(model, "keys")
        if key_nodes:
            key_nodes = _collect_nodes(model, key_nodes)
            key_bounds = compute_bounds_for_nodes(model, key_nodes)
        string_nodes = find_nodes_by_name(model, "string")
        string_bounds = None
        if string_nodes:
            parents = _parent_map(model)
            string_node_set = set(string_nodes)
            root_string_nodes = []
            for idx in string_nodes:
                cur = parents.get(idx)
                is_child = False
                while cur is not None:
                    if cur in string_node_set:
                        is_child = True
                        break
                    cur = parents.get(cur)
                if not is_child:
                    root_string_nodes.append(idx)
            string_nodes_all = _collect_nodes(model, root_string_nodes)
            string_display_list = create_display_list_for_nodes(
                model, root_string_nodes, use_material=False
            )
            string_bounds = compute_bounds_for_nodes(model, string_nodes_all)
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
    rain_center_x, rain_center_z, rain_width, rain_depth = _piano_rain_area_world()
    light_rain.set_area(rain_center_x, rain_center_z, rain_width, rain_depth)
    skybox_path = Path(__file__).with_name("skybox")
    if skybox_path.exists():
        skybox = Skybox(skybox_path, face_bases=SKYBOX_CUBE_FACE_BASES)
        if not skybox.load():
            skybox = None
        spherical_skybox = SphericalSkybox(radius=80.0, segments=SKYBOX_SPHERE_SEGMENTS)
        img_path = None
        for base in (SKYBOX_TEXTURE_BASENAME,):
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = skybox_path / f"{base}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                break
        if img_path and not spherical_skybox.load_texture(img_path):
            spherical_skybox = None
        if spherical_skybox and spherical_skybox.texture_id:
            water_sim.set_reflection_texture(
                spherical_skybox.texture_id, strength=WATER_REFLECTION_STRENGTH
            )
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
    if LIGHT_RAIN_ENABLED:
        light_rain.update(now)
        rain_activity = light_rain.activity
    else:
        rain_activity = 0.0

    if now - _last_water_time >= 1.0 / WATER_UPDATE_HZ:
        note_activity = _note_activity_base(now)
        water_sim.update(activity=max(note_activity, rain_activity))
        _last_water_time = now

    global _last_string_time
    dt = now - _last_string_time
    piano_strings1.update(dt=dt)
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
    _update_viewport()

def display():
    global rot_x, rot_y, piano_display_list, cam_distance, _last_reflection_time
    _update_viewport()
    cam_x, cam_y, cam_z, forward_x, forward_y, forward_z = _compute_camera()
    now = time.perf_counter()
    update_interval = 0.0
    if REFLECTION_UPDATE_HZ > 0.0:
        update_interval = 1.0 / REFLECTION_UPDATE_HZ
    if update_interval <= 0.0 or _last_reflection_time <= 0.0:
        _render_reflection_pass(cam_x, cam_y, cam_z, forward_x, forward_y, forward_z)
        _last_reflection_time = now
    elif now - _last_reflection_time >= update_interval:
        _render_reflection_pass(cam_x, cam_y, cam_z, forward_x, forward_y, forward_z)
        _last_reflection_time = now
    glDisable(GL_SCISSOR_TEST)
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(
        cam_x,
        cam_y,
        cam_z,
        cam_x + forward_x,
        cam_y + forward_y,
        cam_z + forward_z,
        0,
        1,
        0,
    )

    draw_sky()
    _apply_string_light()
    _draw_scene(include_water=True, include_rain=True)
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
