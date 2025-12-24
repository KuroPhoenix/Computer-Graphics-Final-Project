from OpenGL.GL import *
from OpenGL.GLUT import *

glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
glutInitWindowSize(1, 1)
glutCreateWindow(b"glver")
print("OpenGL:", glGetString(GL_VERSION).decode())
print("GLSL:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())