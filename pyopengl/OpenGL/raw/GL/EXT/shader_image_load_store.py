'''Autogenerated by xml_generate script, do not edit!'''
from OpenGL import platform as _p, arrays
# Code generation uses this
from OpenGL.raw.GL import _types as _cs
# End users want this...
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C

import ctypes
_EXTENSION_NAME = 'GL_EXT_shader_image_load_store'
def _f( function ):
    return _p.createFunction( function,_p.PLATFORM.GL,'GL_EXT_shader_image_load_store',error_checker=_errors._error_checker)
GL_ALL_BARRIER_BITS_EXT=_C('GL_ALL_BARRIER_BITS_EXT',0xFFFFFFFF)
GL_ATOMIC_COUNTER_BARRIER_BIT_EXT=_C('GL_ATOMIC_COUNTER_BARRIER_BIT_EXT',0x00001000)
GL_BUFFER_UPDATE_BARRIER_BIT_EXT=_C('GL_BUFFER_UPDATE_BARRIER_BIT_EXT',0x00000200)
GL_COMMAND_BARRIER_BIT_EXT=_C('GL_COMMAND_BARRIER_BIT_EXT',0x00000040)
GL_ELEMENT_ARRAY_BARRIER_BIT_EXT=_C('GL_ELEMENT_ARRAY_BARRIER_BIT_EXT',0x00000002)
GL_FRAMEBUFFER_BARRIER_BIT_EXT=_C('GL_FRAMEBUFFER_BARRIER_BIT_EXT',0x00000400)
GL_IMAGE_1D_ARRAY_EXT=_C('GL_IMAGE_1D_ARRAY_EXT',0x9052)
GL_IMAGE_1D_EXT=_C('GL_IMAGE_1D_EXT',0x904C)
GL_IMAGE_2D_ARRAY_EXT=_C('GL_IMAGE_2D_ARRAY_EXT',0x9053)
GL_IMAGE_2D_EXT=_C('GL_IMAGE_2D_EXT',0x904D)
GL_IMAGE_2D_MULTISAMPLE_ARRAY_EXT=_C('GL_IMAGE_2D_MULTISAMPLE_ARRAY_EXT',0x9056)
GL_IMAGE_2D_MULTISAMPLE_EXT=_C('GL_IMAGE_2D_MULTISAMPLE_EXT',0x9055)
GL_IMAGE_2D_RECT_EXT=_C('GL_IMAGE_2D_RECT_EXT',0x904F)
GL_IMAGE_3D_EXT=_C('GL_IMAGE_3D_EXT',0x904E)
GL_IMAGE_BINDING_ACCESS_EXT=_C('GL_IMAGE_BINDING_ACCESS_EXT',0x8F3E)
GL_IMAGE_BINDING_FORMAT_EXT=_C('GL_IMAGE_BINDING_FORMAT_EXT',0x906E)
GL_IMAGE_BINDING_LAYERED_EXT=_C('GL_IMAGE_BINDING_LAYERED_EXT',0x8F3C)
GL_IMAGE_BINDING_LAYER_EXT=_C('GL_IMAGE_BINDING_LAYER_EXT',0x8F3D)
GL_IMAGE_BINDING_LEVEL_EXT=_C('GL_IMAGE_BINDING_LEVEL_EXT',0x8F3B)
GL_IMAGE_BINDING_NAME_EXT=_C('GL_IMAGE_BINDING_NAME_EXT',0x8F3A)
GL_IMAGE_BUFFER_EXT=_C('GL_IMAGE_BUFFER_EXT',0x9051)
GL_IMAGE_CUBE_EXT=_C('GL_IMAGE_CUBE_EXT',0x9050)
GL_IMAGE_CUBE_MAP_ARRAY_EXT=_C('GL_IMAGE_CUBE_MAP_ARRAY_EXT',0x9054)
GL_INT_IMAGE_1D_ARRAY_EXT=_C('GL_INT_IMAGE_1D_ARRAY_EXT',0x905D)
GL_INT_IMAGE_1D_EXT=_C('GL_INT_IMAGE_1D_EXT',0x9057)
GL_INT_IMAGE_2D_ARRAY_EXT=_C('GL_INT_IMAGE_2D_ARRAY_EXT',0x905E)
GL_INT_IMAGE_2D_EXT=_C('GL_INT_IMAGE_2D_EXT',0x9058)
GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT=_C('GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT',0x9061)
GL_INT_IMAGE_2D_MULTISAMPLE_EXT=_C('GL_INT_IMAGE_2D_MULTISAMPLE_EXT',0x9060)
GL_INT_IMAGE_2D_RECT_EXT=_C('GL_INT_IMAGE_2D_RECT_EXT',0x905A)
GL_INT_IMAGE_3D_EXT=_C('GL_INT_IMAGE_3D_EXT',0x9059)
GL_INT_IMAGE_BUFFER_EXT=_C('GL_INT_IMAGE_BUFFER_EXT',0x905C)
GL_INT_IMAGE_CUBE_EXT=_C('GL_INT_IMAGE_CUBE_EXT',0x905B)
GL_INT_IMAGE_CUBE_MAP_ARRAY_EXT=_C('GL_INT_IMAGE_CUBE_MAP_ARRAY_EXT',0x905F)
GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS_EXT=_C('GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS_EXT',0x8F39)
GL_MAX_IMAGE_SAMPLES_EXT=_C('GL_MAX_IMAGE_SAMPLES_EXT',0x906D)
GL_MAX_IMAGE_UNITS_EXT=_C('GL_MAX_IMAGE_UNITS_EXT',0x8F38)
GL_PIXEL_BUFFER_BARRIER_BIT_EXT=_C('GL_PIXEL_BUFFER_BARRIER_BIT_EXT',0x00000080)
GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT=_C('GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT',0x00000020)
GL_TEXTURE_FETCH_BARRIER_BIT_EXT=_C('GL_TEXTURE_FETCH_BARRIER_BIT_EXT',0x00000008)
GL_TEXTURE_UPDATE_BARRIER_BIT_EXT=_C('GL_TEXTURE_UPDATE_BARRIER_BIT_EXT',0x00000100)
GL_TRANSFORM_FEEDBACK_BARRIER_BIT_EXT=_C('GL_TRANSFORM_FEEDBACK_BARRIER_BIT_EXT',0x00000800)
GL_UNIFORM_BARRIER_BIT_EXT=_C('GL_UNIFORM_BARRIER_BIT_EXT',0x00000004)
GL_UNSIGNED_INT_IMAGE_1D_ARRAY_EXT=_C('GL_UNSIGNED_INT_IMAGE_1D_ARRAY_EXT',0x9068)
GL_UNSIGNED_INT_IMAGE_1D_EXT=_C('GL_UNSIGNED_INT_IMAGE_1D_EXT',0x9062)
GL_UNSIGNED_INT_IMAGE_2D_ARRAY_EXT=_C('GL_UNSIGNED_INT_IMAGE_2D_ARRAY_EXT',0x9069)
GL_UNSIGNED_INT_IMAGE_2D_EXT=_C('GL_UNSIGNED_INT_IMAGE_2D_EXT',0x9063)
GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT=_C('GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY_EXT',0x906C)
GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_EXT=_C('GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_EXT',0x906B)
GL_UNSIGNED_INT_IMAGE_2D_RECT_EXT=_C('GL_UNSIGNED_INT_IMAGE_2D_RECT_EXT',0x9065)
GL_UNSIGNED_INT_IMAGE_3D_EXT=_C('GL_UNSIGNED_INT_IMAGE_3D_EXT',0x9064)
GL_UNSIGNED_INT_IMAGE_BUFFER_EXT=_C('GL_UNSIGNED_INT_IMAGE_BUFFER_EXT',0x9067)
GL_UNSIGNED_INT_IMAGE_CUBE_EXT=_C('GL_UNSIGNED_INT_IMAGE_CUBE_EXT',0x9066)
GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT=_C('GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT',0x906A)
GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT_EXT=_C('GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT_EXT',0x00000001)
@_f
@_p.types(None,_cs.GLuint,_cs.GLuint,_cs.GLint,_cs.GLboolean,_cs.GLint,_cs.GLenum,_cs.GLint)
def glBindImageTextureEXT(index,texture,level,layered,layer,access,format):pass
@_f
@_p.types(None,_cs.GLbitfield)
def glMemoryBarrierEXT(barriers):pass
