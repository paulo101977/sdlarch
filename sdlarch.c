#include <SDL.h>
#include "libretro.h"
#include "libretro_d3d.h"
#include "glad.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <d3d11.h>
#include <SDL_syswm.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>

static ID3D11Device* g_d3d_device = NULL;
static ID3D11DeviceContext* g_d3d_context = NULL;
static IDXGISwapChain* g_swap_chain = NULL;
static ID3D11RenderTargetView* g_render_target_view = NULL;
static ID3D11Texture2D* g_core_texture = NULL;
static ID3D11ShaderResourceView* g_texture_srv = NULL;
static ID3D11SamplerState* g_sampler_state = NULL;



#endif

static SDL_Window *g_win = NULL;
static SDL_GLContext *g_ctx = NULL;
static SDL_AudioDeviceID g_pcm = 0;
static struct retro_frame_time_callback runloop_frame_time;
static retro_usec_t runloop_frame_time_last = 0;
static const uint8_t *g_kbd = NULL;
static struct retro_audio_callback audio_callback;

static GLuint g_shader_program = 0;

#ifdef _WIN32
#ifdef main
#undef main
#endif

#ifdef WinMain
#undef WinMain
#endif
#endif

static void noop() {}

static float g_scale = 3;
bool running = true;

static struct {
    uint32_t  fbo_id;
    uint32_t  rbo_id;

    int glmajor;
    int glminor;


	uint32_t  pitch;
	int tex_w, tex_h;
	int clip_w, clip_h;

	uint32_t  pixfmt;
	uint32_t  pixtype;
	uint32_t  bpp;

    struct retro_hw_render_callback hw;

    #ifdef _WIN32
    ID3D11Texture2D* tex_id;
    ID3D11RenderTargetView* rtv;
    #else
    uint32_t  tex_id;
    #endif
} g_video  = {0};

static struct {
    GLuint vao;
    GLuint vbo;
    GLuint program;

    GLint i_pos;
    GLint i_coord;
    GLint u_tex;
    GLint u_mvp;

} g_shader = {0};

static struct retro_variable *g_vars = NULL;

// d3d shaders
#ifdef _WIN32
static const char* g_vshader_src =
    "struct VS_INPUT { float4 pos : POSITION; float2 tex : TEXCOORD0; };\n"
    "struct VS_OUTPUT { float4 pos : SV_POSITION; float2 tex : TEXCOORD0; };\n"
    "VS_OUTPUT main(VS_INPUT input) {\n"
    "    VS_OUTPUT output;\n"
    "    output.pos = input.pos;\n"
    "    output.tex = input.tex;\n"
    "    return output;\n"
    "}";

static const char* g_fshader_src =
    "Texture2D texture0 : register(t0);\n"
    "SamplerState sampler0 : register(s0);\n"
    "struct PS_INPUT { float4 pos : SV_POSITION; float2 tex : TEXCOORD0; };\n"
    "float4 main(PS_INPUT input) : SV_TARGET {\n"
    "    return texture0.Sample(sampler0, input.tex);\n"
    "}";


static ID3D11VertexShader* g_vertex_shader = NULL;
static ID3D11PixelShader* g_pixel_shader = NULL;
static ID3D11InputLayout* g_input_layout = NULL;
static ID3D11Buffer* g_vertex_buffer = NULL;

#ifndef IID_ID3D11Texture2D
DEFINE_GUID(IID_ID3D11Texture2D, 0x6f15aaf2, 0xd208, 0x4e89, 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c);
#endif

#else
static const char *g_vshader_src =
    "#version 150\n"
    "in vec2 i_pos;\n"
    "in vec2 i_coord;\n"
    "out vec2 o_coord;\n"
    "uniform mat4 u_mvp;\n"
    "void main() {\n"
        "o_coord = i_coord;\n"
        "gl_Position = vec4(i_pos, 0.0, 1.0) * u_mvp;\n"
    "}";

static const char *g_fshader_src =
    "#version 150\n"
    "in vec2 o_coord;\n"
    "uniform sampler2D u_tex;\n"
    "void main() {\n"
        "gl_FragColor = texture2D(u_tex, o_coord);\n"
    "}";
#endif



static struct {
	void *handle;
	bool initialized;
	bool supports_no_game;
	// The last performance counter registered. TODO: Make it a linked list.
	struct retro_perf_counter* perf_counter_last;

	void (*retro_init)(void);
	void (*retro_deinit)(void);
	unsigned (*retro_api_version)(void);
	void (*retro_get_system_info)(struct retro_system_info *info);
	void (*retro_get_system_av_info)(struct retro_system_av_info *info);
	void (*retro_set_controller_port_device)(unsigned port, unsigned device);
	void (*retro_reset)(void);
	void (*retro_run)(void);
//	size_t retro_serialize_size(void);
//	bool retro_serialize(void *data, size_t size);
//	bool retro_unserialize(const void *data, size_t size);
//	void retro_cheat_reset(void);
//	void retro_cheat_set(unsigned index, bool enabled, const char *code);
	bool (*retro_load_game)(const struct retro_game_info *game);
//	bool retro_load_game_special(unsigned game_type, const struct retro_game_info *info, size_t num_info);
	void (*retro_unload_game)(void);
//	unsigned retro_get_region(void);
//	void *retro_get_memory_data(unsigned id);
//	size_t retro_get_memory_size(unsigned id);
} g_retro;


struct keymap {
	unsigned k;
	unsigned rk;
};

struct EnvVariable {
    const char* key;
    const char* value;
};

struct EnvVariable s_envVariables[] = {
	{ "pcsx2_enable_hw_hacks", "enabled" },
#ifdef _WIN32
    { "pcsx2_renderer", "D3D11" },
#else
    { "pcsx2_renderer", "OpenGL" },
#endif
	
	{ "pcsx2_software_clut_render", "Normal" },
	{ "pcsx2_fastboot", "enabled" },
    { "pcsx2_blending_accuracy", "Medium" },
	{ "pcsx2_pgs_ssaa", "Native" },
	{ "pcsx2_pgs_ss_tex", "disabled" },
	{ "pcsx2_pgs_deblur", "disabled" },
	{ "pcsx2_pgs_high_res_scanout", "disabled" },
	{ "pcsx2_pgs_disable_mipmaps", "disabled" },
	{ "pcsx2_nointerlacing_hint", "disabled" },
	{ "pcsx2_pcrtc_antiblur", "disabled" },
	{ "pcsx2_pcrtc_screen_offsets", "disabled" },
	{ "pcsx2_disable_interlace_offset", "disabled" },
	{ "pcsx2_deinterlace_mode", "Automatic" },
	{ "pcsx2_enable_cheats", "disabled" },
	{ "pcsx2_hint_language_unlock", "disabled" },
	{ "pcsx2_ee_cycle_rate", "100% (Normal Speed)" },
	{ "pcsx2_widescreen_hint", "disabled" },
	{ "pcsx2_uncapped_framerate_hint", "disabled" },
	{ "pcsx2_game_enhancements_hint", "disabled" },
	{ "pcsx2_ee_cycle_skip", "disabled" },
	{ "pcsx2_axis_scale1", "133%" },
	{ "pcsx2_axis_deadzone1", "0%" },
	{ "pcsx2_button_deadzone1", "0%" },
    { "pcsx2_button_deadzone2", "0%" },
	{ "pcsx2_enable_rumble1", "disabled" },
    { "pcsx2_enable_rumble2", "disabled" },
	{ "pcsx2_invert_left_stick1", "disabled" },
	{ "pcsx2_invert_right_stick1", "disabled" },
	{ "pcsx2_axis_scale2", "133%" },
	{ "pcsx2_axis_deadzone2", "15%" },
	{ "pcsx2_button_deadzone2", "0%" },
	{ "pcsx2_invert_left_stick2", "disabled" },
	{ "pcsx2_invert_right_stick2", "disabled" },
	{ "dolphin_efb_scale", "x1 (640 x 528)" },
	{ "dolphin_log_level", "Info" },
	{ "dolphin_cpu_clock_rate", "100%" },
    { "dolphin_enable_rumble", "disabled" },
	// { "dolphin_renderer", "Software" },
	// { "dolphin_fastmem", "disabled" },
	// { "dolphin_dsp_hle", "enabled" },
	// { "dolphin_dsp_jit", "enabled" },
	// { "dolphin_cpu_core", "JIT64" },
	// { "dolphin_language", "English" },
	// { "dolphin_widescreen", "disabled" },
	// { "dolphin_widescreen_hack", "disabled" },
	// { "dolphin_progressive_scan", "disabled" },
	// { "dolphin_pal60", "disabled" },
	// { "dolphin_sensor_bar_position", "Bottom" },
	// { "dolphin_wiimote_continuous_scanning", "disabled" },
	// { "dolphin_mixer_rate", "32000" },
	{ "dolphin_shader_compilation_mode", "sync" },
	// { "dolphin_max_anisotropy", "0" },
	{ "dolphin_efb_scaled_copy", "enabled" },
	{ "dolphin_efb_to_texture", "enabled" },
	// { "dolphin_efb_to_vram", "disabled" },
	// { "dolphin_fast_depth_calculation", "disabled" },
	// { "dolphin_bbox_enabled", "disabled" },
	// { "dolphin_gpu_texture_decoding", "disabled" },
	{ "dolphin_wait_for_shaders", "disabled" },
	// { "dolphin_force_texture_filtering", "disabled" },
	// { "dolphin_load_custom_textures", "disabled" },
	// { "dolphin_cheats_enabled", "disabled" },
	// { "dolphin_texture_cache_accuracy", "disabled" },
	{ "dolphin_osd_enabled", "disabled" },
    {NULL, NULL},
};

static struct keymap g_binds[] = {
    { SDL_SCANCODE_X, RETRO_DEVICE_ID_JOYPAD_A },
    { SDL_SCANCODE_Z, RETRO_DEVICE_ID_JOYPAD_B },
    { SDL_SCANCODE_A, RETRO_DEVICE_ID_JOYPAD_Y },
    { SDL_SCANCODE_S, RETRO_DEVICE_ID_JOYPAD_X },
    { SDL_SCANCODE_UP, RETRO_DEVICE_ID_JOYPAD_UP },
    { SDL_SCANCODE_DOWN, RETRO_DEVICE_ID_JOYPAD_DOWN },
    { SDL_SCANCODE_LEFT, RETRO_DEVICE_ID_JOYPAD_LEFT },
    { SDL_SCANCODE_RIGHT, RETRO_DEVICE_ID_JOYPAD_RIGHT },
    { SDL_SCANCODE_RETURN, RETRO_DEVICE_ID_JOYPAD_START },
    { SDL_SCANCODE_BACKSPACE, RETRO_DEVICE_ID_JOYPAD_SELECT },
    { SDL_SCANCODE_Q, RETRO_DEVICE_ID_JOYPAD_L },
    { SDL_SCANCODE_W, RETRO_DEVICE_ID_JOYPAD_R },
    { 0, 0 }
};

static unsigned g_joy[RETRO_DEVICE_ID_JOYPAD_R3+1] = { 0 };

#define load_sym(V, S) do {\
    if (!((*(void**)&V) = SDL_LoadFunction(g_retro.handle, #S))) \
        die("Failed to load symbol '" #S "'': %s", SDL_GetError()); \
	} while (0)
#define load_retro_sym(S) load_sym(g_retro.S, S)


static void die(const char *fmt, ...) {
	char buffer[4096];

	va_list va;
	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	fputs(buffer, stderr);
	fputc('\n', stderr);
	fflush(stderr);

	exit(EXIT_FAILURE);
}

static GLuint compile_shader(unsigned type, unsigned count, const char **strings) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, count, strings, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if (status == GL_FALSE) {
        char buffer[4096];
        glGetShaderInfoLog(shader, sizeof(buffer), NULL, buffer);
        die("Failed to compile %s shader: %s", type == GL_VERTEX_SHADER ? "vertex" : "fragment", buffer);
    }

    return shader;
}

void ortho2d(float m[4][4], float left, float right, float bottom, float top) {
    m[0][0] = 1; m[0][1] = 0; m[0][2] = 0; m[0][3] = 0;
    m[1][0] = 0; m[1][1] = 1; m[1][2] = 0; m[1][3] = 0;
    m[2][0] = 0; m[2][1] = 0; m[2][2] = 1; m[2][3] = 0;
    m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;

    m[0][0] = 2.0f / (right - left);
    m[1][1] = 2.0f / (top - bottom);
    m[2][2] = -1.0f;
    m[3][0] = -(right + left) / (right - left);
    m[3][1] = -(top + bottom) / (top - bottom);
}

#ifdef _WIN32
static HRESULT compile_d3d11_shader(const char* source, const char* entry, const char* target, ID3DBlob** blob) {
    ID3DBlob* error_blob = NULL;
    
    HRESULT hr = D3DCompile(source, strlen(source), NULL, NULL, NULL, 
                           entry, target, D3DCOMPILE_DEBUG, 0, blob, &error_blob);
    
    if (FAILED(hr) && error_blob) {
        OutputDebugStringA((char*)error_blob->lpVtbl->GetBufferPointer(error_blob));
        error_blob->lpVtbl->Release(error_blob);
    }
    
    return hr;
}

static void init_d3d11_shaders() {
    ID3DBlob* vs_blob = NULL;
    ID3DBlob* ps_blob = NULL;


    if (FAILED(compile_d3d11_shader(g_vshader_src, "main", "vs_5_0", &vs_blob))) {
        die("Failed to compile vertex shader");
    }


    if (FAILED(compile_d3d11_shader(g_fshader_src, "main", "ps_5_0", &ps_blob))) {
        die("Failed to compile pixel shader");
    }


    if (FAILED(g_d3d_device->lpVtbl->CreateVertexShader(g_d3d_device, 
        vs_blob->lpVtbl->GetBufferPointer(vs_blob), vs_blob->lpVtbl->GetBufferSize(vs_blob), 
        NULL, &g_vertex_shader))) {
        die("Failed to create vertex shader");
    }


    if (FAILED(g_d3d_device->lpVtbl->CreatePixelShader(g_d3d_device,
        ps_blob->lpVtbl->GetBufferPointer(ps_blob), ps_blob->lpVtbl->GetBufferSize(ps_blob),
        NULL, &g_pixel_shader))) {
        die("Failed to create pixel shader");
    }


    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    if (FAILED(g_d3d_device->lpVtbl->CreateInputLayout(g_d3d_device, layout, 2,
        vs_blob->lpVtbl->GetBufferPointer(vs_blob), vs_blob->lpVtbl->GetBufferSize(vs_blob),
        &g_input_layout))) {
        die("Failed to create input layout");
    }


    float vertices[] = {
        // pos              // tex
        -1.0f,  1.0f, 0.0f, 1.0f,  0.0f, 0.0f,  // top-left
         1.0f,  1.0f, 0.0f, 1.0f,  1.0f, 0.0f,  // top-right
        -1.0f, -1.0f, 0.0f, 1.0f,  0.0f, 1.0f,  // bottom-left
         1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 1.0f   // bottom-right
    };

    D3D11_BUFFER_DESC vb_desc = {0};
    vb_desc.ByteWidth = sizeof(vertices);
    vb_desc.Usage = D3D11_USAGE_DEFAULT;
    vb_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vb_data = {0};
    vb_data.pSysMem = vertices;

    if (FAILED(g_d3d_device->lpVtbl->CreateBuffer(g_d3d_device, &vb_desc, &vb_data, &g_vertex_buffer))) {
        die("Failed to create vertex buffer");
    }


    D3D11_SAMPLER_DESC sampler_desc = {0};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;

    if (FAILED(g_d3d_device->lpVtbl->CreateSamplerState(g_d3d_device, &sampler_desc, &g_sampler_state))) {
        die("Failed to create sampler state");
    }

    if (vs_blob) vs_blob->lpVtbl->Release(vs_blob);
    if (ps_blob) ps_blob->lpVtbl->Release(ps_blob);
}
#else

static void init_shaders() {
    if (g_shader_program != 0) {
        return;
    }

    GLuint vshader = compile_shader(GL_VERTEX_SHADER, 1, &g_vshader_src);
    GLuint fshader = compile_shader(GL_FRAGMENT_SHADER, 1, &g_fshader_src);
    GLuint program = glCreateProgram();

    SDL_assert(program);

    glAttachShader(program, vshader);
    glAttachShader(program, fshader);
    glLinkProgram(program);

    glDeleteShader(vshader);
    glDeleteShader(fshader);

    glValidateProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);

    if(status == GL_FALSE) {
        char buffer[4096];
        glGetProgramInfoLog(program, sizeof(buffer), NULL, buffer);
        die("Failed to link shader program: %s", buffer);
    }

    g_shader.program = program;
    g_shader.i_pos   = glGetAttribLocation(program,  "i_pos");
    g_shader.i_coord = glGetAttribLocation(program,  "i_coord");
    g_shader.u_tex   = glGetUniformLocation(program, "u_tex");
    g_shader.u_mvp   = glGetUniformLocation(program, "u_mvp");

    glGenVertexArrays(1, &g_shader.vao);
    glGenBuffers(1, &g_shader.vbo);

    glUseProgram(g_shader.program);

    glUniform1i(g_shader.u_tex, 0);

    float m[4][4];
    if (g_video.hw.bottom_left_origin)
        ortho2d(m, -1, 1, 1, -1);
    else
        ortho2d(m, -1, 1, -1, 1);

    glUniformMatrix4fv(g_shader.u_mvp, 1, GL_FALSE, (float*)m);

    glUseProgram(0);

    g_shader_program = program;
}


static void refresh_vertex_data() {
    SDL_assert(g_video.tex_w);
    SDL_assert(g_video.tex_h);
    SDL_assert(g_video.clip_w);
    SDL_assert(g_video.clip_h);

    float bottom = (float)g_video.clip_h / g_video.tex_h;
    float right  = (float)g_video.clip_w / g_video.tex_w;

    float vertex_data[] = {
        // pos, coord
        -1.0f, -1.0f, 0.0f,  bottom, // left-bottom
        -1.0f,  1.0f, 0.0f,  0.0f,   // left-top
         1.0f, -1.0f, right,  bottom,// right-bottom
         1.0f,  1.0f, right,  0.0f,  // right-top
    };

    glBindVertexArray(g_shader.vao);

    glBindBuffer(GL_ARRAY_BUFFER, g_shader.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STREAM_DRAW);

    glEnableVertexAttribArray(g_shader.i_pos);
    glEnableVertexAttribArray(g_shader.i_coord);
    glVertexAttribPointer(g_shader.i_pos, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, 0);
    glVertexAttribPointer(g_shader.i_coord, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void init_framebuffer(int width, int height)
{
    glGenFramebuffers(1, &g_video.fbo_id);
    glBindFramebuffer(GL_FRAMEBUFFER, g_video.fbo_id);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_video.tex_id, 0);

    if (g_video.hw.depth && g_video.hw.stencil) {
        glGenRenderbuffers(1, &g_video.rbo_id);
        glBindRenderbuffer(GL_RENDERBUFFER, g_video.rbo_id);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, g_video.rbo_id);
    } else if (g_video.hw.depth) {
        glGenRenderbuffers(1, &g_video.rbo_id);
        glBindRenderbuffer(GL_RENDERBUFFER, g_video.rbo_id);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_video.rbo_id);
    }

    if (g_video.hw.depth || g_video.hw.stencil)
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    SDL_assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

#endif // end linux or macOS

static void resize_cb(int w, int h) {
#ifndef _WIN32
	glViewport(0, 0, w, h);
#else
    if (g_render_target_view) {
        g_render_target_view->lpVtbl->Release(g_render_target_view);
        g_render_target_view = NULL;
    }

    g_d3d_context->lpVtbl->OMSetRenderTargets(g_d3d_context, 0, NULL, NULL);
    g_swap_chain->lpVtbl->ResizeBuffers(g_swap_chain, 0, w, h, DXGI_FORMAT_UNKNOWN, 0);

    ID3D11Texture2D* back_buffer = NULL;
    g_swap_chain->lpVtbl->GetBuffer(g_swap_chain, 0, &IID_ID3D11Texture2D, (void**)&back_buffer);
    g_d3d_device->lpVtbl->CreateRenderTargetView(g_d3d_device, (ID3D11Resource*)back_buffer, NULL, &g_render_target_view);
    back_buffer->lpVtbl->Release(back_buffer);

    D3D11_VIEWPORT viewport = {0};
    viewport.Width = (float)w;
    viewport.Height = (float)h;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    g_d3d_context->lpVtbl->RSSetViewports(g_d3d_context, 1, &viewport);

    g_d3d_context->lpVtbl->OMSetRenderTargets(g_d3d_context, 1, &g_render_target_view, NULL);
#endif
}


static void create_window(int width, int height) {
#ifndef _WIN32
    SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

    if (g_video.hw.context_type == RETRO_HW_CONTEXT_OPENGL_CORE || g_video.hw.version_major >= 3) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, g_video.hw.version_major);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, g_video.hw.version_minor);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    }

    switch (g_video.hw.context_type) {
    case RETRO_HW_CONTEXT_OPENGL_CORE:
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        break;
    case RETRO_HW_CONTEXT_OPENGLES2:
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
        break;
    case RETRO_HW_CONTEXT_OPENGL:
        if (g_video.hw.version_major >= 3)
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
        break;
    default:
        die("Unsupported hw context %i. (only OPENGL, OPENGL_CORE and OPENGLES2 supported)", g_video.hw.context_type);
    }

    g_win = SDL_CreateWindow("sdlarch", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);

    if (!g_win)
        die("Failed to create window: %s", SDL_GetError());

    g_ctx = SDL_GL_CreateContext(g_win);

    SDL_GL_MakeCurrent(g_win, g_ctx);

    if (!g_ctx)
        die("Failed to create OpenGL context: %s", SDL_GetError());

    if (g_video.hw.context_type == RETRO_HW_CONTEXT_OPENGLES2) {
        if (!gladLoadGLES2Loader((GLADloadproc)SDL_GL_GetProcAddress))
            die("Failed to initialize glad.");
    } else {
        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
            die("Failed to initialize glad.");
    }

    fprintf(stderr, "GL_SHADING_LANGUAGE_VERSION: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    fprintf(stderr, "GL_VERSION: %s\n", glGetString(GL_VERSION));


    init_shaders();

    SDL_GL_SetSwapInterval(1);
    SDL_GL_SwapWindow(g_win); // make apitrace output nicer

    resize_cb(width, height);
#else
    g_win = SDL_CreateWindow(
        "sdlarch-d3d11", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        width, height, 
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );

    if (!g_win)
        die("Failed to create window: %s", SDL_GetError());

    SDL_SysWMinfo wm_info;
    SDL_VERSION(&wm_info.version);
    SDL_GetWindowWMInfo(g_win, &wm_info);
    HWND hwnd = wm_info.info.win.window;

    printf("Window HWND: %p\n", hwnd);

    // Usar D3D11.1 explicitamente
    ID3D11Device1* device1 = NULL;
    ID3D11DeviceContext1* context1 = NULL;
    IDXGISwapChain1* swap_chain1 = NULL;

    // Feature levels - incluir 11.1
    D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // Criar dispositivo D3D11.1
    D3D_FEATURE_LEVEL actual_feature_level;
    HRESULT hr = D3D11CreateDevice(
        NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        flags,
        feature_levels,
        sizeof(feature_levels) / sizeof(feature_levels[0]),
        D3D11_SDK_VERSION,
        (ID3D11Device**)&device1,
        &actual_feature_level,
        (ID3D11DeviceContext**)&context1
    );

    if (FAILED(hr)) {
        die("Failed to create D3D11.1 device: 0x%08X", hr);
    }

    printf("D3D11.1 device created successfully!\n");
    printf("Feature Level: %d.%d\n", 
           (actual_feature_level >> 12) & 0xF, 
           (actual_feature_level >> 8) & 0xF);

    // Obter DXGI factory
    IDXGIDevice2* dxgi_device = NULL;
    hr = device1->lpVtbl->QueryInterface(device1, &IID_IDXGIDevice2, (void**)&dxgi_device);
    if (FAILED(hr)) {
        die("Failed to get DXGI device 2: 0x%08X", hr);
    }

    IDXGIAdapter* adapter = NULL;
    hr = dxgi_device->lpVtbl->GetAdapter(dxgi_device, &adapter);
    if (FAILED(hr)) {
        dxgi_device->lpVtbl->Release(dxgi_device);
        die("Failed to get adapter: 0x%08X", hr);
    }

    IDXGIFactory2* factory = NULL;
    hr = adapter->lpVtbl->GetParent(adapter, &IID_IDXGIFactory2, (void**)&factory);
    if (FAILED(hr)) {
        adapter->lpVtbl->Release(adapter);
        dxgi_device->lpVtbl->Release(dxgi_device);
        die("Failed to get DXGI factory 2: 0x%08X", hr);
    }

    // Configurar swap chain desc para D3D11.1
    DXGI_SWAP_CHAIN_DESC1 sc_desc = {0};
    sc_desc.Width = width;
    sc_desc.Height = height;
    sc_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.SampleDesc.Quality = 0;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = 2;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc_desc.Scaling = DXGI_SCALING_STRETCH;
    sc_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

    // Criar swap chain
    hr = factory->lpVtbl->CreateSwapChainForHwnd(factory, 
                                                (IUnknown*)device1, 
                                                hwnd, 
                                                &sc_desc, 
                                                NULL, 
                                                NULL, 
                                                &swap_chain1);
    if (FAILED(hr)) {
        factory->lpVtbl->Release(factory);
        adapter->lpVtbl->Release(adapter);
        dxgi_device->lpVtbl->Release(dxgi_device);
        die("Failed to create swap chain 1: 0x%08X", hr);
    }

    // Converter para interfaces regulares (para compatibilidade)
    g_d3d_device = (ID3D11Device*)device1;
    g_d3d_context = (ID3D11DeviceContext*)context1;
    
    hr = swap_chain1->lpVtbl->QueryInterface(swap_chain1, &IID_IDXGISwapChain, (void**)&g_swap_chain);
    swap_chain1->lpVtbl->Release(swap_chain1);
    
    if (FAILED(hr)) {
        factory->lpVtbl->Release(factory);
        adapter->lpVtbl->Release(adapter);
        dxgi_device->lpVtbl->Release(dxgi_device);
        die("Failed to get regular swap chain: 0x%08X", hr);
    }

    // Criar render target view
    ID3D11Texture2D* back_buffer = NULL;
    hr = g_swap_chain->lpVtbl->GetBuffer(g_swap_chain, 0, &IID_ID3D11Texture2D, (void**)&back_buffer);
    if (FAILED(hr)) {
        die("Failed to get back buffer: 0x%08X", hr);
    }

    hr = g_d3d_device->lpVtbl->CreateRenderTargetView(g_d3d_device, (ID3D11Resource*)back_buffer, NULL, &g_render_target_view);
    back_buffer->lpVtbl->Release(back_buffer);
    
    if (FAILED(hr)) {
        die("Failed to create render target view: 0x%08X", hr);
    }

    // Configurar viewport
    D3D11_VIEWPORT viewport = {0};
    viewport.Width = (float)width;
    viewport.Height = (float)height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    g_d3d_context->lpVtbl->RSSetViewports(g_d3d_context, 1, &viewport);

    g_d3d_context->lpVtbl->OMSetRenderTargets(g_d3d_context, 1, &g_render_target_view, NULL);

    // Limpar recursos
    factory->lpVtbl->Release(factory);
    adapter->lpVtbl->Release(adapter);
    dxgi_device->lpVtbl->Release(dxgi_device);

    printf("D3D11.1 initialization completed successfully!\n");
    printf("Device1: %p, Context1: %p, SwapChain: %p\n", device1, context1, g_swap_chain);

    // HRESULT hr = D3D11CreateDeviceAndSwapChain(
    //     NULL,
    //     D3D_DRIVER_TYPE_HARDWARE,
    //     NULL,
    //     flags,
    //     feature_levels,
    //     sizeof(feature_levels) / sizeof(feature_levels[0]),
    //     D3D11_SDK_VERSION,
    //     &sc_desc,
    //     &g_swap_chain,
    //     &g_d3d_device,
    //     NULL,
    //     &g_d3d_context
    // );

    // if (FAILED(hr)) {
    //     die("Failed to create D3D11 device and swap chain");
    // }

    // printf("D3D11 device created successfully >>>>>>>>>>>>>>>!\n");
    // printf("Feature Level: %d.%d\n", 
    //        (actual_feature_level >> 12) & 0xF, 
    //        (actual_feature_level >> 8) & 0xF);
    // printf("Device: %p, Context: %p\n", g_d3d_device, g_d3d_context);

    // ID3D11Texture2D* back_buffer = NULL;
    // hr = g_swap_chain->lpVtbl->GetBuffer(g_swap_chain, 0, &IID_ID3D11Texture2D, (void**)&back_buffer);
    // if (FAILED(hr)) {
    //     die("Failed to get back buffer");
    // }

    // hr = g_d3d_device->lpVtbl->CreateRenderTargetView(g_d3d_device, (ID3D11Resource*)back_buffer, NULL, &g_render_target_view);
    // back_buffer->lpVtbl->Release(back_buffer);
    
    // if (FAILED(hr)) {
    //     die("Failed to create render target view");
    // }

    // D3D11_VIEWPORT viewport = {0};
    // viewport.Width = (float)width;
    // viewport.Height = (float)height;
    // viewport.MinDepth = 0.0f;
    // viewport.MaxDepth = 1.0f;
    // g_d3d_context->lpVtbl->RSSetViewports(g_d3d_context, 1, &viewport);

    // init_d3d11_shaders();

    // g_d3d_context->lpVtbl->OMSetRenderTargets(g_d3d_context, 1, &g_render_target_view, NULL);
    
    // UINT stride = 6 * sizeof(float);
    // UINT offset = 0;
    // g_d3d_context->lpVtbl->IASetVertexBuffers(g_d3d_context, 0, 1, &g_vertex_buffer, &stride, &offset);
    // g_d3d_context->lpVtbl->IASetInputLayout(g_d3d_context, g_input_layout);
    // g_d3d_context->lpVtbl->IASetPrimitiveTopology(g_d3d_context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    // g_d3d_context->lpVtbl->VSSetShader(g_d3d_context, g_vertex_shader, NULL, 0);
    // g_d3d_context->lpVtbl->PSSetShader(g_d3d_context, g_pixel_shader, NULL, 0);
    // g_d3d_context->lpVtbl->PSSetSamplers(g_d3d_context, 0, 1, &g_sampler_state);


#endif

    // TODO: make the same in sdlarch-rl
    if (g_video.hw.context_reset) {
        g_video.hw.context_reset();
    }
}


static void resize_to_aspect(double ratio, int sw, int sh, int *dw, int *dh) {
	*dw = sw;
	*dh = sh;

	if (ratio <= 0)
		ratio = (double)sw / sh;

	if ((float)sw / sh < 1)
		*dw = *dh * ratio;
	else
		*dh = *dw / ratio;
}


static void video_configure(const struct retro_game_geometry *geom) {
	int nwidth, nheight;

	resize_to_aspect(geom->aspect_ratio, geom->base_width * 1, geom->base_height * 1, &nwidth, &nheight);

	if (!g_win)
		create_window(nwidth, nheight);

#ifndef _WIN32
	if (g_video.tex_id)
		glDeleteTextures(1, &g_video.tex_id);

	g_video.tex_id = 0;

	if (!g_video.pixfmt)
		g_video.pixfmt = GL_UNSIGNED_SHORT_5_5_5_1;

    SDL_SetWindowSize(g_win, nwidth, nheight);

	glGenTextures(1, &g_video.tex_id);

	if (!g_video.tex_id)
		die("Failed to create the video texture");

	g_video.pitch = geom->max_width * g_video.bpp;

	glBindTexture(GL_TEXTURE_2D, g_video.tex_id);
    // glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, geom->max_width, geom->max_height);
    // TODO: make the same in sdlarch-rl
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, geom->max_width, geom->max_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);

//	glPixelStorei(GL_UNPACK_ALIGNMENT, s_video.pixfmt == GL_UNSIGNED_INT_8_8_8_8_REV ? 4 : 2);
//	glPixelStorei(GL_UNPACK_ROW_LENGTH, s_video.pitch / s_video.bpp);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, geom->max_width, geom->max_height, 0,
			g_video.pixtype, g_video.pixfmt, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

    init_framebuffer(geom->max_width, geom->max_height);

	g_video.tex_w = geom->max_width;
	g_video.tex_h = geom->max_height;
	g_video.clip_w = geom->base_width;
	g_video.clip_h = geom->base_height;

	refresh_vertex_data();
#else
    if (g_video.tex_id) {
        g_video.tex_id->lpVtbl->Release(g_video.tex_id);
        g_video.tex_id = NULL;
    }

    if (g_video.rtv) {
        g_video.rtv->lpVtbl->Release(g_video.rtv);
        g_video.rtv = NULL;
    }

    D3D11_TEXTURE2D_DESC tex_desc = {0};
    tex_desc.Width = geom->max_width;
    tex_desc.Height = geom->max_height;
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    tex_desc.CPUAccessFlags = 0;

    HRESULT hr = g_d3d_device->lpVtbl->CreateTexture2D(g_d3d_device, &tex_desc, NULL, &g_video.tex_id);
    if (FAILED(hr)) {
        die("Failed to create core texture");
    }

    hr = g_d3d_device->lpVtbl->CreateRenderTargetView(g_d3d_device, (ID3D11Resource*)g_video.tex_id, NULL, &g_video.rtv);
    if (FAILED(hr)) {
        die("Failed to create RTV for core texture");
    }

    if (g_texture_srv) {
        g_texture_srv->lpVtbl->Release(g_texture_srv);
        g_texture_srv = NULL;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {0};
    srv_desc.Format = tex_desc.Format;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;

    hr = g_d3d_device->lpVtbl->CreateShaderResourceView(g_d3d_device, (ID3D11Resource*)g_video.tex_id, &srv_desc, &g_texture_srv);
    if (FAILED(hr)) {
        die("Failed to create SRV for core texture");
    }

    g_video.tex_w = geom->max_width;
    g_video.tex_h = geom->max_height;
    g_video.clip_w = geom->base_width;
    g_video.clip_h = geom->base_height;

    SDL_SetWindowSize(g_win, nwidth, nheight);
#endif

    // TODO: make the same in sdlarch-rl
    if (g_video.hw.context_reset) {
        g_video.hw.context_reset();
    }
}


static bool video_set_pixel_format(unsigned format) {
	switch (format) {
	case RETRO_PIXEL_FORMAT_0RGB1555:
		g_video.pixfmt = GL_UNSIGNED_SHORT_5_5_5_1;
		g_video.pixtype = GL_BGRA;
		g_video.bpp = sizeof(uint16_t);
		break;
	case RETRO_PIXEL_FORMAT_XRGB8888:
		g_video.pixfmt = GL_UNSIGNED_INT_8_8_8_8_REV;
		g_video.pixtype = GL_BGRA;
		g_video.bpp = sizeof(uint32_t);
		break;
	case RETRO_PIXEL_FORMAT_RGB565:
		g_video.pixfmt  = GL_UNSIGNED_SHORT_5_6_5;
		g_video.pixtype = GL_RGB;
		g_video.bpp = sizeof(uint16_t);
		break;
	default:
		die("Unknown pixel type %u", format);
	}

	return true;
}


static void video_refresh(const void *data, unsigned width, unsigned height, unsigned pitch) {
#ifndef _WIN32
    // TODO: make the same in sdlarch-rl
    if ((g_video.clip_w != width || g_video.clip_h != height) && (width != 0 && height != 0)) {
        g_video.clip_h = height;
        g_video.clip_w = width;
        refresh_vertex_data();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    if (data == RETRO_HW_FRAME_BUFFER_VALID) {
        // Hardware rendering
        glBindFramebuffer(GL_READ_FRAMEBUFFER, g_video.hw.get_current_framebuffer());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    } else if (data && data != RETRO_HW_FRAME_BUFFER_VALID) {
        // Software rendering
        glBindTexture(GL_TEXTURE_2D, g_video.tex_id);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, pitch / g_video.bpp);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        g_video.pixtype, g_video.pixfmt, data);

        glUseProgram(g_shader.program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_video.tex_id);
        glBindVertexArray(g_shader.vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    SDL_GL_SwapWindow(g_win);
#else
    if (g_d3d_device == NULL || g_d3d_context == NULL) {
        printf("ERROR: D3D11 context is NULL in video_refresh!\n");
        return;
    }
    static int frame_count = 0;
    printf("=== VIDEO_REFRESH Frame %d ===\n", frame_count);
    printf("Data pointer: %p\n", data);
    printf("Size: %ux%u, Pitch: %u\n", width, height, pitch);

    
    float clear_color[4] = {0.1f, 0.1f, 0.1f, 1.0f};

    g_d3d_context->lpVtbl->ClearRenderTargetView(g_d3d_context, g_render_target_view, clear_color);
    g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);

    if (data == RETRO_HW_FRAME_BUFFER_VALID) {
        printf(">>> HW FRAME BUFFER VALID <<<\n");
        
        // O Dolphin diz que tem um frame válido - apresentar
        HRESULT hr = g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
        printf("Present result: 0x%08X\n", hr);
        
    } else if (data == NULL) {
        printf(">>> NULL FRAME <<<\n");
        // Apenas apresentar o que já está no buffer
        g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
        
    } else {
        printf(">>> SOFTWARE FRAME - UNEXPECTED FOR D3D11 <<<\n");
        // Frame de software - isso não deveria acontecer com D3D11
        g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
    }
    
    frame_count++;
    printf("=== END FRAME %d ===\n\n", frame_count);
    
    // g_d3d_context->lpVtbl->ClearRenderTargetView(g_d3d_context, g_render_target_view, clear_color);

    // if (data == RETRO_HW_FRAME_BUFFER_VALID) {
    //     g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
        
    // } else if (data && data != RETRO_HW_FRAME_BUFFER_VALID) {
    //     if (g_video.tex_id && width > 0 && height > 0) {
    //         D3D11_BOX box = {0};
    //         box.right = width;
    //         box.bottom = height;
    //         box.back = 1;

    //         g_d3d_context->lpVtbl->UpdateSubresource(g_d3d_context, 
    //             (ID3D11Resource*)g_video.tex_id, 0, &box, data, pitch, 0);

    //         g_d3d_context->lpVtbl->PSSetShaderResources(g_d3d_context, 0, 1, &g_texture_srv);
    //         g_d3d_context->lpVtbl->Draw(g_d3d_context, 4, 0);
    //     }
        
    //     g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
    // } else {
    //     g_swap_chain->lpVtbl->Present(g_swap_chain, 1, 0);
    // }
#endif
}

static void video_deinit() {
#ifndef _WIN32
    if (g_video.fbo_id)
        glDeleteFramebuffers(1, &g_video.fbo_id);

	if (g_video.tex_id)
		glDeleteTextures(1, &g_video.tex_id);

    if (g_shader.vao)
        glDeleteVertexArrays(1, &g_shader.vao);

    if (g_shader.vbo)
        glDeleteBuffers(1, &g_shader.vbo);

    if (g_shader.program)
        glDeleteProgram(g_shader.program);

    g_video.fbo_id = 0;
	g_video.tex_id = 0;
    g_shader.vao = 0;
    g_shader.vbo = 0;
    g_shader.program = 0;

    SDL_GL_MakeCurrent(g_win, g_ctx);
    SDL_GL_DeleteContext(g_ctx);

    g_ctx = NULL;

    SDL_DestroyWindow(g_win);
#else
    if (g_texture_srv) {
        g_texture_srv->lpVtbl->Release(g_texture_srv);
        g_texture_srv = NULL;
    }
    
    if (g_video.tex_id) {
        g_video.tex_id->lpVtbl->Release(g_video.tex_id);
        g_video.tex_id = NULL;
    }
    
    if (g_video.rtv) {
        g_video.rtv->lpVtbl->Release(g_video.rtv);
        g_video.rtv = NULL;
    }
    
    if (g_vertex_shader) {
        g_vertex_shader->lpVtbl->Release(g_vertex_shader);
        g_vertex_shader = NULL;
    }
    
    if (g_pixel_shader) {
        g_pixel_shader->lpVtbl->Release(g_pixel_shader);
        g_pixel_shader = NULL;
    }
    
    if (g_input_layout) {
        g_input_layout->lpVtbl->Release(g_input_layout);
        g_input_layout = NULL;
    }
    
    if (g_vertex_buffer) {
        g_vertex_buffer->lpVtbl->Release(g_vertex_buffer);
        g_vertex_buffer = NULL;
    }
    
    if (g_sampler_state) {
        g_sampler_state->lpVtbl->Release(g_sampler_state);
        g_sampler_state = NULL;
    }
    
    if (g_render_target_view) {
        g_render_target_view->lpVtbl->Release(g_render_target_view);
        g_render_target_view = NULL;
    }
    
    if (g_swap_chain) {
        g_swap_chain->lpVtbl->Release(g_swap_chain);
        g_swap_chain = NULL;
    }
    
    if (g_d3d_context) {
        g_d3d_context->lpVtbl->Release(g_d3d_context);
        g_d3d_context = NULL;
    }
    
    if (g_d3d_device) {
        g_d3d_device->lpVtbl->Release(g_d3d_device);
        g_d3d_device = NULL;
    }

    if (g_win) {
        SDL_DestroyWindow(g_win);
        g_win = NULL;
    }
#endif
}


static void audio_init(int frequency) {
    SDL_AudioSpec desired;
    SDL_AudioSpec obtained;

    SDL_zero(desired);
    SDL_zero(obtained);

    desired.format = AUDIO_S16;
    desired.freq   = frequency;
    desired.channels = 2;
    desired.samples = 4096;

    g_pcm = SDL_OpenAudioDevice(NULL, 0, &desired, &obtained, 0);
    if (!g_pcm)
        die("Failed to open playback device: %s", SDL_GetError());

    SDL_PauseAudioDevice(g_pcm, 0);

    // Let the core know that the audio device has been initialized.
    if (audio_callback.set_state) {
        audio_callback.set_state(true);
    }
}


static void audio_deinit() {
    SDL_CloseAudioDevice(g_pcm);
}

static size_t audio_write(const int16_t *buf, unsigned frames) {
    SDL_QueueAudio(g_pcm, buf, sizeof(*buf) * frames * 2);
    return frames;
}


static void core_log(enum retro_log_level level, const char *fmt, ...) {
	char buffer[4096] = {0};
	static const char * levelstr[] = { "dbg", "inf", "wrn", "err" };
	va_list va;

	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	if (level == 0)
		return;

	fprintf(stderr, "[%s] %s", levelstr[level], buffer);
	fflush(stderr);

	if (level == RETRO_LOG_ERROR)
		exit(EXIT_FAILURE);
}

static uintptr_t core_get_current_framebuffer() {
#ifndef _WIN32
    return g_video.fbo_id;
#else
    printf("core_get_current_framebuffer called >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    if (g_video.tex_id == NULL && g_d3d_device != NULL) {
        D3D11_TEXTURE2D_DESC tex_desc = {0};
        tex_desc.Width = 640;
        tex_desc.Height = 528;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        
        HRESULT hr = g_d3d_device->lpVtbl->CreateTexture2D(g_d3d_device, &tex_desc, NULL, &g_video.tex_id);
        if (FAILED(hr)) {
            printf("Failed to create framebuffer texture for Dolphin\n");
            return 0;
        }
        
        hr = g_d3d_device->lpVtbl->CreateRenderTargetView(g_d3d_device, (ID3D11Resource*)g_video.tex_id, NULL, &g_video.rtv);
        if (FAILED(hr)) {
            printf("Failed to create RTV for Dolphin framebuffer\n");
            return 0;
        }
    }
    
    return (uintptr_t)g_video.rtv;
#endif
}

// #ifdef _WIN32
// static struct retro_hw_render_interface_d3d11 g_d3d11_interface = {
//     .interface_type = RETRO_HW_RENDER_INTERFACE_D3D11,
//     .interface_version = RETRO_HW_RENDER_INTERFACE_D3D11_VERSION,
//     .handle = NULL,
//     .device = NULL,
//     .context = NULL,
//     .featureLevel = D3D_FEATURE_LEVEL_11_0,
//     .D3DCompile = NULL
// };
// #endif

/**
 * cpu_features_get_time_usec:
 *
 * Gets time in microseconds.
 *
 * Returns: time in microseconds.
 **/
retro_time_t cpu_features_get_time_usec(void) {
    return (retro_time_t)SDL_GetTicks() * 1000;
}

/**
 * Get the CPU Features.
 *
 * @see retro_get_cpu_features_t
 * @return uint64_t Returns a bit-mask of detected CPU features (RETRO_SIMD_*).
 */
static uint64_t core_get_cpu_features() {
    uint64_t cpu = 0;
    if (SDL_HasAVX()) {
        cpu |= RETRO_SIMD_AVX;
    }
    if (SDL_HasAVX2()) {
        cpu |= RETRO_SIMD_AVX2;
    }
    if (SDL_HasMMX()) {
        cpu |= RETRO_SIMD_MMX;
    }
    if (SDL_HasSSE()) {
        cpu |= RETRO_SIMD_SSE;
    }
    if (SDL_HasSSE2()) {
        cpu |= RETRO_SIMD_SSE2;
    }
    if (SDL_HasSSE3()) {
        cpu |= RETRO_SIMD_SSE3;
    }
    if (SDL_HasSSE41()) {
        cpu |= RETRO_SIMD_SSE4;
    }
    if (SDL_HasSSE42()) {
        cpu |= RETRO_SIMD_SSE42;
    }
    return cpu;
}

/**
 * A simple counter. Usually nanoseconds, but can also be CPU cycles.
 *
 * @see retro_perf_get_counter_t
 * @return retro_perf_tick_t The current value of the high resolution counter.
 */
static retro_perf_tick_t core_get_perf_counter() {
    return (retro_perf_tick_t)SDL_GetPerformanceCounter();
}

/**
 * Register a performance counter.
 *
 * @see retro_perf_register_t
 */
static void core_perf_register(struct retro_perf_counter* counter) {
    g_retro.perf_counter_last = counter;
    counter->registered = true;
}

/**
 * Starts a registered counter.
 *
 * @see retro_perf_start_t
 */
static void core_perf_start(struct retro_perf_counter* counter) {
    if (counter->registered) {
        counter->start = core_get_perf_counter();
    }
}

/**
 * Stops a registered counter.
 *
 * @see retro_perf_stop_t
 */
static void core_perf_stop(struct retro_perf_counter* counter) {
    counter->total = core_get_perf_counter() - counter->start;
}

/**
 * Log and display the state of performance counters.
 *
 * @see retro_perf_log_t
 */
static void core_perf_log() {
    // TODO: Use a linked list of counters, and loop through them all.
    core_log(RETRO_LOG_INFO, "[timer] %s: %i - %i", g_retro.perf_counter_last->ident, g_retro.perf_counter_last->start, g_retro.perf_counter_last->total);
}

static int key_exists(const char* key) {
    for (int i = 0; s_envVariables[i].key != NULL; i++) {
        if (strcmp(s_envVariables[i].key, key) == 0) {
            return 1;
        }
    }
    return 0;
}

static bool core_environment(unsigned cmd, void *data) {
	switch (cmd) {
    case RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE:
        return false;
    case RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES: {
        uint64_t* caps = (uint64_t*)data;
        *caps = (1 << RETRO_DEVICE_JOYPAD);
        return true;
    }

    // case RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS: {
    //     return true;
    // }

    case RETRO_ENVIRONMENT_SET_VARIABLES: {
        const struct retro_variable *vars = (const struct retro_variable *)data;
        size_t num_vars = 0;

        for (const struct retro_variable *v = vars; v->key; ++v) {
            num_vars++;
        }

        g_vars = (struct retro_variable*)calloc(num_vars + 1, sizeof(*g_vars));
        for (unsigned i = 0; i < num_vars; ++i) {
            const struct retro_variable *invar = &vars[i];
            struct retro_variable *outvar = &g_vars[i];

            const char *semicolon = strchr(invar->value, ';');
            const char *first_pipe = strchr(invar->value, '|');

            SDL_assert(semicolon && *semicolon);
            semicolon++;
            while (isspace(*semicolon))
                semicolon++;

            if (first_pipe) {
                outvar->value = malloc((first_pipe - semicolon) + 1);
                memcpy((char*)outvar->value, semicolon, first_pipe - semicolon);
                ((char*)outvar->value)[first_pipe - semicolon] = '\0';
            } else {
                outvar->value = _strdup(semicolon);
            }

            outvar->key = _strdup(invar->key);

            // if(!strcmp(outvar->key, "dolphin_renderer")) {
            //     free(outvar->value);
            //     outvar->value = _strdup("Software");
            // }

            // pcsx2_enable_rumble
            if(!strcmp(outvar->key, "pcsx2_enable_rumble1")) {
                free(outvar->value);
                outvar->value = _strdup("disabled");
            }
            if(!strcmp(outvar->key, "pcsx2_button_deadzone1")) {
                free(outvar->value);
                outvar->value = _strdup("0%");
            }

            if (key_exists(outvar->key)) {
                for (int i = 0; s_envVariables[i].key != NULL; i++) {
                    if (strcmp(s_envVariables[i].key, outvar->key) == 0) {
                        outvar->value = _strdup(s_envVariables[i].value);
                        break;
                    }
                }
            }

            printf("Variable: %s = %s\n", outvar->key, outvar->value);

            SDL_assert(outvar->key && outvar->value);
        }

        return true;
    }
    case RETRO_ENVIRONMENT_GET_VARIABLE: {
        struct retro_variable *var = (struct retro_variable *)data;

        if (!g_vars)
            return false;

        for (const struct retro_variable *v = g_vars; v->key; ++v) {
            // if(!strcmp(var->key, "dolphin_renderer")) {
            //     free(var->value);
            //     var->value = _strdup("Software");
            //     break;
            // }

            if (strcmp(var->key, v->key) == 0) {
                var->value = v->value;
                break;
            }
        }

        return true;
    }
    case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE: {
        bool *bval = (bool*)data;
		*bval = false;
        return true;
    }
	case RETRO_ENVIRONMENT_GET_LOG_INTERFACE: {
		struct retro_log_callback *cb = (struct retro_log_callback *)data;
		cb->log = core_log;
        return true;
	}
    case RETRO_ENVIRONMENT_GET_PERF_INTERFACE: {
        struct retro_perf_callback *perf = (struct retro_perf_callback *)data;
        perf->get_time_usec = cpu_features_get_time_usec;
        perf->get_cpu_features = core_get_cpu_features;
        perf->get_perf_counter = core_get_perf_counter;
        perf->perf_register = core_perf_register;
        perf->perf_start = core_perf_start;
        perf->perf_stop = core_perf_stop;
        perf->perf_log = core_perf_log;
        return true;
    }
	case RETRO_ENVIRONMENT_GET_CAN_DUPE: {
		bool *bval = (bool*)data;
		*bval = true;
        return true;
    }
	case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: {
		const enum retro_pixel_format *fmt = (enum retro_pixel_format *)data;

		if (*fmt > RETRO_PIXEL_FORMAT_RGB565)
			return false;

		return video_set_pixel_format(*fmt);
	}
    case RETRO_ENVIRONMENT_SET_HW_RENDER: {
        struct retro_hw_render_callback *hw = (struct retro_hw_render_callback*)data;
        hw->get_current_framebuffer = core_get_current_framebuffer;
#ifndef _WIN32
        hw->get_proc_address = (retro_hw_get_proc_address_t)SDL_GL_GetProcAddress;
#else
        if(hw->context_type == RETRO_HW_CONTEXT_D3D11) {
            hw->get_proc_address = NULL;
            hw->context_reset = noop;
            hw->context_destroy = noop;
            hw->depth = true;
            hw->stencil = false;
            hw->bottom_left_origin = false;
            
            g_video.hw = *hw;
            printf("D3D11 context configured for Dolphin\n");
            return true;
        } else {
            printf("Unsupported context type for Dolphin: %u\n", hw->context_type);
        }
        
#endif
        g_video.hw = *hw;
        return true;
    }
    case RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER: {
        unsigned* context_type = (unsigned*)data;
#ifdef _WIN32
        *context_type = RETRO_HW_CONTEXT_D3D11;
#else
        *context_type = RETRO_HW_CONTEXT_OPENGL_CORE;
#endif
        return true;
    }


    case RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE: {
#ifdef _WIN32
        void** interface_ptr = (void**)data;
        
        struct retro_hw_render_interface_d3d11 d3d11_interface = {
            .interface_type = RETRO_HW_RENDER_INTERFACE_D3D11,
            .interface_version = RETRO_HW_RENDER_INTERFACE_D3D11_VERSION,
            .handle = NULL,
            .device = g_d3d_device,
            .context = g_d3d_context,
            .featureLevel = D3D_FEATURE_LEVEL_11_0,
            .D3DCompile = NULL
        };
        
        // Obter o feature level real do dispositivo
        if (g_d3d_device) {
            D3D_FEATURE_LEVEL feature_level = g_d3d_device->lpVtbl->GetFeatureLevel(g_d3d_device);
            d3d11_interface.featureLevel = feature_level;
            printf("Reporting feature level: %d.%d to core\n", 
                   (feature_level >> 12) & 0xF, (feature_level >> 8) & 0xF);
        } else {
            d3d11_interface.featureLevel = D3D_FEATURE_LEVEL_11_0;
            printf("WARNING: No D3D device, assuming feature level 11.0\n");
        }
        
        // Carregar D3DCompile
        HMODULE d3dcompiler = GetModuleHandleA("D3DCompiler_47.dll");
        if (!d3dcompiler) {
            d3dcompiler = LoadLibraryA("D3DCompiler_47.dll");
        }
        if (d3dcompiler) {
            d3d11_interface.D3DCompile = GetProcAddress(d3dcompiler, "D3DCompile");
        } else {
            d3d11_interface.D3DCompile = NULL;
        }
        
        *interface_ptr  = (void*)&d3d11_interface;
        return true;
#else
        return false;
#endif
    }
    case RETRO_ENVIRONMENT_SET_FRAME_TIME_CALLBACK: {
        const struct retro_frame_time_callback *frame_time =
            (const struct retro_frame_time_callback*)data;
        runloop_frame_time = *frame_time;
        return true;
    }
    case RETRO_ENVIRONMENT_SET_AUDIO_CALLBACK: {
        struct retro_audio_callback *audio_cb = (struct retro_audio_callback*)data;
        audio_callback = *audio_cb;
        return true;
    }
    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: {
        const char **dir = (const char**)data;
        *dir = "./system";
        return true;
    }
    case RETRO_ENVIRONMENT_SET_GEOMETRY: {
        const struct retro_game_geometry *geom = (const struct retro_game_geometry *)data;
        g_video.clip_w = geom->base_width;
        g_video.clip_h = geom->base_height;

        printf("Set geometry: ----->>>> %u %u %u %u\n", geom->base_width, geom->base_height, geom->max_width, geom->max_height);

        // some cores call this before we even have a window
        if (g_win) {
#ifndef _WIN32
            refresh_vertex_data();
#endif

            int ow = 0, oh = 0;
            resize_to_aspect(geom->aspect_ratio, geom->base_width, geom->base_height, &ow, &oh);

            // ow *= g_scale;
            // oh *= g_scale;

            SDL_SetWindowSize(g_win, ow, oh);
        }
        return true;
    }
    case RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME: {
        g_retro.supports_no_game = *(bool*)data;
        return true;
    }
    case RETRO_ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE: {
        int *value = (int*)data;
        *value = 1 << 0 | 1 << 1;
        return true;
    }
	default:
        // printf("Unhandled env #%u \n", cmd);
		core_log(RETRO_LOG_DEBUG, "Unhandled env #%u", cmd);
		return false;
	}

    return false;
}


static void core_video_refresh(const void *data, unsigned width, unsigned height, size_t pitch) {
    video_refresh(data, width, height, pitch);
}


static void core_input_poll(void) {
	int i;
    g_kbd = SDL_GetKeyboardState(NULL);

	for (i = 0; g_binds[i].k || g_binds[i].rk; ++i)
        g_joy[g_binds[i].rk] = g_kbd[g_binds[i].k];

    if (g_kbd[SDL_SCANCODE_ESCAPE])
        running = false;
}


static int16_t core_input_state(unsigned port, unsigned device, unsigned index, unsigned id) {
    if (port >= 1) return 0;

    if (index == RETRO_DEVICE_INDEX_ANALOG_BUTTON && device == RETRO_DEVICE_ANALOG) {
        int16_t value = g_joy[id] ? 32767 : 0;
        return value;
    }

    if (device == RETRO_DEVICE_JOYPAD && id == RETRO_DEVICE_ID_JOYPAD_MASK) {
        uint32_t mask = 0;
        for (int i = 0; i < 16; i++) {
            if (g_joy[i]) {
                mask |= (1 << i);
            }
        }
        return mask;
    }

    if (device == RETRO_DEVICE_JOYPAD && id < 16) {
        return g_joy[id] ? 1 : 0;
    }

    return 0;
}



static void core_audio_sample(int16_t left, int16_t right) {
	int16_t buf[2] = {left, right};
	audio_write(buf, 1);
}


static size_t core_audio_sample_batch(const int16_t *data, size_t frames) {
	return audio_write(data, frames);
}


static void core_load(const char *sofile) {
	void (*set_environment)(retro_environment_t) = NULL;
	void (*set_video_refresh)(retro_video_refresh_t) = NULL;
	void (*set_input_poll)(retro_input_poll_t) = NULL;
	void (*set_input_state)(retro_input_state_t) = NULL;
	void (*set_audio_sample)(retro_audio_sample_t) = NULL;
	void (*set_audio_sample_batch)(retro_audio_sample_batch_t) = NULL;
	memset(&g_retro, 0, sizeof(g_retro));
    g_retro.handle = SDL_LoadObject(sofile);

	if (!g_retro.handle)
        die("Failed to load core: %s", SDL_GetError());

	load_retro_sym(retro_init);
	load_retro_sym(retro_deinit);
	load_retro_sym(retro_api_version);
	load_retro_sym(retro_get_system_info);
	load_retro_sym(retro_get_system_av_info);
	load_retro_sym(retro_set_controller_port_device);
	load_retro_sym(retro_reset);
	load_retro_sym(retro_run);
	load_retro_sym(retro_load_game);
	load_retro_sym(retro_unload_game);

	load_sym(set_environment, retro_set_environment);
	load_sym(set_video_refresh, retro_set_video_refresh);
	load_sym(set_input_poll, retro_set_input_poll);
	load_sym(set_input_state, retro_set_input_state);
	load_sym(set_audio_sample, retro_set_audio_sample);
	load_sym(set_audio_sample_batch, retro_set_audio_sample_batch);

	set_environment(core_environment);
	set_video_refresh(core_video_refresh);
	set_input_poll(core_input_poll);
	set_input_state(core_input_state);
	set_audio_sample(core_audio_sample);
	set_audio_sample_batch(core_audio_sample_batch);

	g_retro.retro_init();
	g_retro.initialized = true;

	puts("Core loaded");
}


static void core_load_game(const char *filename) {
	struct retro_system_av_info av = {0};
	struct retro_system_info system = {0};
	struct retro_game_info info = { filename, 0 };

    info.path = filename;
    info.meta = "";
    info.data = NULL;
    info.size = 0;

    if (filename) {
        g_retro.retro_get_system_info(&system);

        if (!system.need_fullpath) {
            SDL_RWops *file = SDL_RWFromFile(filename, "rb");
            Sint64 size;

            if (!file)
                die("Failed to load %s: %s", filename, SDL_GetError());

            size = SDL_RWsize(file);

            if (size < 0)
                die("Failed to query game file size: %s", SDL_GetError());

            info.size = size;
            info.data = SDL_malloc(info.size);

            if (!info.data)
                die("Failed to allocate memory for the content");

            if (!SDL_RWread(file, (void*)info.data, info.size, 1))
                die("Failed to read file data: %s", SDL_GetError());

            SDL_RWclose(file);
        }
    }

	if (!g_retro.retro_load_game(&info))
		die("The core failed to load the content.");

	g_retro.retro_get_system_av_info(&av);

	video_configure(&av.geometry);
	audio_init(av.timing.sample_rate);

    if (info.data)
        SDL_free((void*)info.data);

    // Now that we have the system info, set the window title.
    char window_title[255];
    snprintf(window_title, sizeof(window_title), "sdlarch %s %s", system.library_name, system.library_version);
    SDL_SetWindowTitle(g_win, window_title);
}

static void core_unload() {
	if (g_retro.initialized)
		g_retro.retro_deinit();

	if (g_retro.handle)
        SDL_UnloadObject(g_retro.handle);
}

int main(int argc, char *argv[]) {
	if (argc < 2)
		die("usage: %s <core> [game]", argv[0]);

    if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO|SDL_INIT_EVENTS) < 0)
        die("Failed to initialize SDL");

#ifndef _WIN32
    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
    SDL_SetHint(SDL_HINT_RENDER_OPENGL_SHADERS, "1");
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0"); // Nearest neighbor
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");

    system("rm -rf ./system/User");

    g_video.hw.version_major = 4;
    g_video.hw.version_minor = 5;
    g_video.hw.context_type  = RETRO_HW_CONTEXT_OPENGL_CORE;
#else
    g_video.hw.context_type = RETRO_HW_CONTEXT_D3D11;
    g_video.hw.version_major = 11;
    g_video.hw.version_minor = 0;
#endif
    // g_video.hw.context_type = RETRO_HW_CONTEXT_NONE;
    g_video.hw.context_reset   = noop;
    g_video.hw.context_destroy = noop;

    create_window(640, 480);

    // Load the core.
    core_load(argv[1]);

    if (!g_retro.supports_no_game && argc < 3)
        die("This core requires a game in order to run");

    // Load the game.
    core_load_game(argc > 2 ? argv[2] : NULL);

    // Configure the player input devices.
    g_retro.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD);
    // g_retro.retro_set_controller_port_device(0, RETRO_DEVICE_KEYBOARD);

    SDL_Event ev;

    while (running) {
        // Update the game loop timer.
        if (runloop_frame_time.callback) {
            retro_time_t current = cpu_features_get_time_usec();
            retro_time_t delta = current - runloop_frame_time_last;

            if (!runloop_frame_time_last)
                delta = runloop_frame_time.reference;
            runloop_frame_time_last = current;
            runloop_frame_time.callback(delta);
        }

        // Ask the core to emit the audio.
        if (audio_callback.callback) {
            audio_callback.callback();
        }

        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT: running = false; break;
            case SDL_WINDOWEVENT:
                switch (ev.window.event) {
                case SDL_WINDOWEVENT_CLOSE: running = false; break;
                case SDL_WINDOWEVENT_RESIZED:
                    resize_cb(ev.window.data1, ev.window.data2);
                    break;
                }
            }
        }

        SDL_GL_MakeCurrent(g_win, g_ctx);
        // glBindFramebuffer(GL_FRAMEBUFFER, 0);

		g_retro.retro_run();
	}

	core_unload();
	audio_deinit();
	video_deinit();

    if (g_vars) {
        for (const struct retro_variable *v = g_vars; v->key; ++v) {
            free((char*)v->key);
            free((char*)v->value);
        }
        free(g_vars);
    }

    SDL_Quit();

    return EXIT_SUCCESS;
}