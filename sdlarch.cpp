
#include <SDL.h>
#include "libretro.h"
#include "glad.h"
#include <map>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

using namespace std;

namespace py = pybind11;

#ifdef __cplusplus
extern "C" {
#endif

static GLuint g_shader_program = 0;

static SDL_Window *g_win = NULL;
static SDL_GLContext g_ctx = NULL;
static struct retro_frame_time_callback runloop_frame_time;
static retro_usec_t runloop_frame_time_last = 0;
static const uint8_t *g_kbd = NULL;
static struct retro_audio_callback audio_callback;
static char* m_romPath;
static char* m_corePath;
static bool gameLoaded = false;

static float g_scale = 1;
bool running = true;

const int N_BUTTONS = 16;
const int MAX_PLAYERS = 2;
static bool m_buttonMask[MAX_PLAYERS][N_BUTTONS]{};

// Audio buffer; accumulated during run()
static std::vector<int16_t> audioData;
static retro_system_av_info avInfo = {};

// log function that prints to python console
void c_printf(const char* format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    py::print(buffer);
}

static struct {
	GLuint tex_id;
    GLuint fbo_id;
    GLuint rbo_id;

    int glmajor;
    int glminor;


	GLuint pitch;
	GLint tex_w, tex_h;
	GLuint clip_w, clip_h;

	GLuint pixfmt;
	GLuint pixtype;
	GLuint bpp;

    struct retro_hw_render_callback hw;
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

static map<string, const char*> s_envVariables = {
	{ "pcsx2_enable_hw_hacks", "enabled" },
	// { "pcsx2_renderer", "Software" },
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
};


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
    size_t (*retro_serialize_size)(void);
    bool (*retro_serialize)(void *data, size_t size);
    bool (*retro_unserialize)(const void *data, size_t size);
//	void retro_cheat_reset(void);
//	void retro_cheat_set(unsigned index, bool enabled, const char *code);
	bool (*retro_load_game)(const struct retro_game_info *game);
//	bool retro_load_game_special(unsigned game_type, const struct retro_game_info *info, size_t num_info);
	void (*retro_unload_game)(void);
//	unsigned retro_get_region(void);
	void* (*retro_get_memory_data)(unsigned id);
	size_t (*retro_get_memory_size)(unsigned id);
    int width;
    int height;
} g_retro;


struct keymap {
	unsigned k;
	unsigned rk;
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


static void resize_cb(int w, int h) {
	glViewport(0, 0, w, h);
}


static void create_window(int width, int height) {
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

    g_win = SDL_CreateWindow(
        "sdlarch", 
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width, 
        height, 
        SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN
    );

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

    SDL_GL_SetSwapInterval(0); // disable vsync
    SDL_GL_SwapWindow(g_win); // make apitrace output nicer

    resize_cb(width, height);
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

	nwidth *= g_scale;
	nheight *= g_scale;

	if (!g_win)
		create_window(nwidth, nheight);

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
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, geom->max_width, geom->max_height);

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

    g_retro.width = geom->base_width;
    g_retro.height = geom->base_height;

	refresh_vertex_data();

    g_video.hw.context_reset();
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
    if( width != 0 && height != 0) {
        g_retro.width = width;
        g_retro.height = height; 
    }

    if ((g_video.clip_w != width || g_video.clip_h != height) && (width != 0 && height != 0)) {
        g_video.clip_h = height;
        g_video.clip_w = width;
        refresh_vertex_data();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    if (data && data != RETRO_HW_FRAME_BUFFER_VALID) {
        glBindTexture(GL_TEXTURE_2D, g_video.tex_id);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, g_video.pitch / g_video.bpp);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        g_video.pixtype, g_video.pixfmt, data);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g_shader.program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_video.tex_id);
    glBindVertexArray(g_shader.vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    // SDL_GL_SwapWindow(g_win);
}

static void video_deinit() {
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
    return g_video.fbo_id;
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


static bool core_environment(unsigned cmd, void *data) {
	switch (cmd) {
    case RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE:
        
        return false;
    case RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES: {
        uint64_t* caps = (uint64_t*)data;
        *caps = (1 << RETRO_DEVICE_JOYPAD);
        return true;
    }

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
                outvar->value = (const char*)malloc((first_pipe - semicolon) + 1);
                memcpy((char*)outvar->value, semicolon, first_pipe - semicolon);
                ((char*)outvar->value)[first_pipe - semicolon] = '\0';
            } else {
                outvar->value = strdup(semicolon);
            }

            outvar->key = strdup(invar->key);

            if (s_envVariables.count(string(outvar->key))) {
                // var->value = s_envVariables[string(var->key)];
                outvar->value = strdup(s_envVariables[string(outvar->key)]);
            }

            if(!strcmp(outvar->key, "dolphin_renderer")) {
                free((void*)outvar->value);
                outvar->value = strdup("Software");
            }

            // if(!strcmp(outvar->key, "pcsx2_renderer")) {
            //     free((void*)outvar->value);
            //     outvar->value = strdup("Software");
            // }

            c_printf("Variable: %s = %s\n", outvar->key, outvar->value);

            SDL_assert(outvar->key && outvar->value);
        }

        return true;
    }


    case RETRO_ENVIRONMENT_GET_VARIABLE: {
        struct retro_variable *var = (struct retro_variable *)data;

        if (!g_vars)
            return false;

        // if(!strcmp(var->key, "pcsx2_renderer")) {
        //     var->value = strdup("Software");
        //     printf("Get Variable: %s = %s\n", var->key, var->value);
        //     return true;
        // }

        for (const struct retro_variable *v = g_vars; v->key; ++v) {
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
        hw->get_proc_address = (retro_hw_get_proc_address_t)SDL_GL_GetProcAddress;
        g_video.hw = *hw;
        return true;
    }
    // case RETRO_ENVIRONMENT_SET_FRAME_TIME_CALLBACK: {
    //     // const struct retro_frame_time_callback *frame_time =
    //     //     (const struct retro_frame_time_callback*)data;
    //     // runloop_frame_time = *frame_time;
    //     return true;
    // }

    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: {
        const char **dir = (const char**)data;
        *dir = ".";
        return true;
    }
    case RETRO_ENVIRONMENT_SET_GEOMETRY: {
        const struct retro_game_geometry *geom = (const struct retro_game_geometry *)data;
        g_video.clip_w = geom->base_width;
        g_video.clip_h = geom->base_height;

        g_retro.width = geom->base_width;
        g_retro.height = geom->base_height;

        // some cores call this before we even have a window
        if (g_win) {
            refresh_vertex_data();

            int ow = 0, oh = 0;
            resize_to_aspect(geom->aspect_ratio, geom->base_width, geom->base_height, &ow, &oh);

            ow *= g_scale;
            oh *= g_scale;

            SDL_SetWindowSize(g_win, ow, oh);
        }
        return true;
    }
    case RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME: {
        g_retro.supports_no_game = *(bool*)data;
        return true;
    }
	default:
		core_log(RETRO_LOG_DEBUG, "Unhandled env #%u", cmd);
		return false;
	}

    return false;
}


static void core_video_refresh(const void *data, unsigned width, unsigned height, size_t pitch) {
    video_refresh(data, width, height, pitch);
}


static void core_input_poll(void) {
}


static int16_t core_input_state(unsigned port, unsigned device, unsigned index, unsigned id) {

    if (port >= MAX_PLAYERS) return 0;

    // analog button (treat as digital)
    if (index == RETRO_DEVICE_INDEX_ANALOG_BUTTON && device == RETRO_DEVICE_ANALOG) {
        // int16_t value = g_joy[id] ? 255 : 0;
        int16_t value = g_joy[id] ? 32767 : 0;
        return value;
    }
    
    // convert to button mask (PCSX2 style)
    if (device == RETRO_DEVICE_JOYPAD && id == RETRO_DEVICE_ID_JOYPAD_MASK) {
        int16_t mask = 0;
        for (int i = 0; i < N_BUTTONS; i++) {
            if (m_buttonMask[port][i]) {
                mask |= (1 << i);
            }
        }
        return mask;
    }
    

    if (device == RETRO_DEVICE_JOYPAD && id < N_BUTTONS) {
        return m_buttonMask[port][id] ? 1 : 0;
    }
    

    if (device == RETRO_DEVICE_ANALOG) {
        return 0;
    }
    
    return 0;
}

static void core_audio_sample(int16_t left, int16_t right) {
    audioData.push_back(left);
	audioData.push_back(right);
}

static size_t core_audio_sample_batch(const int16_t *data, size_t frames) {
	audioData.insert(audioData.end(), data, &data[frames * 2]);
	return frames;
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
    load_retro_sym(retro_unserialize);
    load_retro_sym(retro_serialize);
    load_retro_sym(retro_serialize_size);
	load_retro_sym(retro_unload_game);
    load_retro_sym(retro_get_memory_data);
    load_retro_sym(retro_get_memory_size);

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

static void unload_game() {
    g_retro.retro_unload_game();
}


static void core_load_game(const char *filename) {
	// struct retro_system_av_info av = {0};
	struct retro_system_info system = {0};
	struct retro_game_info info = { filename, 0 };
    
    if (gameLoaded) {
        unload_game();
        gameLoaded = false;
    }

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

	g_retro.retro_get_system_av_info(&avInfo);
    gameLoaded = true;

	video_configure(&avInfo.geometry);

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

static void noop() {}


bool get_state(void* data) {
    size_t size = g_retro.retro_serialize_size();
    return g_retro.retro_serialize(data, size);
}

size_t get_state_size() {
    return g_retro.retro_serialize_size();
}

bool load_state(const void* data, size_t size) {
    return g_retro.retro_unserialize(data, size);
}

void get_frame(uint8_t* buffer, int width, int height) {
    SDL_GL_MakeCurrent(g_win, g_ctx);
    
    static GLuint pbo = 0;
    if (pbo == 0) {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, NULL, GL_STREAM_READ);
    }
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
    
    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(buffer, ptr, width * height * 3);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void run() {
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    SDL_GL_MakeCurrent(g_win, g_ctx);
    audioData.clear();
    g_retro.retro_run();
}

void reset() {
    memset(m_buttonMask, 0, sizeof(m_buttonMask));

	g_retro.retro_reset();
}

void setKey(int port, int key, bool active) { 
    m_buttonMask[port][key] = active; 
}

void init(char *core, char *game) {
    if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_EVENTS) < 0) {
    // if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO|SDL_INIT_EVENTS) < 0) {
        printf("SDL_Init failed: %s\n", SDL_GetError());
        die("Failed to initialize SDL");
    }

    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
    SDL_SetHint(SDL_HINT_RENDER_OPENGL_SHADERS, "1");
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0"); // Nearest neighbor
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");

    g_video.hw.version_major = 4;
    g_video.hw.version_minor = 5;
    g_video.hw.context_type  = RETRO_HW_CONTEXT_OPENGLES3;
    g_video.hw.context_reset   = noop;
    g_video.hw.context_destroy = noop;

    m_romPath = strdup(game);
    m_corePath = strdup(core);

    // Load the core.
    core_load(core);

    // Load the game.
    core_load_game(game);

    // Configure the player input devices.
    // g_retro.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD);
    g_retro.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD);
    g_retro.retro_set_controller_port_device(1, RETRO_DEVICE_JOYPAD);
}

void kill() {
    core_unload();
	video_deinit();

    if (g_vars) {
        for (const struct retro_variable *v = g_vars; v->key; ++v) {
            free((char*)v->key);
            free((char*)v->value);
        }
        free(g_vars);
    }

    SDL_Quit();
}

#ifdef __cplusplus
}
#endif

struct RetroEmulator {
    
    double getFrameRate() { 
        return avInfo.timing.fps; 
    }

    double getAudioRate() { 
        return avInfo.timing.sample_rate;
    }

	int getAudioSamples() { 
        return audioData.size() / 2; 
    }
	const int16_t* getAudioData() {
        return audioData.data(); 
    }

    py::array_t<int16_t> getAudio() {
		py::array_t<int16_t> arr(py::array::ShapeContainer{ getAudioSamples(), 2 });
		int16_t* data = arr.mutable_data();
		memcpy(data, getAudioData(), getAudioSamples() * 4);
		return arr;
	}

    void initCore(char *core, char *game) {
        init(core, game);
    }

    void runCore() {
        run();
    }

    void resetCore() {
        reset();
    }

    void closeCore() {
        kill();
    }

    bool setState(py::bytes o) {
        try {
            return g_retro.retro_unserialize(PyBytes_AsString(o.ptr()), PyBytes_Size(o.ptr()));
        } catch(...) {
            return false;
        }
		
	}

    py::bytes getState() {
		size_t size = get_state_size();
		py::bytes bytes(NULL, size);
		g_retro.retro_serialize(PyBytes_AsString(bytes.ptr()), size);
		return bytes;
	}

    py::array_t<uint8_t> getMemoryByType(unsigned type) {
        // Get memory pointer and size from core
        void* memory_data = g_retro.retro_get_memory_data(type);
        size_t memory_size = g_retro.retro_get_memory_size(type);
        
        if (!memory_data || memory_size == 0) {
            throw std::runtime_error("Invalid memory region or not available");
        }
        
        // Create a numpy array that references the memory without copying
        py::array_t<uint8_t> array(
            {memory_size},                            // shape
            {sizeof(uint8_t)},                        // strides
            static_cast<uint8_t*>(memory_data),       // data pointer
            py::capsule(memory_data, [](void* f) {})  // capsule (no deleter since we don't own the memory)
        );
        
        return array;
    }

    py::array_t<uint8_t> getRAM() {
        return getMemoryByType(RETRO_MEMORY_SYSTEM_RAM);
    }

    void getFrame(py::buffer buf, int width, int height) {
        py::buffer_info info = buf.request();

        uint8_t* buffer = static_cast<uint8_t*>(info.ptr);

        get_frame(buffer, width, height);
    }

    py::tuple getShape() {
        return py::make_tuple(g_retro.height, g_retro.width);
    }

    void setButtonMask(py::array_t<uint8_t> mask, unsigned player) {
		if (mask.size() > N_BUTTONS) {
			throw std::runtime_error("mask.size() > N_BUTTONS");
		}
		if (player >= MAX_PLAYERS) {
			throw std::runtime_error("player >= MAX_PLAYERS");
		}

		for (int key = 0; key < mask.size(); ++key) {
			setKey(player, key, mask.data()[key]);
		}
	}
};

PYBIND11_MODULE(_retro, m) {
    py::class_<RetroEmulator>(m, "RetroEmulator")
        .def(py::init<>())
        .def("run", &RetroEmulator::runCore)
        .def("reset", &RetroEmulator::resetCore)
        .def("set_button_mask", &RetroEmulator::setButtonMask, py::arg("mask"), py::arg("player")=0)
        .def("get_state", &RetroEmulator::getState)
        .def("set_state", &RetroEmulator::setState)
        .def("get_frame", &RetroEmulator::getFrame, py::arg("buffer"), py::arg("width"), py::arg("height"))
        .def("get_shape", &RetroEmulator::getShape)
        .def("get_ram", &RetroEmulator::getRAM)
        .def("close", &RetroEmulator::closeCore)
        .def("init", &RetroEmulator::initCore, py::arg("core"), py::arg("game"))
        .def("get_frame_rate", &RetroEmulator::getFrameRate)
        .def("get_audio_rate", &RetroEmulator::getAudioRate)
        .def("get_audio", &RetroEmulator::getAudio);
}