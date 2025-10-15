#include <SDL.h>
#include <SDL_vulkan.h>
#include "libretro.h"
#include "libretro_vulkan.h"
#include <vulkan/vulkan.h>
#include <set>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vulkan/vulkan.h>
#include <libretro_vulkan.h>
#include "vulkan_common.h"
#include <vulkan/vulkan_symbol_wrapper.h>

#ifdef _WIN32
#include <windows.h>
#include <excpt.h>
#endif

#ifndef _WIN32
#define _strdup strdup
#endif

using namespace std;

static SDL_Window *g_win = NULL;
static SDL_AudioDeviceID g_pcm = 0;
static struct retro_frame_time_callback runloop_frame_time;
static retro_usec_t runloop_frame_time_last = 0;
static const uint8_t *g_kbd = NULL;
static struct retro_audio_callback audio_callback;
static bool g_context_reset = false;

// static struct retro_hw_render_interface_vulkan _g_render_iface = {};
static struct retro_hw_render_interface_vulkan g_render_iface = {};

static gfx_ctx_vulkan_data_t _vk = {};
static gfx_ctx_vulkan_data_t *g_vk = &_vk;

VkSurfaceKHR surface;
VkFormat depthFormat;//
VkImage depthImage;//
VkDeviceMemory depthImageMemory;//
VkImageView depthImageView;//
VkSurfaceFormatKHR surfaceFormat;
VkRenderPass render_pass;
vector<VkFramebuffer> swapchainFramebuffers;
vector<VkImageView> swapchainImageViews;
VkCommandPool commandPool;
vector<VkCommandBuffer> commandBuffers;
VkSemaphore imageAvailableSemaphore;
VkSemaphore renderingFinishedSemaphore;


static struct {
    struct retro_hw_render_callback hw;
} g_video  = {};

#ifdef _WIN32
#ifdef main
#undef main
#endif
#endif

static float g_scale = 3;
bool running = true;


static struct retro_variable *g_vars = NULL;


#define RETRO_DEVICE_WIIMOTE RETRO_DEVICE_JOYPAD
#define RETRO_DEVICE_WIIMOTE_SW ((2 << 8) | RETRO_DEVICE_JOYPAD)
#define RETRO_DEVICE_WIIMOTE_NC ((3 << 8) | RETRO_DEVICE_JOYPAD)
#define RETRO_DEVICE_WIIMOTE_CC ((4 << 8) | RETRO_DEVICE_JOYPAD)
#define RETRO_DEVICE_WIIMOTE_CC_PRO ((5 << 8) | RETRO_DEVICE_JOYPAD)
#define RETRO_DEVICE_GC_ON_WII ((6 << 8) | RETRO_DEVICE_JOYPAD)
#define RETRO_DEVICE_REAL_WIIMOTE ((6 << 8) | RETRO_DEVICE_NONE)


static struct {
	void *handle;
	bool initialized;
	bool supports_no_game;
	struct retro_perf_counter* perf_counter_last;
	void (*retro_init)(void);
	void (*retro_deinit)(void);
	unsigned (*retro_api_version)(void);
	void (*retro_get_system_info)(struct retro_system_info *info);
	void (*retro_get_system_av_info)(struct retro_system_av_info *info);
	void (*retro_set_controller_port_device)(unsigned port, unsigned device);
	void (*retro_reset)(void);
	void (*retro_run)(void);
	bool (*retro_load_game)(const struct retro_game_info *game);
	void (*retro_unload_game)(void);
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
	{ "pcsx2_renderer", "Vulkan" },
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
	{ "dolphin_renderer", "Hardware" },
	{ "dolphin_fastmem", "disabled" },
	{ "dolphin_dsp_hle", "enabled" },
	{ "dolphin_dsp_jit", "enabled" },
	{ "dolphin_cpu_core", "JIT64" },
	{ "dolphin_language", "English" },
	{ "dolphin_widescreen", "disabled" },
	{ "dolphin_widescreen_hack", "disabled" },
	{ "dolphin_progressive_scan", "disabled" },
	{ "dolphin_pal60", "disabled" },
	{ "dolphin_sensor_bar_position", "Bottom" },
	{ "dolphin_wiimote_continuous_scanning", "disabled" },
	{ "dolphin_mixer_rate", "32000" },
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
    { "desmume_opengl_mode", "disabled" },
    { "desmume_input_rotation", "180" },
    { "citra_is_new_3ds", "New 3DS" },
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


// vulkan render functions
void drawFrame(const void* frame_data, unsigned width, unsigned height, size_t pitch);

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

static void resize_cb(int w, int h) {
	// glViewport(0, 0, w, h);
}



static void create_window(int width, int height) {
    g_win = SDL_CreateWindow(
        "sdlarch", 
        SDL_WINDOWPOS_CENTERED, 
        SDL_WINDOWPOS_CENTERED, 
        width, 
        height, 
        SDL_WINDOW_VULKAN
    );

	if (!g_win)
        die("Failed to create window: %s", SDL_GetError());

    resize_cb(width, height);

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

	// nwidth *= g_scale;
	// nheight *= g_scale;
    // nwidth *= g_scale;
	// nheight *= g_scale;

	if (!g_win)
		create_window(nwidth, nheight);
    else
        SDL_SetWindowSize(g_win, nwidth, nheight);

}


static void video_refresh(const void *data, unsigned width, unsigned height, unsigned pitch) {
    printf("[FRAME] video_refresh called: %ux%u, pitch %u, data %p\n", width, height, pitch, data);
}

static void video_deinit() {

    SDL_DestroyWindow(g_win);
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

	// if (level == RETRO_LOG_ERROR)
	// 	exit(EXIT_FAILURE);
}

static uintptr_t core_get_current_framebuffer() {
    return 0;
}

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

static uint32_t vulkan_get_sync_index_mask(void *handle)
{
   printf("[VULKAN] vulkan_get_sync_index_mask called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
   return (1 << g_vk ->context.num_swapchain_images) - 1;
}

void vulkan_context_reset() {
    printf("[VULKAN] vulkan_context_reset() called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
}

static void vulkan_lock_queue(void *handle)
{
    printf("[VULKAN] vulkan_lock_queue called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
//    slock_lock(vk->context.queue_lock);
}

static void vulkan_unlock_queue(void *handle)
{
    printf("[VULKAN] vulkan_unlock_queue called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
//    slock_unlock(vk->context.queue_lock);
}

static void vulkan_wait_sync_index(void *handle) {
    printf("[VULKAN] vulkan_wait_sync_index called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
}

static uint32_t vulkan_get_sync_index(void *handle)
{
    printf("[VULKAN] vulkan_get_sync_index called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    return g_vk->context.current_frame_index;
}

// TODO implement and mabe work!!!!!!!!!!!!
static void vulkan_set_image(void *handle,
      const struct retro_vulkan_image *image,
      uint32_t num_semaphores,
      const VkSemaphore *semaphores,
      uint32_t src_queue_family)
{
   vk_t *vk              = (vk_t*)handle;

   printf("[VULKAN] vulkan_set_image called  ---------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");

   printf("[VULKAN] image: %p, num_semaphores: %u, semaphores: %p, src_queue_family: %u\n",
         (void*)image, num_semaphores, (void*)semaphores, src_queue_family);
    
   printf("[VULKAN] vulkan_set_image vk: %p \n", vk);
   printf("[VULKAN] vulkan_set_image vk->hw: %p \n", vk->hw);
   printf("[VULKAN] vulkan_set_image image: %p \n", image);

//    TODO: fix segmentation fault
//    vk->hw.image          = image;
//    vk->hw.num_semaphores = num_semaphores;

//    if (num_semaphores > 0)
//    {
//       int i;

//       /* Allocate one extra in case we need to use WSI acquire semaphores. */
//       VkPipelineStageFlags *stage_flags = (VkPipelineStageFlags*)realloc(vk->hw.wait_dst_stages,
//             sizeof(VkPipelineStageFlags) * (vk->hw.num_semaphores + 1));

//       VkSemaphore *new_semaphores = (VkSemaphore*)realloc(vk->hw.semaphores,
//             sizeof(VkSemaphore) * (vk->hw.num_semaphores + 1));

//       vk->hw.wait_dst_stages = stage_flags;
//       vk->hw.semaphores      = new_semaphores;

//       for (i = 0; i < (int) vk->hw.num_semaphores; i++)
//       {
//          vk->hw.wait_dst_stages[i] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
//          vk->hw.semaphores[i]      = semaphores[i];
//       }

//       vk->flags                   |= VK_FLAG_HW_VALID_SEMAPHORE;
//       vk->hw.src_queue_family      = src_queue_family;
//    }
}

// TODO implement and mabe work!!!!!!!!!!!!
static void vulkan_init_textures(vk_t *vk)
{
   const uint32_t zero = 0;

   if (!(vk->flags & VK_FLAG_HW_ENABLE))
   {
      int i;
      for (i = 0; i < (int) vk->num_swapchain_images; i++)
      {
         vk->swapchain[i].texture = vulkan_create_texture(
               vk, NULL, vk->tex_w, vk->tex_h, vk->tex_fmt,
               NULL, NULL, VULKAN_TEXTURE_STREAMED);

         {
            struct vk_texture *texture = &vk->swapchain[i].texture;
            VK_MAP_PERSISTENT_TEXTURE(vk->context->device, texture);
         }

         if (vk->swapchain[i].texture.type == VULKAN_TEXTURE_STAGING)
            vk->swapchain[i].texture_optimal = vulkan_create_texture(
                  vk, NULL, vk->tex_w, vk->tex_h, vk->tex_fmt,
                  NULL, NULL, VULKAN_TEXTURE_DYNAMIC);
      }
   }

   vk->default_texture = vulkan_create_texture(vk, NULL,
         1, 1, VK_FORMAT_B8G8R8A8_UNORM,
         &zero, NULL, VULKAN_TEXTURE_STATIC);
}

static bool core_environment(unsigned cmd, void *data) {
	switch (cmd) {
    case RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE:
        return true;
    // case RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES: {
    //     uint64_t* caps = (uint64_t*)data;
    //     *caps = (1 << RETRO_DEVICE_JOYPAD);
    //     return true;
    // }

    case RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS: {
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
                outvar->value = ((const char*)malloc((first_pipe - semicolon) + 1));
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
            if(!strcmp(outvar->key, "parallel-n64-rspplugin")) {
                free((void *)outvar->value);
                outvar->value = strdup("glide64");
            }

            // pcsx2_enable_rumble
            if(!strcmp(outvar->key, "pcsx2_enable_rumble1")) {
                free((void *)outvar->value);
                outvar->value = _strdup("disabled");
            }
            if(!strcmp(outvar->key, "pcsx2_button_deadzone1")) {
                free((void *)outvar->value);
                outvar->value = _strdup("0%");
            }

            if (key_exists(outvar->key)) {
                for (int i = 0; s_envVariables[i].key != NULL; i++) {
                    if (strcmp(s_envVariables[i].key, outvar->key) == 0) {
                        outvar->value = _strdup((const char*)s_envVariables[i].value);
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

            // if(!strcmp(var->key, "desmume_input_rotation")) {
            //     free(var->value);
            //     var->value = 180;
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
		return true;
	}

    // already true, dont need to change
    case RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER: {
        unsigned* context_type = (unsigned*)data;
        *context_type = RETRO_HW_CONTEXT_VULKAN;
        printf("[ENV] RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER RETRO_HW_CONTEXT_VULKAN\n");
        return true;
    }
    // TODO: implement here!!
    case RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE: {
        
        struct retro_hw_render_interface **iface = (struct retro_hw_render_interface **)data;
        printf("[ENV] RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE - Vulkan version\n");
        
       
        *iface = (retro_hw_render_interface*)&g_render_iface;

        // vulkan_init_hw_render(vk, &g_video.hw, &g_render_iface);

        printf("[ENV] vulkan_init_hw_render done, interface_type: %d\n", g_render_iface.interface_type);
        printf("[ENV] vulkan_init_hw_render done, interface_type: %d\n", g_render_iface.interface_version);
        printf("[ENV] vulkan_init_hw_render done, interface_type: %d\n", ((retro_hw_render_interface_vulkan*)*iface) -> interface_type);
        printf("[ENV] vulkan_init_hw_render done, interface_type: %d\n", ((retro_hw_render_interface_vulkan*)*iface) -> interface_version);

        
        return true;
    }

    // already true, dont need to change
    case RETRO_ENVIRONMENT_GET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_SUPPORT:
    {
        printf("[ENV] RETRO_ENVIRONMENT_GET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_SUPPORT\n");
        struct retro_hw_render_context_negotiation_interface *iface =
                (struct retro_hw_render_context_negotiation_interface*)data;

        
        iface->interface_version = RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION;
        return true;
    }

    // TODO Implement this
    // TODO mabe call static bool vulkan_context_init_device(gfx_ctx_vulkan_data_t *vk) in vulkan_driver.c
    case RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE: {
        printf("[ENV] RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE\n");
        // video_driver_state_t *video_st = video_state_get_ptr();
        struct retro_hw_render_context_negotiation_interface_vulkan* iface = 
            (struct retro_hw_render_context_negotiation_interface_vulkan*)data;

        if (!iface) {
            printf("[ENV] ERROR: Null pointer in SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE\n");
            return false;
        }

        // What to do here?
        g_iface = iface;
#ifdef _WIN32
        vulkan_context_init(g_vk, VULKAN_WSI_WIN32);
#else
        vulkan_context_init(g_vk, VULKAN_WSI_XLIB);
#endif
        SDL_Vulkan_CreateSurface(g_win, g_vk->context.instance, &surface);
        g_vk->vk_surface = surface;
        vulkan_context_init_device(g_vk);
        vulkan_create_swapchain(g_vk, 640, 480, 1);
        vk_t _vk = {};
        _vk.context = &g_vk->context;
        _vk.num_swapchain_images = g_vk->context.num_swapchain_images;
        _vk.hw = {};
        // _vk.swapchain = g_vk->context.swapchain;
        // vulkan_init_textures(&_vk);

        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = vulkan_symbol_wrapper_instance_proc_addr();

        printf("[ENV] vulkan_symbol_wrapper_instance_proc_addr: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> %p\n", (void*)vkGetInstanceProcAddr);

        g_render_iface.interface_type         = RETRO_HW_RENDER_INTERFACE_VULKAN;
        g_render_iface.interface_version      = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION;
        g_render_iface.instance               = g_vk->context.instance; // vk->context->instance;
        g_render_iface.gpu                    = g_vk->context.gpu;
        g_render_iface.device                 = g_vk->context.device;

        g_render_iface.queue                  = g_vk->context.queue;
        g_render_iface.queue_index            = g_vk->context.graphics_queue_index;

        g_render_iface.handle                 = g_vk;
        g_render_iface.get_sync_index_mask = vulkan_get_sync_index_mask;
        g_render_iface.lock_queue             = vulkan_lock_queue;
        g_render_iface.unlock_queue           = vulkan_unlock_queue;
        g_render_iface.wait_sync_index = vulkan_wait_sync_index;
        g_render_iface.get_sync_index         = vulkan_get_sync_index;
        g_render_iface.set_image              = vulkan_set_image;
        g_render_iface.get_device_proc_addr   = vkGetDeviceProcAddr;
        g_render_iface.get_instance_proc_addr = vkGetInstanceProcAddr;

        g_video.hw.context_reset();

        printf("[ENV] Stored Vulkan negotiation interface: %p\n", (void*)g_iface);


        return true;
    }

    // TODO Implement this
    case RETRO_ENVIRONMENT_SET_HW_RENDER:
    case RETRO_ENVIRONMENT_SET_HW_RENDER | RETRO_ENVIRONMENT_EXPERIMENTAL: {
        struct retro_hw_render_callback *hw = (struct retro_hw_render_callback*)data;

        if(hw->context_type == RETRO_HW_CONTEXT_VULKAN) {
            printf("[ENV] RETRO_ENVIRONMENT_SET_HW_RENDER RETRO_HW_CONTEXT_VULKAN\n");
        } else {
            printf("[ENV] RETRO_ENVIRONMENT_SET_HW_RENDER received - context_type: %d\n", hw->context_type);
        }
        
        // What to do here?
        // if (hw->context_type == RETRO_HW_CONTEXT_VULKAN) {
        //     hw->context_reset = vulkan_context_reset;
        //     hw->context_destroy = vulkan_context_destroy;
        //     hw->bottom_left_origin = true;
        // }

        // hw->context_reset = vulkan_context_reset;
        
        g_video.hw = *hw;
        return true;
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
        *dir = "./system";   // BIOS, flash, assets
        return true;
    }
    case RETRO_ENVIRONMENT_SET_GEOMETRY: {
        struct retro_game_geometry *geom = (struct retro_game_geometry*)data;
        printf("[ENV] RETRO_ENVIRONMENT_SET_GEOMETRY ------------------------------>>>>>>>>>>> \n");
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

static void noop() {}

int main(int argc, char *argv[]) {
	if (argc < 2)
		die("usage: %s <core> [game]", argv[0]);

    if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO|SDL_INIT_EVENTS) < 0)
        die("Failed to initialize SDL");

    g_video.hw.context_type = RETRO_HW_CONTEXT_VULKAN;
    // g_video.hw.context_reset = vulkan_context_reset;
    // g_video.hw.context_destroy = vulkan_context_destroy;
    // g_video.hw.version_major = VK_API_VERSION_1_0;
    // g_video.hw.version_minor = 0;

    create_window(640, 480);
    

#ifdef _WIN32
    vulkan_context_init(g_vk, VULKAN_WSI_WIN32);
#else
    vulkan_context_init(g_vk, VULKAN_WSI_XLIB);
#endif




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

    int frame_count = 0;
    while (running) {
        printf("[FRAME %d] === START FRAME ===\n", frame_count);
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    break;
                case SDL_KEYDOWN:
                    break;
            }
        }
        
#ifdef _WIN32
        __try {
#endif
            printf("current_frame_index %d\n", g_vk->context.current_frame_index);
            printf("retro_run ----> vkGetInstanceProcAddr: %p\n", (void*)vulkan_symbol_wrapper_instance_proc_addr());
            g_retro.retro_run();
            printf("[FRAME %d] retro_run completed\n", frame_count);
#ifdef _WIN32
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            printf("[FRAME %d] Exception retro_run: 0x%08X\n", 
                frame_count, GetExceptionCode());
            break;
        }
#endif
        
        printf("[FRAME %d] === END FRAME ===\n", frame_count);
        frame_count++;
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
