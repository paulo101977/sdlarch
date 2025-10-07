#include <SDL.h>
#include "libretro.h"
#include "glad.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef HAS_VULKAN
// #include <SDL_vulkan.h>
#include "libretro_vulkan.h"
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>

#define RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION 5

struct retro_vulkan_context;

struct retro_hw_render_interface_vulkan {
    struct retro_hw_render_interface interface;
    void* handle;
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDevice device;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    VkQueue queue;
    unsigned queue_index;
    retro_vulkan_set_image_t set_image;
    retro_vulkan_get_sync_index_t get_sync_index;
    retro_vulkan_get_sync_index_mask_t get_sync_index_mask;
    retro_vulkan_set_command_buffers_t set_command_buffers;
    retro_vulkan_wait_sync_index_t wait_sync_index;
    retro_vulkan_lock_queue_t lock_queue;
    retro_vulkan_unlock_queue_t unlock_queue;
    retro_vulkan_set_signal_semaphore_t set_signal_semaphore;
};

static struct retro_hw_render_interface_vulkan g_vk_interface = {
    .interface = {
        .interface_type = RETRO_HW_RENDER_INTERFACE_VULKAN,
        .interface_version = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION,
    },
    .instance = VK_NULL_HANDLE,
    .gpu = VK_NULL_HANDLE,
    .device = VK_NULL_HANDLE,
    .queue = VK_NULL_HANDLE,
    .queue_index = 0,
    .get_instance_proc_addr = NULL,
    .get_device_proc_addr = NULL,
    .set_image = NULL,
    .get_sync_index = NULL,
    .get_sync_index_mask = NULL,
    .set_command_buffers = NULL,
    .wait_sync_index = NULL,
    .lock_queue = NULL,
    .unlock_queue = NULL,
    .set_signal_semaphore = NULL
};
#endif

#ifndef _WIN32
    #define _strdup strdup
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
#endif

static float g_scale = 3;
bool running = true;

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

#ifdef HAS_VULKAN
static VkInstance g_vk_instance = VK_NULL_HANDLE;
static VkSurfaceKHR g_vk_surface = VK_NULL_HANDLE;
static VkPhysicalDevice g_vk_physical_device = VK_NULL_HANDLE;
static VkDevice g_vk_device = VK_NULL_HANDLE;
static VkQueue g_vk_queue = VK_NULL_HANDLE;
static uint32_t g_vk_queue_family = 0;
static int g_vk_initialized = 0;
static int g_vk_failed = 0;


static bool create_device(struct retro_vulkan_context* context, VkInstance instance, VkPhysicalDevice gpu,
                         VkSurfaceKHR surface, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                         const char** required_device_extensions, unsigned num_required_device_extensions,
                         const char** required_device_layers, unsigned num_required_device_layers,
                         const VkPhysicalDeviceFeatures* required_features) {
    
    printf("Core requesting Vulkan device creation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    printf("  Instance: %p\n", instance);
    printf("  Physical Device: %p\n", gpu);
    printf("  Surface: %p\n", surface);
    printf("  Required device extensions: %u\n", num_required_device_extensions);
    
    for (unsigned i = 0; i < num_required_device_extensions; i++) {
        printf("    - %s\n", required_device_extensions[i]);
    }
    
    if (gpu == VK_NULL_HANDLE) {
        gpu = g_vk_physical_device;
        printf("Using pre-selected physical device: %p\n", gpu);
    }
    
    float queue_priority = 1.0f;
    
    VkDeviceQueueCreateInfo queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g_vk_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    
    const char* base_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    size_t total_extensions = 1 + num_required_device_extensions;
    const char** all_extensions = SDL_malloc(sizeof(const char*) * total_extensions);
    
    if (!all_extensions) {
        printf("Failed to allocate device extensions array\n");
        return false;
    }
    
    all_extensions[0] = base_extensions[0];
    
    for (unsigned i = 0; i < num_required_device_extensions; i++) {
        all_extensions[1 + i] = required_device_extensions[i];
        printf("Adding required extension: %s\n", required_device_extensions[i]);
    }
    
    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledExtensionCount = (uint32_t)total_extensions,
        .ppEnabledExtensionNames = all_extensions,
        .pEnabledFeatures = required_features
    };
    
    VkResult result = vkCreateDevice(gpu, &device_create_info, NULL, &g_vk_device);
    SDL_free(all_extensions);
    
    if (result != VK_SUCCESS) {
        printf("Failed to create Vulkan logical device: %d\n", result);
        return false;
    }
    
    vkGetDeviceQueue(g_vk_device, g_vk_queue_family, 0, &g_vk_queue);
    
    context->gpu = gpu;
    context->device = g_vk_device;
    context->queue = g_vk_queue;
    context->queue_family_index = g_vk_queue_family;
    context->presentation_queue = g_vk_queue;
    context->presentation_queue_family_index = g_vk_queue_family;
    
    g_vk_interface.instance = instance;
    g_vk_interface.gpu = gpu;
    g_vk_interface.device = g_vk_device;
    g_vk_interface.queue = g_vk_queue;
    g_vk_interface.queue_index = g_vk_queue_family;
    g_vk_interface.get_instance_proc_addr = get_instance_proc_addr;
    g_vk_interface.get_device_proc_addr = vkGetDeviceProcAddr;
    
    printf("Vulkan device created successfully for core\n");
    printf("  Device: %p\n", g_vk_device);
    printf("  Queue: %p\n", g_vk_queue);
    printf("  Queue Family: %u\n", g_vk_queue_family);
    
    return true;
}

static const VkApplicationInfo* get_application_info(void) {
    static VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "sdlarch",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "sdlarch",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    printf("Providing application info to core\n");

    return &app_info;
}

static void cleanup_vulkan() {
    printf("Cleaning up Vulkan resources...\n");
    
    if (g_vk_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(g_vk_device);
        vkDestroyDevice(g_vk_device, NULL);
        g_vk_device = VK_NULL_HANDLE;
    }
    
    if (g_vk_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(g_vk_instance, g_vk_surface, NULL);
        g_vk_surface = VK_NULL_HANDLE;
    }
    
    if (g_vk_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_vk_instance, NULL);
        g_vk_instance = VK_NULL_HANDLE;
    }
    
    g_vk_physical_device = VK_NULL_HANDLE;
    g_vk_queue = VK_NULL_HANDLE;
    g_vk_queue_family = 0;
    g_vk_initialized = 0;
    g_vk_failed = 0;
    printf("Vulkan resources cleaned up\n");
}

static VkInstance create_vulkan_instance() {
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "sdlarch",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "sdlarch",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    unsigned int extension_count = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(g_win, &extension_count, NULL)) {
        printf("Failed to get Vulkan instance extension count: %s\n", SDL_GetError());
        return VK_NULL_HANDLE;
    }

    const char **extensions = SDL_malloc(sizeof(const char *) * (extension_count + 1));
    if (!extensions) {
        printf("Failed to allocate extensions array\n");
        return VK_NULL_HANDLE;
    }

    if (!SDL_Vulkan_GetInstanceExtensions(g_win, &extension_count, extensions)) {
        printf("Failed to get Vulkan instance extensions: %s\n", SDL_GetError());
        SDL_free(extensions);
        return VK_NULL_HANDLE;
    }

    const char *required_extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
    };

    const char **all_extensions = SDL_malloc(sizeof(const char *) * (extension_count + 4));
    for (unsigned int i = 0; i < extension_count; i++) {
        all_extensions[i] = extensions[i];
    }
    for (int i = 0; i < 4; i++) {
        all_extensions[extension_count + i] = required_extensions[i];
    }

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = extension_count + 4,
        .ppEnabledExtensionNames = all_extensions,
        .enabledLayerCount = 0
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&create_info, NULL, &instance);

    SDL_free(extensions);
    SDL_free(all_extensions);

    if (result != VK_SUCCESS) {
        printf("Failed to create Vulkan instance: %d\n", result);
        return VK_NULL_HANDLE;
    }

    printf("Vulkan instance created successfully\n");
    return instance;
}

static int create_vulkan_surface() {
    if (!SDL_Vulkan_CreateSurface(g_win, g_vk_instance, &g_vk_surface)) {
        printf("Failed to create Vulkan surface: %s\n", SDL_GetError());
        return 0;
    }
    printf("Vulkan surface created successfully\n");
    return 1;
}

static int select_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(g_vk_instance, &device_count, NULL);
    
    if (device_count == 0) {
        printf("No Vulkan physical devices found\n");
        return 0;
    }
    
    VkPhysicalDevice* devices = SDL_malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(g_vk_instance, &device_count, devices);
    
    // Selecionar o primeiro dispositivo adequado
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        
        printf("Found Vulkan device: %s\n", props.deviceName);
        
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, NULL);
        
        VkQueueFamilyProperties* queue_families = SDL_malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, queue_families);
        
        for (uint32_t j = 0; j < queue_family_count; j++) {
            if (queue_families[j].queueCount > 0 && 
                (queue_families[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                
                VkBool32 present_support = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], j, g_vk_surface, &present_support);
                
                if (present_support) {
                    g_vk_physical_device = devices[i];
                    g_vk_queue_family = j;
                    SDL_free(queue_families);
                    SDL_free(devices);
                    printf("Selected Vulkan device: %s (queue family %u)\n", props.deviceName, j);
                    return 1;
                }
            }
        }
        
        SDL_free(queue_families);
    }
    
    SDL_free(devices);
    printf("No suitable Vulkan device found\n");
    return 0;
}

static const struct retro_hw_render_context_negotiation_interface_vulkan g_vk_negotiation_interface = {
    RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN,
    RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION,
    get_application_info, // get_application_info
    create_device, // create_device
    NULL, // get_physical_device_extensions
};

static void init_vulkan_interface() {
    printf("Initializing Vulkan interface...\n");
    
    if (g_vk_failed) {
        printf("Vulkan previously failed, skipping\n");
        return;
    }
    
    if (g_vk_instance == VK_NULL_HANDLE) {
        g_vk_instance = create_vulkan_instance();
        if (g_vk_instance == VK_NULL_HANDLE) {
            printf("Failed to create Vulkan instance\n");
            g_vk_failed = 1;
            return;
        }
    }
    
    if (g_vk_surface == VK_NULL_HANDLE) {
        if (!create_vulkan_surface()) {
            printf("Failed to create Vulkan surface\n");
            g_vk_failed = 1;
            return;
        }
    }
    
    if (g_vk_physical_device == VK_NULL_HANDLE) {
        if (!select_physical_device()) {
            printf("Failed to select Vulkan physical device\n");
            g_vk_failed = 1;
            return;
        }
    }
    
    
    g_vk_interface.instance = g_vk_instance;
    g_vk_interface.gpu = g_vk_physical_device;
    g_vk_interface.get_instance_proc_addr = vkGetInstanceProcAddr;
    g_vk_interface.get_device_proc_addr = vkGetDeviceProcAddr;
    
    g_vk_interface.device = VK_NULL_HANDLE;
    g_vk_interface.queue = VK_NULL_HANDLE;
    g_vk_interface.queue_index = g_vk_queue_family;
    
    g_vk_initialized = 1;
    printf("Vulkan interface initialized successfully (waiting for core to create device)\n");
}

#endif

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
	{ "dolphin_efb_scaled_copy", "disabled" },
	{ "dolphin_efb_to_texture", "disabled" },
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

    // TODO: make the same in sdlarch-rl
    if (g_video.hw.context_reset) {
        g_video.hw.context_reset();
    }
}


static void create_vulkan_window(int width, int height) {
    printf("Creating Vulkan window...\n");
    
    g_win = SDL_CreateWindow("sdlarch-vulkan",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!g_win) {
        die("Failed to create Vulkan window: %s", SDL_GetError());
    }
    
    printf("Vulkan window created successfully\n");

#ifdef HAS_VULKAN
    init_vulkan_interface();
#endif
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
#ifndef HAS_VULKAN
		create_window(nwidth, nheight);
#else
        create_vulkan_window(nwidth, nheight); 
#endif

#ifndef HAS_VULKAN
	if (g_video.tex_id)
		glDeleteTextures(1, &g_video.tex_id);

	g_video.tex_id = 0;

	if (!g_video.pixfmt)
		g_video.pixfmt = GL_UNSIGNED_SHORT_5_5_5_1;
#endif

    SDL_SetWindowSize(g_win, nwidth, nheight);

#ifndef HAS_VULKAN

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
     printf("Vulkan configuration completed - core handles everything\n");
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
    return g_video.fbo_id;
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

static bool core_environment(unsigned cmd, void *data) {

    printf("cmd >>>>>>> %d \n", cmd);

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

    case RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO: {
        struct retro_system_av_info* av_info = (struct retro_system_av_info*)data;
        printf("AV_INFO: %dx%d @ %.2f FPS >>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n", 
               av_info->geometry.base_width, av_info->geometry.base_height,
               av_info->timing.fps);
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

#ifdef HAS_VULKAN
    case RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER: {
        unsigned* context_type = (unsigned*)data;
        *context_type = RETRO_HW_CONTEXT_VULKAN;
        printf("Preferring Vulkan\n");
        return true;
    }

    case RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE: {
        struct retro_hw_render_interface** interface = (struct retro_hw_render_interface**)data;
        
        printf("Core requesting Vulkan hardware render interface\n");
        
        if (!g_vk_initialized && !g_vk_failed) {
            printf("Vulkan not initialized yet, initializing now...\n");
            init_vulkan_interface();
        }
        
        if (g_vk_initialized && interface) {
            *interface = (struct retro_hw_render_interface*)&g_vk_interface;
            printf("Vulkan interface provided to core successfully\n");
            return true;
        } else {
            printf("Cannot provide Vulkan interface (initialized: %d, failed: %d)\n", 
                   g_vk_initialized, g_vk_failed);
            return false;
        }
    }

    case RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE: {
        const struct retro_hw_render_context_negotiation_interface* iface = 
            (const struct retro_hw_render_context_negotiation_interface*)data;
        
        printf("Core setting hardware render context negotiation interface\n");
        printf("  Interface type: %u\n", iface->interface_type);
        printf("  Interface version: %u\n", iface->interface_version);
        
        if (iface->interface_type == RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN) {
            const struct retro_hw_render_context_negotiation_interface_vulkan* vk_iface =
                (const struct retro_hw_render_context_negotiation_interface_vulkan*)iface;
            
            printf("Vulkan negotiation interface provided by core\n");
            
            if (vk_iface->get_application_info) {
                printf("Core provided get_application_info - will use core's function\n");
            } else {
                printf("Core did not provide get_application_info - will use ours\n");
            }
            
            if (vk_iface->create_device) {
                printf("Core provided create_device - will use core's function\n");
            } else {
                printf("Core did not provide create_device - will use ours\n");
            }
            
            return true;
        }
        
        return false;
    }
#endif

    case RETRO_ENVIRONMENT_SET_HW_RENDER: {
        struct retro_hw_render_callback *hw = (struct retro_hw_render_callback*)data;
#ifdef HAS_VULKAN
        if (hw->context_type == RETRO_HW_CONTEXT_VULKAN) {
            printf("Core configured for Vulkan rendering\n");
            
            if (!g_vk_initialized && !g_vk_failed) {
                init_vulkan_interface();
            }

            
            if (g_vk_initialized) {
                printf("Vulkan ready for core\n");
            } else {
                printf("WARNING: Vulkan not ready but core requested it\n");
            }
            
            return true;
        }
#else
        hw->get_current_framebuffer = core_get_current_framebuffer;
        hw->get_proc_address = (retro_hw_get_proc_address_t)SDL_GL_GetProcAddress;
        g_video.hw = *hw;
#endif
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
#ifndef HAS_VULKAN
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

#ifdef HAS_VULKAN
static void video_refresh_vulkan(const void* data, unsigned width, unsigned height, unsigned pitch) {
    static int frame_count = 0;
    // printf("Vulkan frame %d - core handles everything\n", frame_count++);
    if (!g_vk_initialized) {
        // Se Vulkan não foi inicializado pelo core, apenas contamos os frames
        printf("Vulkan frame %d - waiting for core initialization\n", frame_count++);
        return;
    }

    if (data == RETRO_HW_FRAME_BUFFER_VALID) {
        printf("HARDWARE FRAME - %ux%u\n", width, height);
    } 
    else if (data && data != RETRO_HW_FRAME_BUFFER_VALID) {
        printf("SOFTWARE FRAME - %ux%u, pitch: %u\n", width, height, pitch);
        
        if (width > 0 && height > 0) {
            printf(">>> FIRST SOFTWARE FRAME RECEIVED! <<<\n");
            // process_software_frame(data, width, height, pitch);
        }
    }
    else {
        printf("NULL frame\n");
    }

    SDL_Event redraw_event;
    redraw_event.type = SDL_WINDOWEVENT;
    redraw_event.window.event = SDL_WINDOWEVENT_EXPOSED;
    SDL_PushEvent(&redraw_event);
}
#endif


static void core_video_refresh(const void *data, unsigned width, unsigned height, size_t pitch) {
static int frame_count = 0;
#ifdef HAS_VULKAN
    video_refresh_vulkan(data, width, height, pitch);
#else
    video_refresh(data, width, height, pitch);
#endif
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

#ifdef HAS_VULKAN
    create_vulkan_window(640, 480);
#else

    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
    SDL_SetHint(SDL_HINT_RENDER_OPENGL_SHADERS, "1");
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0"); // Nearest neighbor
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");

    system("rm -rf ./system/User");

    g_video.hw.version_major = 4;
    g_video.hw.version_minor = 5;
    g_video.hw.context_type  = RETRO_HW_CONTEXT_OPENGL_CORE;
    // g_video.hw.context_type = RETRO_HW_CONTEXT_NONE;
    g_video.hw.context_reset   = noop;
    g_video.hw.context_destroy = noop;
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
                    // resize_cb(ev.window.data1, ev.window.data2);
                    break;
                case SDL_WINDOWEVENT_EXPOSED:
                    // Redesenhar a janela quando necessário
                    printf("Window exposed, forcing redraw\n");
                    break;
                }
            }
        }

#ifndef HAS_VULKAN
        SDL_GL_MakeCurrent(g_win, g_ctx);
#endif
        // glBindFramebuffer(GL_FRAMEBUFFER, 0);
        printf("before retro_run\n");
		g_retro.retro_run();
        printf("after retro_run\n");
	}

	core_unload();
	audio_deinit();
#ifdef HAS_VULKAN
    cleanup_vulkan();
#else
    video_deinit();
#endif

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