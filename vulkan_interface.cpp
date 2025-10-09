#include "vulkan_interface.h"
#include "vulkan_common.h"
#include <stdio.h>

// static struct retro_hw_render_interface_vulkan g_vulkan_iface;

// Callbacks - use as assinaturas EXATAS do libretro_vulkan.h
static void set_image(void* handle, const struct retro_vulkan_image* image,
    uint32_t num_semaphores, const VkSemaphore* semaphores, uint32_t src_queue_family) {
    printf("[VULKAN] set_image chamado - image: %p, semaphores: %u\n", 
           image, num_semaphores);
}

static unsigned get_sync_index(void* handle) {
    // Para simplificar, retorne 0
    return 0;
}

static uint32_t get_sync_index_mask(void* handle) {
    // Máscara simples - assumindo 2 imagens no swapchain
    return 0x3; // binary: 11
}

static void wait_sync_index(void* handle) {
    printf("[VULKAN] wait_sync_index chamado\n");
}

static void lock_queue(void* handle) {
    printf("[VULKAN] lock_queue chamado\n");
}

static void unlock_queue(void* handle) {
    printf("[VULKAN] unlock_queue chamado\n");
}

static void set_command_buffers(void* handle, const VkCommandBuffer* cmd, unsigned count) {
    printf("[VULKAN] set_command_buffers: %u buffers\n", count);
    for (unsigned i = 0; i < count; i++) {
        printf("  Command buffer %u: %p\n", i, cmd[i]);
    }
}

static void set_signal_semaphore(void* handle, VkSemaphore semaphore) {
    printf("[VULKAN] set_signal_semaphore: %p\n", semaphore);
}

// Inicializa a interface Vulkan para libretro
void vulkan_interface_init(struct vulkan_context* ctx) {
    if (!ctx) {
        printf("[VULKAN] ERRO: Contexto Vulkan não inicializado\n");
        return;
    }

    // Zera a estrutura primeiro
    memset(&g_vulkan_iface, 0, sizeof(g_vulkan_iface));
    
    // Preenche a interface Vulkan
    g_vulkan_iface.interface_type = RETRO_HW_RENDER_INTERFACE_VULKAN;
    g_vulkan_iface.interface_version = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION;
    g_vulkan_iface.instance = ctx->instance;
    g_vulkan_iface.gpu = ctx->gpu;
    g_vulkan_iface.device = ctx->device;
    g_vulkan_iface.queue = ctx->queue;
    g_vulkan_iface.queue_index = ctx->graphics_queue_index;
    
    // ⚠️ USE OS TIPOS JÁ DEFINIDOS - SEM CAST NECESSÁRIO
    g_vulkan_iface.set_image = set_image;
    g_vulkan_iface.get_sync_index = get_sync_index;
    g_vulkan_iface.get_sync_index_mask = get_sync_index_mask;
    g_vulkan_iface.wait_sync_index = wait_sync_index;
    g_vulkan_iface.lock_queue = lock_queue;
    g_vulkan_iface.unlock_queue = unlock_queue;
    g_vulkan_iface.set_command_buffers = (retro_vulkan_set_command_buffers_t)set_command_buffers;
    g_vulkan_iface.set_signal_semaphore = set_signal_semaphore;
    
    g_vulkan_iface.get_device_proc_addr = vkGetDeviceProcAddr;
    g_vulkan_iface.get_instance_proc_addr = vkGetInstanceProcAddr;
    
    printf("[VULKAN] Interface Vulkan inicializada:\n");
    printf("  Instance: %p\n", g_vulkan_iface.instance);
    printf("  Device: %p\n", g_vulkan_iface.device);
    printf("  Physical Device: %p\n", g_vulkan_iface.gpu);
    printf("  Queue: %p\n", g_vulkan_iface.queue);
}

struct retro_hw_render_interface_vulkan* vulkan_interface_get(void) {
    return &g_vulkan_iface;
}