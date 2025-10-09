#pragma once

#include <vulkan/vulkan.h>
#include <libretro.h>
#include <libretro_vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vulkan_context;

static struct retro_hw_render_interface_vulkan g_vulkan_iface;

// typedef void (*retro_vulkan_set_image_t)(void* handle,
//     const struct retro_vulkan_image* image,
//     uint32_t num_semaphores, const VkSemaphore* semaphores,
//     uint32_t src_queue_family);

// typedef unsigned (*retro_vulkan_get_sync_index_t)(void* handle);
// typedef uint32_t (*retro_vulkan_get_sync_index_mask_t)(void* handle);
// typedef void (*retro_vulkan_wait_sync_index_t)(void* handle);
// typedef void (*retro_vulkan_lock_queue_t)(void* handle);
// typedef void (*retro_vulkan_unlock_queue_t)(void* handle);
// typedef void (*retro_vulkan_set_command_buffers_t)(void* handle,
//     const VkCommandBuffer* cmd, unsigned count);
// typedef void (*retro_vulkan_set_signal_semaphore_t)(void* handle,
//     VkSemaphore semaphore);


void vulkan_interface_init(struct vulkan_context* ctx);
struct retro_hw_render_interface_vulkan* vulkan_interface_get(void);

#ifdef __cplusplus
}
#endif