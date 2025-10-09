#pragma once

#include <SDL.h>
#include <vulkan/vulkan.h>
#include <libretro.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VULKAN_MAX_SWAPCHAIN_IMAGES 8
#define VULKAN_MAX_DESCRIPTOR_POOL_SIZES 16

struct vulkan_context {
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDevice device;
    VkQueue queue;
    uint32_t graphics_queue_index;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    uint32_t num_swapchain_images;
    uint32_t current_frame_index;
    uint32_t current_swapchain_index;
};

// Funções essenciais
bool vulkan_context_init(struct vulkan_context *ctx, SDL_Window* window);
void vulkan_context_destroy(struct vulkan_context *ctx);

#ifdef __cplusplus
}
#endif