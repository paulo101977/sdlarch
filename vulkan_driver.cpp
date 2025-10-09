#include "vulkan_common.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <vector>

bool vulkan_context_init(struct vulkan_context *ctx, SDL_Window *window) {
    printf("[VULKAN] Inicializando contexto Vulkan...\n");

    unsigned int extensionCount = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
    std::vector<const char *> extensions(extensionCount);
    SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data());
    

    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "sdlarch";
    app_info.apiVersion = VK_API_VERSION_1_0;
    
    // const char* extensions[] = {
    //     VK_KHR_SURFACE_EXTENSION_NAME,
    //     VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    //     VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    // };
    
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    inst_info.pApplicationInfo = &app_info;
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    // inst_info.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
    // inst_info.ppEnabledExtensionNames = extensions;
    inst_info.enabledExtensionCount = extensions.size();
    inst_info.ppEnabledExtensionNames = extensions.data();
    
    if (vkCreateInstance(&inst_info, NULL, &ctx->instance) != VK_SUCCESS) {
        printf("[VULKAN] ERRO: Falha ao criar instância\n");
        return false;
    }
    
    if (!SDL_Vulkan_CreateSurface(window, ctx->instance, &ctx->surface)) {
        printf("[VULKAN] ERRO: Falha ao criar surface\n");
        return false;
    }
    

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &gpu_count, NULL);
    VkPhysicalDevice gpus[16];
    vkEnumeratePhysicalDevices(ctx->instance, &gpu_count, gpus);
    ctx->gpu = gpus[0];
    

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    
    const char* device_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    
    VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = sizeof(device_extensions) / sizeof(device_extensions[0]);
    device_info.ppEnabledExtensionNames = device_extensions;
    
    if (vkCreateDevice(ctx->gpu, &device_info, NULL, &ctx->device) != VK_SUCCESS) {
        printf("[VULKAN] ERRO: Falha ao criar device\n");
        return false;
    }
    

    vkGetDeviceQueue(ctx->device, 0, 0, &ctx->queue);
    
    printf("[VULKAN] Contexto Vulkan inicializado com sucesso\n");
    return true;
}