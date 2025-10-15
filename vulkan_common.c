#include "string_common.h"
#include <SDL.h>
#include "vulkan_common.h"
#include <vulkan/vulkan_symbol_wrapper.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>


// #include <dynamic/dylib.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

#define VK_REMAP_TO_TEXFMT(fmt) ((fmt == VK_FORMAT_R5G6B5_UNORM_PACK16) ? VK_FORMAT_R8G8B8A8_UNORM : fmt)

#define VULKAN_COLORSPACE_EXTENSION_NAME "VK_EXT_swapchain_colorspace"

static void*                       g_vulkan_library;
struct retro_hw_render_context_negotiation_interface_vulkan *g_iface = NULL;

enum gfx_ctx_api
{
   GFX_CTX_NONE = 0,
   GFX_CTX_OPENGL_API,
   GFX_CTX_OPENGL_ES_API,
   GFX_CTX_DIRECT3D8_API,
   GFX_CTX_DIRECT3D9_API,
   GFX_CTX_DIRECT3D10_API,
   GFX_CTX_DIRECT3D11_API,
   GFX_CTX_DIRECT3D12_API,
   GFX_CTX_OPENVG_API,
   GFX_CTX_VULKAN_API,
   GFX_CTX_METAL_API,
   GFX_CTX_RSX_API
};

typedef struct
{
   struct string_list *list;
   enum gfx_ctx_api api;
} gfx_api_gpu_map;

static gfx_api_gpu_map gpu_map[] = {
   { NULL,                   GFX_CTX_VULKAN_API     },
   { NULL,                   GFX_CTX_DIRECT3D10_API },
   { NULL,                   GFX_CTX_DIRECT3D11_API },
   { NULL,                   GFX_CTX_DIRECT3D12_API }
};

static VkDevice                      cached_device_vk;

static retro_vulkan_destroy_device_t cached_destroy_device_vk;

static const char *vulkan_optional_instance_extensions[] = {
};

static const char *vulkan_device_extensions[]  = {
   "VK_KHR_swapchain",
};

static const char *vulkan_optional_device_extensions[] = {
   "VK_KHR_sampler_mirror_clamp_to_edge",
};

void video_driver_set_gpu_api_devices(
      enum gfx_ctx_api api, struct string_list *list)
{
   int i;

   for (i = 0; i < (int)ARRAY_SIZE(gpu_map); i++)
   {
      if (api == gpu_map[i].api)
      {
         gpu_map[i].list = list;
         break;
      }
   }
}

// FIXME if necessary
size_t video_driver_set_gpu_api_version_string(const char *str)
{
   return 0;
}

static bool vulkan_find_extensions(const char * const *exts, unsigned num_exts,
      const VkExtensionProperties *properties, unsigned property_count)
{
   unsigned i, ext;
   bool found;
   for (ext = 0; ext < num_exts; ext++)
   {
      found = false;
      for (i = 0; i < property_count; i++)
      {
         if (string_is_equal(exts[ext], properties[i].extensionName))
         {
            found = true;
            break;
         }
      }

      if (!found)
         return false;
   }
   return true;
}

static bool vulkan_find_device_extensions(VkPhysicalDevice gpu,
      const char **enabled, unsigned *inout_enabled_count,
      const char **exts, unsigned num_exts,
      const char **optional_exts, unsigned num_optional_exts)
{
   uint32_t property_count;
   unsigned i;
   unsigned count                    = *inout_enabled_count;
   bool ret                          = true;
   VkExtensionProperties *properties = NULL;

   if (vkEnumerateDeviceExtensionProperties(gpu, NULL, &property_count, NULL) != VK_SUCCESS)
      return false;

   if (!(properties = (VkExtensionProperties*)malloc(property_count *
               sizeof(*properties))))
   {
      ret = false;
      goto end;
   }

   if (vkEnumerateDeviceExtensionProperties(gpu, NULL, &property_count, properties) != VK_SUCCESS)
   {
      ret = false;
      goto end;
   }

   if (!vulkan_find_extensions(exts, num_exts, properties, property_count))
   {
      printf("[Vulkan] Could not find device extension. Will attempt without it.\n");
      ret = false;
      goto end;
   }

   memcpy((void*)(enabled + count), exts, num_exts * sizeof(*exts));
   count += num_exts;

   for (i = 0; i < num_optional_exts; i++)
      if (vulkan_find_extensions(&optional_exts[i], 1, properties, property_count))
         enabled[count++] = optional_exts[i];

end:
   free(properties);
   *inout_enabled_count = count;
   return ret;
}

static bool vulkan_context_init_gpu(gfx_ctx_vulkan_data_t *vk)
{
   unsigned i;
   uint32_t gpu_count               = 0;
   VkPhysicalDevice *gpus           = NULL;
   union string_list_elem_attr attr = {0};
   int gpu_index                    = 0;

   printf("LINE %d\n", __LINE__);
   if(!vk) {
      printf("[Vulkan] Invalid Vulkan context data.\n");
      return false;
   }

   if (vkEnumeratePhysicalDevices(vk->context.instance,
            &gpu_count, NULL) != VK_SUCCESS)
   {
      printf("[Vulkan] Failed to enumerate physical devices.\n");
      return false;
   }

   printf("LINE %d\n", __LINE__);

   if (!(gpus = (VkPhysicalDevice*)calloc(gpu_count, sizeof(*gpus))))
   {
      printf("[Vulkan] Failed to enumerate physical devices.\n");
      return false;
   }

   printf("LINE %d\n", __LINE__);

   if (vkEnumeratePhysicalDevices(vk->context.instance,
            &gpu_count, gpus) != VK_SUCCESS)
   {
      printf("[Vulkan] Failed to enumerate physical devices.\n");
      free(gpus);
      return false;
   }

   printf("LINE %d\n", __LINE__);

   if (gpu_count < 1)
   {
      printf("[Vulkan] Failed to enumerate Vulkan physical device.\n");
      free(gpus);
      return false;
   }

   printf("LINE %d\n", __LINE__);

   if (vk->gpu_list)
      string_list_free(vk->gpu_list);

   printf("LINE %d\n", __LINE__);
   vk->gpu_list = string_list_new();

   for (i = 0; i < gpu_count; i++)
   {
      VkPhysicalDeviceProperties gpu_properties;

      vkGetPhysicalDeviceProperties(gpus[i],
            &gpu_properties);

      printf("[Vulkan] Found GPU at index %d: \"%s\".\n", i, gpu_properties.deviceName);

      string_list_append(vk->gpu_list, gpu_properties.deviceName, attr);
   }

   printf("LINE %d\n", __LINE__);

   video_driver_set_gpu_api_devices(GFX_CTX_VULKAN_API, vk->gpu_list);

   if (0 <= gpu_index && gpu_index < (int)gpu_count)
   {
      printf("[Vulkan] Using GPU index %d.\n", gpu_index);
      vk->context.gpu = gpus[gpu_index];
   }
   else
   {
      printf("[Vulkan] Invalid GPU index %d, using first device found.\n", gpu_index);
      vk->context.gpu = gpus[0];
   }

   printf("LINE %d\n", __LINE__);

   free(gpus);
   return true;
}

bool vulkan_load_instance_symbols(gfx_ctx_vulkan_data_t *vk)
{
   if (!vulkan_symbol_wrapper_load_core_instance_symbols(vk->context.instance))
      return false;

   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_EXTENSION_SYMBOL(vk->context.instance, vkDestroySurfaceKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_EXTENSION_SYMBOL(vk->context.instance, vkGetPhysicalDeviceSurfaceSupportKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_EXTENSION_SYMBOL(vk->context.instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_EXTENSION_SYMBOL(vk->context.instance, vkGetPhysicalDeviceSurfaceFormatsKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_EXTENSION_SYMBOL(vk->context.instance, vkGetPhysicalDeviceSurfacePresentModesKHR);
   return true;
}

bool vulkan_load_device_symbols(gfx_ctx_vulkan_data_t *vk)
{
   if (!vulkan_symbol_wrapper_load_core_device_symbols(vk->context.device))
      return false;

   VULKAN_SYMBOL_WRAPPER_LOAD_DEVICE_EXTENSION_SYMBOL(vk->context.device, vkCreateSwapchainKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_DEVICE_EXTENSION_SYMBOL(vk->context.device, vkDestroySwapchainKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_DEVICE_EXTENSION_SYMBOL(vk->context.device, vkGetSwapchainImagesKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_DEVICE_EXTENSION_SYMBOL(vk->context.device, vkAcquireNextImageKHR);
   VULKAN_SYMBOL_WRAPPER_LOAD_DEVICE_EXTENSION_SYMBOL(vk->context.device, vkQueuePresentKHR);
   return true;
}

static VkDevice vulkan_context_create_device_wrapper(
      VkPhysicalDevice gpu, void *opaque,
      const VkDeviceCreateInfo *create_info)
{
   VkResult res;
   VkDeviceCreateInfo info        = *create_info;
   VkDevice device                = VK_NULL_HANDLE;
   const char **device_extensions = (const char **)malloc(
         (info.enabledExtensionCount +
               ARRAY_SIZE(vulkan_device_extensions) +
               ARRAY_SIZE(vulkan_optional_device_extensions)) * sizeof(const char *));

   memcpy((void*)device_extensions, info.ppEnabledExtensionNames, info.enabledExtensionCount * sizeof(const char *));
   info.ppEnabledExtensionNames = device_extensions;

   if (!(vulkan_find_device_extensions(gpu,
         device_extensions, &info.enabledExtensionCount,
         vulkan_device_extensions, ARRAY_SIZE(vulkan_device_extensions),
         vulkan_optional_device_extensions,
         ARRAY_SIZE(vulkan_optional_device_extensions))))
   {
      printf("[Vulkan] Could not find required device extensions.\n");
      free((void*)device_extensions);
      return VK_NULL_HANDLE;
   }

   /* When we get around to using fancier features we can chain in PDF2 stuff. */
   if ((res = vkCreateDevice(gpu, &info, NULL, &device)) != VK_SUCCESS)
   {
      printf("[Vulkan] Failed to create device (%d).\n", res);
      device = VK_NULL_HANDLE;
   }

   free((void*)device_extensions);
   return device;
}

bool vulkan_context_init_device(gfx_ctx_vulkan_data_t *vk)
{
   uint32_t queue_count;
   unsigned i;
   const char *enabled_device_extensions[8];
   VkDeviceCreateInfo device_info;
   VkDeviceQueueCreateInfo queue_info;
   static const float one                  = 1.0f;
   bool found_queue                        = false;
//    video_driver_state_t *video_st          = video_state_get_ptr();

   printf("LINE %d\n", __LINE__);

   VkPhysicalDeviceFeatures features       = { false };

   unsigned enabled_device_extension_count = 0;



   queue_info.sType                        = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
   queue_info.pNext                        = NULL;
   queue_info.flags                        = 0;
   queue_info.queueFamilyIndex             = 0;
   queue_info.queueCount                   = 0;
   queue_info.pQueuePriorities             = NULL;

   device_info.sType                       = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
   device_info.pNext                       = NULL;
   device_info.flags                       = 0;
   device_info.queueCreateInfoCount        = 0;
   device_info.pQueueCreateInfos           = NULL;
   device_info.enabledLayerCount           = 0;
   device_info.ppEnabledLayerNames         = NULL;
   device_info.enabledExtensionCount       = 0;
   device_info.ppEnabledExtensionNames     = NULL;
   device_info.pEnabledFeatures            = NULL;

   printf("LINE %d\n", __LINE__);

   if (g_iface)
   {
      if (g_iface->interface_type != RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN)
      {
         printf("[Vulkan] Got HW context negotiation interface, but it's the wrong API.\n");
         g_iface = NULL;
      }
      else if (g_iface->interface_version == 0)
      {
         printf("[Vulkan] Got HW context negotiation interface, but it's the wrong interface version.\n");
         g_iface = NULL;
      }
      else
         printf("[Vulkan] Got HW context negotiation interface %u.\n", g_iface->interface_version);
   }

   printf("LINE %d\n", __LINE__);

   if (!vulkan_context_init_gpu(vk))
      return false;

   printf("LINE %d\n", __LINE__);

   printf("g_iface >>>>>>>>>>>>>>>>>> %p\n", (void*)g_iface);
   printf("g_iface create_device >>>>>>>>>>>>>>>>>> %p\n", g_iface->create_device);
   // printf("g_iface cached_device_vk >>>>>>>>>>>>>>>>>>%d\n", cached_device_vk);

   // vulkan_symbol_wrapper_init(vkGetInstanceProcAddr);

   if (!cached_device_vk && g_iface && g_iface->create_device)
   {
      printf("LINE %d\n", __LINE__);
      struct retro_vulkan_context context = { 0 };

      bool ret = false;

      if (     (g_iface->interface_version >= 2)
            &&  g_iface->create_device2)
      {
         ret = g_iface->create_device2(&context, vk->context.instance,
               vk->context.gpu,
               vk->vk_surface,
               vulkan_symbol_wrapper_instance_proc_addr(),
               vulkan_context_create_device_wrapper, vk);

         if (!ret)
         {
            printf("[Vulkan] Failed to create_device2 on provided VkPhysicalDevice, letting core decide which GPU to use.\n");
            vk->context.gpu = VK_NULL_HANDLE;
            ret = g_iface->create_device2(&context, vk->context.instance,
                  vk->context.gpu,
                  vk->vk_surface,
                  vulkan_symbol_wrapper_instance_proc_addr(),
                  vulkan_context_create_device_wrapper, vk);
         }
      }
      else
      {
         printf("LINE %d\n", __LINE__);

         printf("g_iface vk->context.gpu >>>>>>>>>>>>>>>>>> %p\n", vk->context.gpu);
         printf("g_iface vk->vk_surface >>>>>>>>>>>>>>>>>> %p\n", vk->vk_surface);
         printf("g_iface vk->context.instance >>>>>>>>>>>>>>>>>> %p\n", vk->context.instance);
         // printf("g_iface ARRAY_SIZE >>>>>>>>>>>>>>>>>> %d\n", ARRAY_SIZE(vulkan_device_extensions));
         printf("g_iface vulkan_device_extensions >>>>>>>>>>>>>>>>>> %s\n", vulkan_device_extensions[0]);

         vulkan_symbol_wrapper_init(vkGetInstanceProcAddr);
         PFN_vkGetInstanceProcAddr fn = vulkan_symbol_wrapper_instance_proc_addr();

         printf("g_iface vk->context.instance >>>>>>>>>>>>>>>>>> %p\n", fn);

         ret = g_iface->create_device(&context, vk->context.instance,
               vk->context.gpu,
               vk->vk_surface,
               fn,
               vulkan_device_extensions,
               ARRAY_SIZE(vulkan_device_extensions),
               NULL,
               0,
               &features);
      }

      printf("LINE %d\n", __LINE__);

      if (ret)
      {
         printf("LINE %d\n", __LINE__);
         if (vk->context.gpu != VK_NULL_HANDLE && context.gpu != vk->context.gpu)
            printf("[Vulkan] Got unexpected VkPhysicalDevice, despite RetroArch using explicit physical device.\n");

         vk->context.destroy_device       = g_iface->destroy_device;

         vk->context.device               = context.device;
         vk->context.queue                = context.queue;
         vk->context.gpu                  = context.gpu;
         vk->context.graphics_queue_index = context.queue_family_index;
         vk->context.queue                = context.queue;

         if (context.presentation_queue != context.queue)
         {
            printf("[Vulkan] Present queue != graphics queue. This is currently not supported.\n");
            return false;
         }
      }
      else
      {
         printf("[Vulkan] Failed to create device with negotiation interface. Falling back to default path.\n");
      }
   }

   printf("LINE %d\n", __LINE__);
   if (cached_device_vk && cached_destroy_device_vk)
   {
      vk->context.destroy_device = cached_destroy_device_vk;
      cached_destroy_device_vk   = NULL;
   }

   printf("LINE %d\n", __LINE__);

   vkGetPhysicalDeviceProperties(vk->context.gpu,
         &vk->context.gpu_properties);
   vkGetPhysicalDeviceMemoryProperties(vk->context.gpu,
         &vk->context.memory_properties);


   printf("[Vulkan] Using GPU: \"%s\".\n", vk->context.gpu_properties.deviceName);

   // {
   //    char version_str[128];
   //    size_t _len            = snprintf(version_str      , sizeof(version_str)      , "%u", VK_VERSION_MAJOR(vk->context.gpu_properties.apiVersion));
   //    version_str[  _len]    = '.';
   //    version_str[++_len]    = '\0';
   //    _len                  += snprintf(version_str + _len, sizeof(version_str) - _len, "%u", VK_VERSION_MINOR(vk->context.gpu_properties.apiVersion));
   //    version_str[  _len]    = '.';
   //    version_str[++_len]    = '\0';
   //    snprintf(version_str + _len, sizeof(version_str) - _len, "%u", VK_VERSION_PATCH(vk->context.gpu_properties.apiVersion));
   //    video_driver_set_gpu_api_version_string(version_str);
   // }

   printf("LINE %d\n", __LINE__);
   if (vk->context.device == VK_NULL_HANDLE)
   {
      printf("LINE %d\n", __LINE__);
      VkQueueFamilyProperties *queue_properties = NULL;
      vkGetPhysicalDeviceQueueFamilyProperties(vk->context.gpu,
            &queue_count, NULL);

      if (queue_count < 1)
      {
         printf("[Vulkan] Invalid number of queues detected.\n");
         return false;
      }

      printf("LINE %d\n", __LINE__);

      if (!(queue_properties = (VkQueueFamilyProperties*)malloc(queue_count *
                  sizeof(*queue_properties))))
         return false;

      vkGetPhysicalDeviceQueueFamilyProperties(vk->context.gpu,
            &queue_count, queue_properties);

      printf("LINE %d\n", __LINE__);

      for (i = 0; i < queue_count; i++)
      {
         VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
         VkBool32 supported    = VK_FALSE;
         vkGetPhysicalDeviceSurfaceSupportKHR(
               vk->context.gpu, i,
               vk->vk_surface, &supported);
         if (supported && ((queue_properties[i].queueFlags & required) == required))
         {
            vk->context.graphics_queue_index = i;
            printf("[Vulkan] Queue family %u supports %u sub-queues.\n",
                  i, queue_properties[i].queueCount);
            found_queue = true;
            break;
         }
      }

      printf("LINE %d\n", __LINE__);

      free(queue_properties);

      if (!found_queue)
      {
         printf("[Vulkan] Did not find suitable graphics queue.\n");
         return false;
      }

      if (!(vulkan_find_device_extensions(vk->context.gpu,
              enabled_device_extensions, &enabled_device_extension_count,
              vulkan_device_extensions, ARRAY_SIZE(vulkan_device_extensions),
              vulkan_optional_device_extensions,
              ARRAY_SIZE(vulkan_optional_device_extensions))))
      {
          printf("[Vulkan] Could not find required device extensions.\n");
          return false;
      }

      queue_info.queueFamilyIndex         = vk->context.graphics_queue_index;
      queue_info.queueCount               = 1;
      queue_info.pQueuePriorities         = &one;

      device_info.queueCreateInfoCount    = 1;
      device_info.pQueueCreateInfos       = &queue_info;
      device_info.enabledExtensionCount   = enabled_device_extension_count;
      device_info.ppEnabledExtensionNames = enabled_device_extensions;
      device_info.pEnabledFeatures        = &features;

      if (cached_device_vk)
      {
         vk->context.device = cached_device_vk;
         cached_device_vk   = NULL;

         // video_st->flags   |= VIDEO_FLAG_CACHE_CONTEXT_ACK;
         printf("[Vulkan] Using cached Vulkan context.\n");
      }
      else if (vkCreateDevice(vk->context.gpu, &device_info,
               NULL, &vk->context.device) != VK_SUCCESS)
      {
         printf("[Vulkan] Failed to create device.\n");
         return false;
      }
   }

   printf("LINE %d\n", __LINE__);
   // if (!vulkan_load_device_symbols(vk))
   // {
   //    printf("[Vulkan] Failed to load device symbols.\n");
   //    return false;
   // }

   VULKAN_SYMBOL_WRAPPER_LOAD_INSTANCE_SYMBOL(vk->context.instance, "vkGetDeviceQueue", vkGetDeviceQueue);

   if (vk->context.queue == VK_NULL_HANDLE)
   {
      printf("LINE %d\n", __LINE__);
      vkGetDeviceQueue(vk->context.device,
            vk->context.graphics_queue_index, 0, &vk->context.queue);
   }

#ifdef HAVE_THREADS
   vk->context.queue_lock = slock_new();
   if (!vk->context.queue_lock)
   {
      printf("[Vulkan] Failed to create queue lock.\n");
      return false;
   }
#endif

   printf("LINE %d\n", __LINE__);
   return true;
}

static void vulkan_create_wait_fences(gfx_ctx_vulkan_data_t *vk)
{
   unsigned i;
   VkFenceCreateInfo fence_info;

   fence_info.sType                = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
   fence_info.pNext                = NULL;
   fence_info.flags                = 0;

   PFN_vkCreateFence vkCreateFence =
      (PFN_vkCreateFence) vkGetInstanceProcAddr(vk->context.instance, "vkCreateFence");

   for (i = 0; i < vk->context.num_swapchain_images; i++)
   {
      if (!vk->context.swapchain_fences[i])
         vkCreateFence(vk->context.device, &fence_info, NULL,
               &vk->context.swapchain_fences[i]);
   }

   vk->context.current_frame_index = 0;
}

static void vulkan_acquire_clear_fences(gfx_ctx_vulkan_data_t *vk)
{
   unsigned i;
   for (i = 0; i < vk->context.num_swapchain_images; i++)
   {
      if (vk->context.swapchain_fences[i])
      {
         vkDestroyFence(vk->context.device,
               vk->context.swapchain_fences[i], NULL);
         vk->context.swapchain_fences[i]        = VK_NULL_HANDLE;
      }
      vk->context.swapchain_fences_signalled[i] = false;

      if (vk->context.swapchain_wait_semaphores[i])
      {
         struct vulkan_context *ctx = &vk->context;
         VkSemaphore sem            = vk->context.swapchain_wait_semaphores[i];
         assert(ctx->num_recycled_acquire_semaphores < VULKAN_MAX_SWAPCHAIN_IMAGES);
         ctx->swapchain_recycled_semaphores[ctx->num_recycled_acquire_semaphores++] = sem;
      }
      vk->context.swapchain_wait_semaphores[i] = VK_NULL_HANDLE;
   }

   vk->context.current_frame_index = 0;
}

bool vulkan_create_swapchain(gfx_ctx_vulkan_data_t *vk,
      unsigned width, unsigned height,
      int8_t swap_interval)
{
   unsigned i;
   uint32_t format_count;
   uint32_t desired_swapchain_images;
   VkSurfaceCapabilitiesKHR surface_properties;
   VkSurfaceFormatKHR formats[256];
   VkPresentModeKHR present_modes[16];
   VkExtent2D swapchain_size;
   VkSurfaceFormatKHR format;
   VkSwapchainKHR old_swapchain;
   VkSwapchainCreateInfoKHR info;
   VkSurfaceTransformFlagBitsKHR pre_transform;
   uint32_t present_mode_count             = 0;
   VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;
   VkCompositeAlphaFlagBitsKHR composite   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

   bool vsync                              = false;
   bool adaptive_vsync                     = false;

   format.format                           = VK_FORMAT_UNDEFINED;
   format.colorSpace                       = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

   printf("LINE %d\n", __LINE__);

   PFN_vkDeviceWaitIdle vkDeviceWaitIdle =
      (PFN_vkDeviceWaitIdle) vkGetInstanceProcAddr(vk->context.instance, "vkDeviceWaitIdle");

   vkDeviceWaitIdle(vk->context.device);
   vulkan_acquire_clear_fences(vk);

   printf("LINE %d\n", __LINE__);

   PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
      (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

   vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk->context.gpu,
         vk->vk_surface, &surface_properties);

   printf("LINE %d\n", __LINE__);

   /* Skip creation when window is minimized */
   if (   !surface_properties.currentExtent.width
       && !surface_properties.currentExtent.height)
      return false;

   if (     (swap_interval == 0)
         && (vk->flags & VK_DATA_FLAG_EMULATE_MAILBOX)
         && vsync)
   {
      swap_interval  =  (adaptive_vsync) ? -1 : 1;
      vk->flags     |=  VK_DATA_FLAG_EMULATING_MAILBOX;
   }
   else
      vk->flags     &= ~VK_DATA_FLAG_EMULATING_MAILBOX;

   vk->flags        |= VK_DATA_FLAG_CREATED_NEW_SWAPCHAIN;

   if (       (vk->swapchain != VK_NULL_HANDLE)
         && (!(vk->context.flags & VK_CTX_FLAG_INVALID_SWAPCHAIN))
         &&   (vk->context.swapchain_width  == width)
         &&   (vk->context.swapchain_height == height)
         &&   (vk->context.swap_interval    == swap_interval))
   {
      vulkan_create_wait_fences(vk);

      if (     (vk->flags & VK_DATA_FLAG_EMULATING_MAILBOX)
            && (vk->mailbox.swapchain == VK_NULL_HANDLE))
      {
         vk->flags                &= ~VK_DATA_FLAG_CREATED_NEW_SWAPCHAIN;
         return true;
      }
      else if (
               (!(vk->flags & VK_DATA_FLAG_EMULATING_MAILBOX))
            &&   (vk->mailbox.swapchain != VK_NULL_HANDLE))
      {
         VkResult res = VK_SUCCESS;

         if (res == VK_SUCCESS)
         {
            vk->context.flags |=  VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN;
            vk->flags         &= ~VK_DATA_FLAG_CREATED_NEW_SWAPCHAIN;
            return true;
         }

         /* We failed for some reason, so create a new swapchain. */
         vk->context.flags    &= ~VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN;
      }
      else
      {
         vk->flags &= ~VK_DATA_FLAG_CREATED_NEW_SWAPCHAIN;
         return true;
      }
   }

   printf("LINE %d\n", __LINE__);

   PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR =
      (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");

   vkGetPhysicalDeviceSurfacePresentModesKHR(
         vk->context.gpu, vk->vk_surface,
         &present_mode_count, NULL);
   if (present_mode_count < 1 || present_mode_count > 16)
   {
      printf("[Vulkan] Bogus present modes found.\n");
      return false;
   }

   printf("LINE %d\n", __LINE__);

   vkGetPhysicalDeviceSurfacePresentModesKHR(
         vk->context.gpu, vk->vk_surface,
         &present_mode_count, present_modes);

   vk->context.swap_interval = swap_interval;

   for (i = 0; i < present_mode_count; i++)
      vk->context.present_modes[i] = present_modes[i];

   /* Prefer IMMEDIATE without vsync */
   for (i = 0; i < present_mode_count; i++)
   {
      if (     !swap_interval
            && !vsync
            && present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
      {
         swapchain_present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
         break;
      }

      if (     swap_interval < 0
            && present_modes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
      {
         swapchain_present_mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
         break;
      }
   }

   printf("LINE %d\n", __LINE__);

   /* If still in FIFO with no swap interval, try MAILBOX */
   for (i = 0; i < present_mode_count; i++)
   {
      if (     !swap_interval
            && swapchain_present_mode == VK_PRESENT_MODE_FIFO_KHR
            && present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
      {
         swapchain_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
         break;
      }
   }

   printf("LINE %d\n", __LINE__);

   /* Present mode logging */
   if (vk->swapchain == VK_NULL_HANDLE)
   {
      for (i = 0; i < present_mode_count; i++)
      {
         switch (present_modes[i])
         {
            case VK_PRESENT_MODE_IMMEDIATE_KHR:
               printf("[Vulkan] Swapchain supports present mode: IMMEDIATE.\n");
               break;
            case VK_PRESENT_MODE_MAILBOX_KHR:
               printf("[Vulkan] Swapchain supports present mode: MAILBOX.\n");
               break;
            case VK_PRESENT_MODE_FIFO_KHR:
               printf("[Vulkan] Swapchain supports present mode: FIFO.\n");
               break;
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
               printf("[Vulkan] Swapchain supports present mode: FIFO_RELAXED.\n");
               break;
            default:
               break;
         }
      }
   }
   else
   {
      switch (swapchain_present_mode)
      {
         case VK_PRESENT_MODE_IMMEDIATE_KHR:
            printf("[Vulkan] Creating swapchain with present mode: IMMEDIATE.\n");
            break;
         case VK_PRESENT_MODE_MAILBOX_KHR:
            printf("[Vulkan] Creating swapchain with present mode: MAILBOX.\n");
            break;
         case VK_PRESENT_MODE_FIFO_KHR:
            printf("[Vulkan] Creating swapchain with present mode: FIFO.\n");
            break;
         case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            printf("[Vulkan] Creating swapchain with present mode: FIFO_RELAXED.\n");
            break;
         default:
            break;
      }
   }

   printf("LINE %d\n", __LINE__);

   PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR =
      (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");

   vkGetPhysicalDeviceSurfaceFormatsKHR(vk->context.gpu,
         vk->vk_surface, &format_count, NULL);
   vkGetPhysicalDeviceSurfaceFormatsKHR(vk->context.gpu,
         vk->vk_surface, &format_count, formats);

   format.format = VK_FORMAT_UNDEFINED;
   if (     format_count == 1
         && (formats[0].format == VK_FORMAT_UNDEFINED))
   {
      format        = formats[0];
      format.format = VK_FORMAT_B8G8R8A8_UNORM;
   }
   else
   {
      if (format_count == 0)
      {
         printf("[Vulkan] Surface has no formats.\n");
         return false;
      }


      {
         for (i = 0; i < format_count; i++)
         {
            if (
                     formats[i].format == VK_FORMAT_R8G8B8A8_UNORM
                  || formats[i].format == VK_FORMAT_B8G8R8A8_UNORM
                  || formats[i].format == VK_FORMAT_A8B8G8R8_UNORM_PACK32)
            {
               format = formats[i];
               break;
            }
         }
      }

      if (format.format == VK_FORMAT_UNDEFINED)
         format = formats[0];
   }

   if (surface_properties.currentExtent.width == UINT32_MAX)
   {
      swapchain_size.width     = width;
      swapchain_size.height    = height;
   }
   else
      swapchain_size           = surface_properties.currentExtent;

   /* Clamp swapchain size to boundaries. */
   if (swapchain_size.width > surface_properties.maxImageExtent.width)
      swapchain_size.width = surface_properties.maxImageExtent.width;
   if (swapchain_size.width < surface_properties.minImageExtent.width)
      swapchain_size.width = surface_properties.minImageExtent.width;
   if (swapchain_size.height > surface_properties.maxImageExtent.height)
      swapchain_size.height = surface_properties.maxImageExtent.height;
   if (swapchain_size.height < surface_properties.minImageExtent.height)
      swapchain_size.height = surface_properties.minImageExtent.height;

   printf("LINE %d\n", __LINE__);

   if (     (swapchain_size.width  == 0)
         && (swapchain_size.height == 0))
   {
      /* Cannot create swapchain yet, try again later. */
      if (vk->swapchain != VK_NULL_HANDLE)
         vkDestroySwapchainKHR(vk->context.device, vk->swapchain, NULL);
      vk->swapchain                    = VK_NULL_HANDLE;
      vk->context.swapchain_width      = width;
      vk->context.swapchain_height     = height;
      vk->context.num_swapchain_images = 1;

      memset(vk->context.swapchain_images, 0, sizeof(vk->context.swapchain_images));
      printf("[Vulkan] Cannot create a swapchain yet. Will try again later...\n");
      return true;
   }

   /* Unless we have other reasons to clamp, we should prefer 3 images.
    * We hard sync against the swapchain, so if we have 2 images,
    * we would be unable to overlap CPU and GPU, which can get very slow
    * for GPU-rendered cores. */
   desired_swapchain_images    = 2;

   /* We don't clamp the number of images requested to what is reported
    * as supported by the implementation in surface_properties.minImageCount,
    * because MESA always reports a minImageCount of 4, but 3 and 2 work
    * perfectly well, even if it's out of spec. */

   if (     (surface_properties.maxImageCount > 0)
         && (desired_swapchain_images > surface_properties.maxImageCount))
      desired_swapchain_images = surface_properties.maxImageCount;

   if (surface_properties.supportedTransforms
         & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
      pre_transform            = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   else
      pre_transform            = surface_properties.currentTransform;

   if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
      composite                = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
      composite                = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
   else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
      composite                = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
   else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
      composite                = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;

   old_swapchain               = vk->swapchain;

   printf("LINE %d\n", __LINE__);

   info.sType                  = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
   info.pNext                  = NULL;
   info.flags                  = 0;
   info.surface                = vk->vk_surface;
   info.minImageCount          = desired_swapchain_images;
   info.imageFormat            = format.format;
   info.imageColorSpace        = format.colorSpace;
   info.imageExtent.width      = swapchain_size.width;
   info.imageExtent.height     = swapchain_size.height;
   info.imageArrayLayers       = 1;
   info.imageUsage             =  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                | VK_IMAGE_USAGE_SAMPLED_BIT;
   info.imageSharingMode       = VK_SHARING_MODE_EXCLUSIVE;
   info.queueFamilyIndexCount  = 0;
   info.pQueueFamilyIndices    = NULL;
   info.preTransform           = pre_transform;
   info.compositeAlpha         = composite;
   info.presentMode            = swapchain_present_mode;
   info.clipped                = VK_TRUE;
   info.oldSwapchain           = old_swapchain;

   printf("LINE %d\n", __LINE__);

   PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR =
      (PFN_vkDestroySwapchainKHR) vkGetInstanceProcAddr(vk->context.instance, "vkDestroySwapchainKHR");
   PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR =
      (PFN_vkCreateSwapchainKHR) vkGetInstanceProcAddr(vk->context.instance, "vkCreateSwapchainKHR");

   info.oldSwapchain = VK_NULL_HANDLE;
   if (old_swapchain != VK_NULL_HANDLE)
      vkDestroySwapchainKHR(vk->context.device, old_swapchain, NULL);

   if (vkCreateSwapchainKHR(vk->context.device,
            &info, NULL, &vk->swapchain) != VK_SUCCESS)
   {
      printf("[Vulkan] Failed to create swapchain.\n");
      return false;
   }

   printf("LINE %d\n", __LINE__);

   vk->context.swapchain_width        = swapchain_size.width;
   vk->context.swapchain_height       = swapchain_size.height;


   /* Make sure we create a backbuffer format that is as we expect. */
   switch (format.format)
   {
      case VK_FORMAT_B8G8R8A8_SRGB:
         vk->context.swapchain_format  = VK_FORMAT_B8G8R8A8_UNORM;
         vk->context.flags            |= VK_CTX_FLAG_SWAPCHAIN_IS_SRGB;
         break;

      case VK_FORMAT_R8G8B8A8_SRGB:
         vk->context.swapchain_format  = VK_FORMAT_R8G8B8A8_UNORM;
         vk->context.flags            |= VK_CTX_FLAG_SWAPCHAIN_IS_SRGB;
         break;

      case VK_FORMAT_R8G8B8_SRGB:
         vk->context.swapchain_format  = VK_FORMAT_R8G8B8_UNORM;
         vk->context.flags            |= VK_CTX_FLAG_SWAPCHAIN_IS_SRGB;
         break;

      case VK_FORMAT_B8G8R8_SRGB:
         vk->context.swapchain_format  = VK_FORMAT_B8G8R8_UNORM;
         vk->context.flags            |= VK_CTX_FLAG_SWAPCHAIN_IS_SRGB;
         break;

      default:
         vk->context.swapchain_format  = format.format;
         break;
   }

   printf("LINE %d\n", __LINE__);

   PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR =
      (PFN_vkGetSwapchainImagesKHR) vkGetInstanceProcAddr(vk->context.instance, "vkGetSwapchainImagesKHR");

   vkGetSwapchainImagesKHR(vk->context.device, vk->swapchain,
         &vk->context.num_swapchain_images, NULL);
   vkGetSwapchainImagesKHR(vk->context.device, vk->swapchain,
         &vk->context.num_swapchain_images, vk->context.swapchain_images);

   if (old_swapchain == VK_NULL_HANDLE)
      printf("[Vulkan] Got %u swapchain images.\n",
            vk->context.num_swapchain_images);

   /* Force driver to reset swapchain image handles. */
   printf("LINE %d\n", __LINE__);
   vk->context.flags                 |=  VK_CTX_FLAG_INVALID_SWAPCHAIN;
   printf("LINE %d\n", __LINE__);
   vk->context.flags                 &= ~VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN;
   printf("LINE %d\n", __LINE__);
   vulkan_create_wait_fences(vk);

   printf("LINE %d\n", __LINE__);

   return true;
}

static bool vulkan_find_instance_extensions(
      const char **enabled, unsigned *inout_enabled_count,
      const char **exts, unsigned num_exts,
      const char **optional_exts, unsigned num_optional_exts)
{
   uint32_t property_count;
   unsigned i;
   unsigned count                    = *inout_enabled_count;
   bool ret                          = true;
   VkExtensionProperties *properties = NULL;

   if (vkEnumerateInstanceExtensionProperties(NULL, &property_count, NULL) != VK_SUCCESS)
      return false;

   if (!(properties = (VkExtensionProperties*)malloc(property_count *
               sizeof(*properties))))
   {
      ret = false;
      goto end;
   }

   if (vkEnumerateInstanceExtensionProperties(NULL, &property_count, properties) != VK_SUCCESS)
   {
      ret = false;
      goto end;
   }

   if (!vulkan_find_extensions(exts, num_exts, properties, property_count))
   {
      printf("[Vulkan] Could not find required instance extensions. Will attempt without them.\n");
      ret = false;
      goto end;
   }

   memcpy((void*)(enabled + count), exts, num_exts * sizeof(*exts));
   count += num_exts;

   for (i = 0; i < num_optional_exts; i++)
      if (vulkan_find_extensions(&optional_exts[i], 1, properties, property_count))
         enabled[count++] = optional_exts[i];

end:
   free(properties);
   *inout_enabled_count = count;
   return ret;
}

static VkInstance vulkan_context_create_instance_wrapper(void *opaque, const VkInstanceCreateInfo *create_info)
{
   VkResult res;
   uint32_t i, layer_count;
   VkLayerProperties properties[128];
   gfx_ctx_vulkan_data_t *vk        = (gfx_ctx_vulkan_data_t *)opaque;
   VkInstanceCreateInfo info        = *create_info;
   VkInstance instance              = VK_NULL_HANDLE;
   const char **instance_extensions = (const char**)malloc((info.enabledExtensionCount + 3
                                                          + ARRAY_SIZE(vulkan_optional_device_extensions)) * sizeof(const char *));
   const char **instance_layers     = (const char**)malloc((info.enabledLayerCount     + 1)                * sizeof(const char *));

   const char *required_extensions[3];
   uint32_t required_extension_count = 0;

   memcpy((void*)instance_extensions, info.ppEnabledExtensionNames, info.enabledExtensionCount * sizeof(const char *));
   memcpy((void*)instance_layers,     info.ppEnabledLayerNames,     info.enabledLayerCount     * sizeof(const char *));
   info.ppEnabledExtensionNames     = instance_extensions;
   info.ppEnabledLayerNames         = instance_layers;

   required_extensions[required_extension_count++] = "VK_KHR_surface";

   switch (vk->wsi_type)
   {
      case VULKAN_WSI_WAYLAND:
         required_extensions[required_extension_count++] = "VK_KHR_wayland_surface";
         break;
      case VULKAN_WSI_ANDROID:
         required_extensions[required_extension_count++] = "VK_KHR_android_surface";
         break;
      case VULKAN_WSI_WIN32:
         required_extensions[required_extension_count++] = "VK_KHR_win32_surface";
         break;
      case VULKAN_WSI_XLIB:
         required_extensions[required_extension_count++] = "VK_KHR_xlib_surface";
         break;
      case VULKAN_WSI_XCB:
         required_extensions[required_extension_count++] = "VK_KHR_xcb_surface";
         break;
      case VULKAN_WSI_MIR:
         required_extensions[required_extension_count++] = "VK_KHR_mir_surface";
         break;
      case VULKAN_WSI_DISPLAY:
         required_extensions[required_extension_count++] = "VK_KHR_display";
         break;
      case VULKAN_WSI_MVK_MACOS:
      case VULKAN_WSI_MVK_IOS:
         required_extensions[required_extension_count++] = "VK_EXT_metal_surface";
         break;
      case VULKAN_WSI_NONE:
      default:
         break;
   }

   layer_count = ARRAY_SIZE(properties);
   vkEnumerateInstanceLayerProperties(&layer_count, properties);

   if (!(vulkan_find_instance_extensions(
            instance_extensions, &info.enabledExtensionCount,
            required_extensions, required_extension_count,
            vulkan_optional_instance_extensions,
            ARRAY_SIZE(vulkan_optional_instance_extensions))))
   {
      printf("[Vulkan] Instance does not support required extensions.\n");
      goto end;
   }

   if (info.pApplicationInfo)
   {
      uint32_t supported_instance_version = VK_API_VERSION_1_0;
      if (!vkEnumerateInstanceVersion || vkEnumerateInstanceVersion(&supported_instance_version) != VK_SUCCESS)
         supported_instance_version = VK_API_VERSION_1_0;

      if (supported_instance_version < info.pApplicationInfo->apiVersion)
      {
         printf("[Vulkan] Core requests apiVersion %u.%u, but it is not supported by loader.\n",
               VK_VERSION_MAJOR(info.pApplicationInfo->apiVersion),
               VK_VERSION_MINOR(info.pApplicationInfo->apiVersion));
         goto end;
      }
   }

   if ((res = vkCreateInstance(&info, NULL, &instance)) != VK_SUCCESS)
   {
      printf("[Vulkan] Failed to create Vulkan instance (%d).\n", res);
      printf("[Vulkan] If VULKAN_DEBUG=1 is enabled, make sure Vulkan validation layers are installed.\n");
      for (i = 0; i < info.enabledLayerCount; i++)
         printf("[Vulkan] Core explicitly enables layer (%s), this might be cause of failure.\n", info.ppEnabledLayerNames[i]);
      instance = VK_NULL_HANDLE;
      goto end;
   }

end:
   free((void*)instance_extensions);
   free((void*)instance_layers);
   return instance;
}

bool vulkan_context_init(gfx_ctx_vulkan_data_t *vk,
      enum vulkan_wsi_type type)
{
   VkApplicationInfo app;
   PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
   const char *prog_name          = NULL;

   if (g_iface && g_iface->interface_type != RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN)
   {
      printf("[Vulkan] Got HW context negotiation interface, but it's the wrong API.\n");
      g_iface = NULL;
   }

   if (g_iface && g_iface->interface_version == 0)
   {
      printf("[Vulkan] Got HW context negotiation interface, but it's the wrong interface version.\n");
      g_iface = NULL;
   }

   vk->wsi_type = type;

   if (!g_vulkan_library)
   {
#ifdef _WIN32
      g_vulkan_library = SDL_LoadObject("vulkan-1.dll");
#else
      g_vulkan_library = SDL_LoadObject("libvulkan.so.1");
      if (!g_vulkan_library)
         g_vulkan_library = SDL_LoadObject("libvulkan.so");
#endif
   }

   if (!g_vulkan_library)
   {
      printf("[Vulkan] Failed to open Vulkan loader.\n");
      return false;
   }

   printf("[Vulkan] Vulkan dynamic library loaded.\n");

   GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)SDL_LoadFunction(g_vulkan_library, "vkGetInstanceProcAddr");

   printf("[Vulkan] Vulkan dynamic library loaded vkGetInstanceProcAddr.\n");

   PFN_vkGetDeviceProcAddr vkGetPhysicalDeviceMemoryProperties =
      (PFN_vkGetDeviceProcAddr) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceMemoryProperties");
   PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceProperties");

   PFN_vkGetPhysicalDeviceImageFormatProperties target_func = 
    (PFN_vkGetPhysicalDeviceImageFormatProperties)vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceImageFormatProperties");
    
   if (!GetInstanceProcAddr)
   {
      printf("[Vulkan] Failed to load vkGetInstanceProcAddr symbol, broken loader?\n");
      return false;
   }

   printf("[Vulkan] Vulkan dynamic library loaded vkGetInstanceProcAddr.\n");

   vulkan_symbol_wrapper_init(GetInstanceProcAddr);

   if (!vulkan_symbol_wrapper_load_global_symbols())
   {
      printf("[Vulkan] Failed to load global Vulkan symbols, broken loader?\n");
      return false;
   }

   // if(!vulkan_symbol_wrapper_load_core_symbols(vk->context.instance))
   // {
   //    printf("[Vulkan] Failed to load core Vulkan symbols, broken loader?\n");
   // }

   prog_name              = "sdlarch";
   app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
   app.pNext              = NULL;
   app.pApplicationName   = prog_name;
   app.applicationVersion = 0;
   app.pEngineName        = prog_name;
   app.engineVersion      = 0;
   app.apiVersion         = VK_API_VERSION_1_0;

   if (g_iface)
   {
      if (!g_iface->get_application_info && g_iface->interface_version >= 2)
      {
         printf("[Vulkan] Core did not provide application info as required by v2.\n");
         return false;
      }

      if (g_iface->get_application_info)
      {
         const VkApplicationInfo *app_info = g_iface->get_application_info();

         if (!app_info && g_iface->interface_version >= 2)
         {
            printf("[Vulkan] Core did not provide application info as required by v2.\n");
            return false;
         }

         if (app_info)
         {
            app = *app_info;
         }
      }
   }

   if (app.apiVersion < VK_API_VERSION_1_1)
   {
      /* Try to upgrade to at least Vulkan 1.1 so that we can more easily make use of advanced features.
       * Vulkan 1.0 drivers are completely irrelevant these days. */
      uint32_t supported;
      if (     vkEnumerateInstanceVersion
            && (vkEnumerateInstanceVersion(&supported) == VK_SUCCESS)
            && (supported >= VK_API_VERSION_1_1))
         app.apiVersion = VK_API_VERSION_1_1;
   }

   if (cached_instance_vk)
   {
      vk->context.instance = cached_instance_vk;
      cached_instance_vk   = NULL;
   }
   else
   {
      if (g_iface
            && g_iface->interface_version >= 2
            && g_iface->create_instance)
         vk->context.instance = g_iface->create_instance(
               GetInstanceProcAddr, &app,
               vulkan_context_create_instance_wrapper, vk);
      else
      {
         VkInstanceCreateInfo info;
         info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
         info.pNext                   = NULL;
         info.flags                   = 0;
         info.pApplicationInfo        = &app;
         info.enabledLayerCount       = 0;
         info.ppEnabledLayerNames     = NULL;
         info.enabledExtensionCount   = 0;
         info.ppEnabledExtensionNames = NULL;
         vk->context.instance         = vulkan_context_create_instance_wrapper(vk, &info);
      }

      if (vk->context.instance == VK_NULL_HANDLE)
      {
         printf("[Vulkan] Failed to create Vulkan instance.\n");
         return false;
      }
   }

   if (!vulkan_load_instance_symbols(vk))
   {
      printf("[Vulkan] Failed to load instance symbols.\n");
      return false;
   }

   return true;
}

static unsigned vulkan_num_miplevels(unsigned width, unsigned height)
{
   unsigned size   = MAX(width, height);
   unsigned levels = 0;
   while (size)
   {
      levels++;
      size >>= 1;
   }
   return levels;
}

static unsigned vulkan_format_to_bpp(VkFormat format)
{
   switch (format)
   {
      case VK_FORMAT_B8G8R8A8_UNORM:
         return 4;
      case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
      case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
      case VK_FORMAT_R5G6B5_UNORM_PACK16:
         return 2;
      case VK_FORMAT_R8_UNORM:
         return 1;
      default: /* Unknown format */
         break;
   }

   return 0;
}

static void vulkan_destroy_texture(
      VkDevice device,
      struct vk_texture *tex)
{
   if (tex->mapped)
      vkUnmapMemory(device, tex->memory);
   if (tex->view)
      vkDestroyImageView(device, tex->view, NULL);
   if (tex->image)
      vkDestroyImage(device, tex->image, NULL);
   if (tex->buffer)
      vkDestroyBuffer(device, tex->buffer, NULL);
   if (tex->memory)
      vkFreeMemory(device, tex->memory, NULL);

#ifdef VULKAN_DEBUG_TEXTURE_ALLOC
   if (tex->image)
      vulkan_track_dealloc(tex->image);
#endif
   tex->type                          = VULKAN_TEXTURE_STREAMED;
   tex->flags                         = 0;
   tex->memory_type                   = 0;
   tex->width                         = 0;
   tex->height                        = 0;
   tex->offset                        = 0;
   tex->stride                        = 0;
   tex->size                          = 0;
   tex->mapped                        = NULL;
   tex->image                         = VK_NULL_HANDLE;
   tex->view                          = VK_NULL_HANDLE;
   tex->memory                        = VK_NULL_HANDLE;
   tex->buffer                        = VK_NULL_HANDLE;
   tex->format                        = VK_FORMAT_UNDEFINED;
   tex->memory_size                   = 0;
   tex->layout                        = VK_IMAGE_LAYOUT_UNDEFINED;
}

uint32_t vulkan_find_memory_type_fallback(
      const VkPhysicalDeviceMemoryProperties *mem_props,
      uint32_t device_reqs, uint32_t host_reqs_first,
      uint32_t host_reqs_second)
{
   uint32_t i;
   for (i = 0; i < VK_MAX_MEMORY_TYPES; i++)
   {
      if (     (device_reqs & (1u << i))
            && (mem_props->memoryTypes[i].propertyFlags & host_reqs_first) == host_reqs_first)
         return i;
   }

   if (host_reqs_first == 0)
   {
      printf("[Vulkan] Failed to find valid memory type. This should never happen.");
      abort();
   }

   return vulkan_find_memory_type_fallback(mem_props,
         device_reqs, host_reqs_second, 0);
}


static struct vk_texture vulkan_create_texture(vk_t *vk,
      struct vk_texture *old,
      unsigned width, unsigned height,
      VkFormat format,
      const void *initial,
      const VkComponentMapping *swizzle,
      enum vk_texture_type type)
{
   unsigned i;
   uint32_t buffer_width;
   struct vk_texture tex;
   VkImageCreateInfo info;
   VkFormat remap_tex_fmt;
   VkMemoryRequirements mem_reqs;
   VkSubresourceLayout layout;
   VkMemoryAllocateInfo alloc;
   VkBufferCreateInfo buffer_info;
   VkDevice device                      = vk->context->device;
   VkImageSubresource subresource       = { VK_IMAGE_ASPECT_COLOR_BIT };

   memset(&tex, 0, sizeof(tex));

   info.sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
   info.pNext                 = NULL;
   info.flags                 = 0;
   info.imageType             = VK_IMAGE_TYPE_2D;
   info.format                = format;
   info.extent.width          = width;
   info.extent.height         = height;
   info.extent.depth          = 1;
   info.mipLevels             = 1;
   info.arrayLayers           = 1;
   info.samples               = VK_SAMPLE_COUNT_1_BIT;
   info.tiling                = VK_IMAGE_TILING_OPTIMAL;
   info.usage                 = 0;
   info.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
   info.queueFamilyIndexCount = 0;
   info.pQueueFamilyIndices   = NULL;
   info.initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED;

   /* Align stride to 4 bytes to make sure we can use compute shader uploads without too many problems. */
   buffer_width                      = width * vulkan_format_to_bpp(format);
   buffer_width                      = (buffer_width + 3u) & ~3u;

   buffer_info.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   buffer_info.pNext                 = NULL;
   buffer_info.flags                 = 0;
   buffer_info.size                  = buffer_width * height;
   buffer_info.usage                 = 0;
   buffer_info.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
   buffer_info.queueFamilyIndexCount = 0;
   buffer_info.pQueueFamilyIndices   = NULL;

   remap_tex_fmt                     = VK_REMAP_TO_TEXFMT(format);

   /* Compatibility concern. Some Apple hardware does not support rgb565.
    * Use compute shader uploads instead.
    * If we attempt to use streamed texture, force staging path.
    * If we're creating fallback dynamic texture, force RGBA8888. */
   if (remap_tex_fmt != format)
   {
      if (type == VULKAN_TEXTURE_STREAMED)
         type        = VULKAN_TEXTURE_STAGING;
      else if (type == VULKAN_TEXTURE_DYNAMIC)
      {
         format      = remap_tex_fmt;
         info.format = format;
         info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      }
   }

   if (type == VULKAN_TEXTURE_STREAMED)
   {
      VkFormatProperties format_properties;
      const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                                          | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

      vkGetPhysicalDeviceFormatProperties(
            vk->context->gpu, format, &format_properties);

      if ((format_properties.linearTilingFeatures & required) != required)
      {
#ifdef VULKAN_DEBUG
         RARCH_DBG("[Vulkan] GPU does not support using linear images as textures. Falling back to copy path.\n");
#endif
         type = VULKAN_TEXTURE_STAGING;
      }
   }

   switch (type)
   {
      case VULKAN_TEXTURE_STATIC:
         /* For simplicity, always build mipmaps for
          * static textures, samplers can be used to enable it dynamically.
          */
         info.mipLevels     = vulkan_num_miplevels(width, height);
         tex.flags         |= VK_TEX_FLAG_MIPMAP;
         assert(initial && "Static textures must have initial data.\n");
         info.tiling        = VK_IMAGE_TILING_OPTIMAL;
         info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
         info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
         break;

      case VULKAN_TEXTURE_DYNAMIC:
         assert(!initial && "Dynamic textures must not have initial data.\n");
         info.tiling        = VK_IMAGE_TILING_OPTIMAL;
         info.usage        |= VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
         info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
         break;

      case VULKAN_TEXTURE_STREAMED:
         info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
         info.tiling        = VK_IMAGE_TILING_LINEAR;
         info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
         break;

      case VULKAN_TEXTURE_STAGING:
         buffer_info.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
         info.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
         info.tiling        = VK_IMAGE_TILING_LINEAR;
         break;

      case VULKAN_TEXTURE_READBACK:
         buffer_info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
         info.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
         info.tiling        = VK_IMAGE_TILING_LINEAR;
         break;
   }

   if (     (type != VULKAN_TEXTURE_STAGING)
         && (type != VULKAN_TEXTURE_READBACK))
   {
      vkCreateImage(device, &info, NULL, &tex.image);
      // vulkan_debug_mark_image(device, tex.image);
#if 0
      vulkan_track_alloc(tex.image);
#endif
      vkGetImageMemoryRequirements(device, tex.image, &mem_reqs);
   }
   else
   {
      /* Linear staging textures are not guaranteed to be supported,
       * use buffers instead. */
      vkCreateBuffer(device, &buffer_info, NULL, &tex.buffer);
      // vulkan_debug_mark_buffer(device, tex.buffer);
      vkGetBufferMemoryRequirements(device, tex.buffer, &mem_reqs);
   }

   alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   alloc.pNext           = NULL;
   alloc.allocationSize  = mem_reqs.size;
   alloc.memoryTypeIndex = 0;

   switch (type)
   {
      case VULKAN_TEXTURE_STATIC:
      case VULKAN_TEXTURE_DYNAMIC:
         alloc.memoryTypeIndex = vulkan_find_memory_type_fallback(
               &vk->context->memory_properties,
               mem_reqs.memoryTypeBits,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
         break;

      default:
         /* Try to find a memory type which is cached,
          * even if it means manual cache management. */
         alloc.memoryTypeIndex = vulkan_find_memory_type_fallback(
               &vk->context->memory_properties,
               mem_reqs.memoryTypeBits,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
               | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

         if ((vk->context->memory_properties.memoryTypes
                  [ alloc.memoryTypeIndex].propertyFlags
                  & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
            tex.flags |= VK_TEX_FLAG_NEED_MANUAL_CACHE_MANAGEMENT;

         /* If the texture is STREAMED and it's not DEVICE_LOCAL, we expect to hit a slower path,
          * so fallback to copy path. */
         if (      type == VULKAN_TEXTURE_STREAMED
               && (vk->context->memory_properties.memoryTypes[
                     alloc.memoryTypeIndex].propertyFlags
                   & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)
         {
            /* Recreate texture but for STAGING this time ... */
#ifdef VULKAN_DEBUG
            RARCH_DBG("[Vulkan] GPU supports linear images as textures, but not DEVICE_LOCAL. Falling back to copy path.\n");
#endif
            type                  = VULKAN_TEXTURE_STAGING;
            vkDestroyImage(device, tex.image, NULL);
            tex.image             = VK_NULL_HANDLE;
            info.initialLayout    = VK_IMAGE_LAYOUT_GENERAL;

            buffer_info.usage     = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            vkCreateBuffer(device, &buffer_info, NULL, &tex.buffer);
            // vulkan_debug_mark_buffer(device, tex.buffer);
            vkGetBufferMemoryRequirements(device, tex.buffer, &mem_reqs);

            alloc.allocationSize  = mem_reqs.size;
            alloc.memoryTypeIndex = vulkan_find_memory_type_fallback(
                    &vk->context->memory_properties,
                    mem_reqs.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                  | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
         }
         break;
   }

   /* We're not reusing the objects themselves. */
   if (old)
   {
      if (old->view != VK_NULL_HANDLE)
         vkDestroyImageView(vk->context->device, old->view, NULL);
      if (old->image != VK_NULL_HANDLE)
      {
         vkDestroyImage(vk->context->device, old->image, NULL);
#ifdef VULKAN_DEBUG_TEXTURE_ALLOC
         vulkan_track_dealloc(old->image);
#endif
      }
      if (old->buffer != VK_NULL_HANDLE)
         vkDestroyBuffer(vk->context->device, old->buffer, NULL);
   }

   /* We can pilfer the old memory and move it over to the new texture. */
   if (     old
         && old->memory_size >= mem_reqs.size
         && old->memory_type == alloc.memoryTypeIndex)
   {
      tex.memory      = old->memory;
      tex.memory_size = old->memory_size;
      tex.memory_type = old->memory_type;

      if (old->mapped)
         vkUnmapMemory(device, old->memory);

      old->memory     = VK_NULL_HANDLE;
   }
   else
   {
      vkAllocateMemory(device, &alloc, NULL, &tex.memory);
      // vulkan_debug_mark_memory(device, tex.memory);
      tex.memory_size = alloc.allocationSize;
      tex.memory_type = alloc.memoryTypeIndex;
   }

   if (old)
   {
      if (old->memory != VK_NULL_HANDLE)
         vkFreeMemory(device, old->memory, NULL);
      memset(old, 0, sizeof(*old));
   }

   if (tex.image)
      vkBindImageMemory(device, tex.image, tex.memory, 0);
   if (tex.buffer)
      vkBindBufferMemory(device, tex.buffer, tex.memory, 0);

   if (     type != VULKAN_TEXTURE_STAGING
         && type != VULKAN_TEXTURE_READBACK)
   {
      VkImageViewCreateInfo view;
      view.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view.pNext                           = NULL;
      view.flags                           = 0;
      view.image                           = tex.image;
      view.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
      view.format                          = format;
      if (swizzle)
         view.components                   = *swizzle;
      else
      {
         view.components.r                 = VK_COMPONENT_SWIZZLE_R;
         view.components.g                 = VK_COMPONENT_SWIZZLE_G;
         view.components.b                 = VK_COMPONENT_SWIZZLE_B;
         view.components.a                 = VK_COMPONENT_SWIZZLE_A;
      }
      view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      view.subresourceRange.baseMipLevel   = 0;
      view.subresourceRange.levelCount     = info.mipLevels;
      view.subresourceRange.baseArrayLayer = 0;
      view.subresourceRange.layerCount     = 1;

      vkCreateImageView(device, &view, NULL, &tex.view);
   }
   else
      tex.view        = VK_NULL_HANDLE;

   if (     tex.image
         && info.tiling == VK_IMAGE_TILING_LINEAR)
      vkGetImageSubresourceLayout(device, tex.image, &subresource, &layout);
   else if (tex.buffer)
   {
      layout.offset   = 0;
      layout.size     = buffer_info.size;
      layout.rowPitch = buffer_width;
   }
   else
      memset(&layout, 0, sizeof(layout));

   tex.stride = layout.rowPitch;
   tex.offset = layout.offset;
   tex.size   = layout.size;
   tex.layout = info.initialLayout;

   tex.width  = width;
   tex.height = height;
   tex.format = format;
   tex.type   = type;

   if (initial)
   {
      switch (type)
      {
         case VULKAN_TEXTURE_STREAMED:
         case VULKAN_TEXTURE_STAGING:
            {
               unsigned y;
               uint8_t *dst       = NULL;
               const uint8_t *src = NULL;
               void *ptr          = NULL;
               unsigned bpp       = vulkan_format_to_bpp(tex.format);
               unsigned stride    = tex.width * bpp;

               vkMapMemory(device, tex.memory, tex.offset, tex.size, 0, &ptr);

               dst                = (uint8_t*)ptr;
               src                = (const uint8_t*)initial;
               for (y = 0; y < tex.height; y++, dst += tex.stride, src += stride)
                  memcpy(dst, src, width * bpp);

               if (     (tex.flags & VK_TEX_FLAG_NEED_MANUAL_CACHE_MANAGEMENT)
                     && (tex.memory != VK_NULL_HANDLE))
                  VULKAN_SYNC_TEXTURE_TO_GPU(vk->context->device, tex.memory);
               vkUnmapMemory(device, tex.memory);
            }
            break;
         case VULKAN_TEXTURE_STATIC:
            {
               VkBufferImageCopy region;
               VkCommandBuffer staging;
               VkSubmitInfo submit_info;
               VkCommandBufferBeginInfo begin_info;
               VkCommandBufferAllocateInfo cmd_info;
               enum VkImageLayout layout_fmt =
                  (tex.flags & VK_TEX_FLAG_MIPMAP)
                  ? VK_IMAGE_LAYOUT_GENERAL
                  : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
               struct vk_texture tmp                = vulkan_create_texture(vk, NULL,
                     width, height, format, initial, NULL, VULKAN_TEXTURE_STAGING);

               cmd_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
               cmd_info.pNext                       = NULL;
               cmd_info.commandPool                 = vk->staging_pool;
               cmd_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
               cmd_info.commandBufferCount          = 1;

               vkAllocateCommandBuffers(vk->context->device,
                     &cmd_info, &staging);

               begin_info.sType                     = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
               begin_info.pNext                     = NULL;
               begin_info.flags                     = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
               begin_info.pInheritanceInfo          = NULL;

               vkBeginCommandBuffer(staging, &begin_info);

               /* If doing mipmapping on upload, keep in general
                * so we can easily do transfers to
                * and transfers from the images without having to
                * mess around with lots of extra transitions at
                * per-level granularity.
                */
               VULKAN_IMAGE_LAYOUT_TRANSITION(
                     staging,
                     tex.image,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     layout_fmt,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);

               memset(&region, 0, sizeof(region));
               region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
               region.imageSubresource.layerCount = 1;
               region.imageExtent.width           = width;
               region.imageExtent.height          = height;
               region.imageExtent.depth           = 1;

               vkCmdCopyBufferToImage(staging, tmp.buffer,
                     tex.image, layout_fmt, 1, &region);

               if (tex.flags & VK_TEX_FLAG_MIPMAP)
               {
                  for (i = 1; i < info.mipLevels; i++)
                  {
                     VkImageBlit blit_region;
                     unsigned src_width                        = MAX(width >> (i - 1), 1);
                     unsigned src_height                       = MAX(height >> (i - 1), 1);
                     unsigned target_width                     = MAX(width >> i, 1);
                     unsigned target_height                    = MAX(height >> i, 1);
                     memset(&blit_region, 0, sizeof(blit_region));

                     blit_region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                     blit_region.srcSubresource.mipLevel       = i - 1;
                     blit_region.srcSubresource.baseArrayLayer = 0;
                     blit_region.srcSubresource.layerCount     = 1;
                     blit_region.dstSubresource                = blit_region.srcSubresource;
                     blit_region.dstSubresource.mipLevel       = i;
                     blit_region.srcOffsets[1].x               = src_width;
                     blit_region.srcOffsets[1].y               = src_height;
                     blit_region.srcOffsets[1].z               = 1;
                     blit_region.dstOffsets[1].x               = target_width;
                     blit_region.dstOffsets[1].y               = target_height;
                     blit_region.dstOffsets[1].z               = 1;

                     /* Only injects execution and memory barriers,
                      * not actual transition. */
                     VULKAN_IMAGE_LAYOUT_TRANSITION(
                           staging,
                           tex.image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);

                     vkCmdBlitImage(
                           staging,
                           tex.image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           tex.image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           1,
                           &blit_region,
                           VK_FILTER_LINEAR);
                  }
               }

               /* Complete our texture. */
               VULKAN_IMAGE_LAYOUT_TRANSITION(
                     staging,
                     tex.image,
                     layout_fmt,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

               vkEndCommandBuffer(staging);
               submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
               submit_info.pNext                = NULL;
               submit_info.waitSemaphoreCount   = 0;
               submit_info.pWaitSemaphores      = NULL;
               submit_info.pWaitDstStageMask    = NULL;
               submit_info.commandBufferCount   = 1;
               submit_info.pCommandBuffers      = &staging;
               submit_info.signalSemaphoreCount = 0;
               submit_info.pSignalSemaphores    = NULL;

#ifdef HAVE_THREADS
               slock_lock(vk->context->queue_lock);
#endif
               vkQueueSubmit(vk->context->queue,
                     1, &submit_info, VK_NULL_HANDLE);

               /* TODO: Very crude, but texture uploads only happen
                * during init, so waiting for GPU to complete transfer
                * and blocking isn't a big deal. */
               vkQueueWaitIdle(vk->context->queue);
#ifdef HAVE_THREADS
               slock_unlock(vk->context->queue_lock);
#endif

               vkFreeCommandBuffers(vk->context->device,
                     vk->staging_pool, 1, &staging);
               vulkan_destroy_texture(
                     vk->context->device, &tmp);
               tex.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            break;
         case VULKAN_TEXTURE_DYNAMIC:
         case VULKAN_TEXTURE_READBACK:
            /* TODO/FIXME - stubs */
            break;
      }
   }

   return tex;
}

// static void vulkan_set_image(void *handle,
//       const struct retro_vulkan_image *image,
//       uint32_t num_semaphores,
//       const VkSemaphore *semaphores,
//       uint32_t src_queue_family)
// {
//    gfx_ctx_vulkan_data_t *vk              = (gfx_ctx_vulkan_data_t*)handle;

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
// }

static void vulkan_init_hw_render(gfx_ctx_vulkan_data_t *vk, struct retro_hw_render_callback *hwr, struct retro_hw_render_interface_vulkan **iface)
{
   if (hwr->context_type != RETRO_HW_CONTEXT_VULKAN)
      return;

   vk->flags                    |= VK_FLAG_HW_ENABLE;

   (*iface)->interface_type         = RETRO_HW_RENDER_INTERFACE_VULKAN;
   (*iface)->interface_version      = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION;
   (*iface)->instance               = vk->context.instance; // vk->context->instance;
   (*iface)->gpu                    = vk->context.gpu;
   (*iface)->device                 = vk->context.device;

   (*iface)->queue                  = vk->context.queue;
   (*iface)->queue_index            = vk->context.graphics_queue_index;

   (*iface)->handle                 = vk;
   // iface->set_image              = vulkan_set_image;
   // iface->get_sync_index         = vulkan_get_sync_index;
   // iface->get_sync_index_mask    = vulkan_get_sync_index_mask;
   // iface->wait_sync_index        = vulkan_wait_sync_index;
   // iface->set_command_buffers    = vulkan_set_command_buffers;
   // iface->lock_queue             = vulkan_lock_queue;
   // iface->unlock_queue           = vulkan_unlock_queue;
   // iface->set_signal_semaphore   = vulkan_set_signal_semaphore;

   (*iface)->get_device_proc_addr   = vkGetDeviceProcAddr;
   (*iface)->get_instance_proc_addr = vulkan_symbol_wrapper_instance_proc_addr();
}

// TODO implement the init and mabe work!!!!!!
static void *vulkan_init(const video_info_t *video,
      input_driver_t **input,
      void **input_data)
{
   unsigned full_x, full_y;
   unsigned win_width;
   unsigned win_height;
   unsigned mode_width                = 0;
   unsigned mode_height               = 0;
   int interval                       = 0;
   unsigned temp_width                = 0;
   unsigned temp_height               = 0;
   bool force_fullscreen              = false;
   const gfx_ctx_driver_t *ctx_driver = NULL;
   settings_t *settings               = config_get_ptr();

   vk_t *vk                           = (vk_t*)calloc(1, sizeof(*vk));
   if (!vk)
      return NULL;
   ctx_driver                         = vulkan_get_context(vk, settings);
   if (!ctx_driver)
   {
      printf("[Vulkan] Failed to get Vulkan context.\n");
      goto error;
   }

   vk->video                          = *video;
   vk->ctx_driver                     = ctx_driver;

   video_context_driver_set((const gfx_ctx_driver_t*)ctx_driver);

   RARCH_DBG("[Vulkan] Found vulkan context: \"%s\".\n", ctx_driver->ident);

   if (vk->ctx_driver->get_video_size)
      vk->ctx_driver->get_video_size(vk->ctx_data,
            &mode_width, &mode_height);

   if (!video->fullscreen && !vk->ctx_driver->has_windowed)
   {
      RARCH_DBG("[Vulkan] Config requires windowed mode, but context driver does not support it. "
                "Forcing fullscreen for this session.\n");
      force_fullscreen = true;
   }

   full_x                             = mode_width;
   full_y                             = mode_height;
   mode_width                         = 0;
   mode_height                        = 0;

   RARCH_DBG("[Vulkan] Detecting screen resolution: %ux%u.\n", full_x, full_y);
   interval = video->vsync ? video->swap_interval : 0;

   if (ctx_driver->swap_interval)
   {
      bool adaptive_vsync_enabled            = video_driver_test_all_flags(
            GFX_CTX_FLAGS_ADAPTIVE_VSYNC) && video->adaptive_vsync;
      if (adaptive_vsync_enabled && interval == 1)
         interval = -1;
      ctx_driver->swap_interval(vk->ctx_data, interval);
   }

   win_width  = video->width;
   win_height = video->height;

   if (video->fullscreen && (win_width == 0) && (win_height == 0))
   {
      win_width  = full_x;
      win_height = full_y;
   }
   /* If fullscreen had to be forced, video->width/height is incorrect */
   else if (force_fullscreen)
   {
      win_width  = settings->uints.video_fullscreen_x;
      win_height = settings->uints.video_fullscreen_y;
   }

   if (     !vk->ctx_driver->set_video_mode
         || !vk->ctx_driver->set_video_mode(vk->ctx_data,
            win_width, win_height, (video->fullscreen || force_fullscreen)))
   {
      RARCH_ERR("[Vulkan] Failed to set video mode.\n");
      goto error;
   }

   if (vk->ctx_driver->get_video_size)
      vk->ctx_driver->get_video_size(vk->ctx_data,
            &mode_width, &mode_height);

   temp_width  = mode_width;
   temp_height = mode_height;

   if (temp_width != 0 && temp_height != 0)
      video_driver_set_size(temp_width, temp_height);
   video_driver_get_size(&temp_width, &temp_height);
   vk->video_width       = temp_width;
   vk->video_height      = temp_height;
   vk->translate_x       = 0.0;
   vk->translate_y       = 0.0;

   RARCH_LOG("[Vulkan] Using resolution %ux%u.\n", temp_width, temp_height);

   if (!vk->ctx_driver || !vk->ctx_driver->get_context_data)
   {
      RARCH_ERR("[Vulkan] Failed to get context data.\n");
      goto error;
   }

   *(void**)&vk->context = vk->ctx_driver->get_context_data(vk->ctx_data);

   if (video->vsync)
      vk->flags         |=  VK_FLAG_VSYNC;
   else
      vk->flags         &= ~VK_FLAG_VSYNC;
   if (video->fullscreen || force_fullscreen)
      vk->flags         |=  VK_FLAG_FULLSCREEN;
   else
      vk->flags         &= ~VK_FLAG_FULLSCREEN;
   vk->tex_w             = RARCH_SCALE_BASE * video->input_scale;
   vk->tex_h             = RARCH_SCALE_BASE * video->input_scale;
   vk->tex_fmt           = video->rgb32 ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R5G6B5_UNORM_PACK16;
   if (video->force_aspect)
      vk->flags         |=  VK_FLAG_KEEP_ASPECT;
   else
      vk->flags         &= ~VK_FLAG_KEEP_ASPECT;
   printf("[Vulkan] Using %s format.\n", video->rgb32 ? "BGRA8888" : "RGB565");

   /* Set the viewport to fix recording, since it needs to know
    * the viewport sizes before we start running. */
   vulkan_set_viewport(vk, temp_width, temp_height, false, true);

   vulkan_init_hw_render(vk);
   if (vk->context)
   {
      int i;
      static const VkDescriptorPoolSize pool_sizes[4] = {
         { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VULKAN_DESCRIPTOR_MANAGER_BLOCK_SETS },
         { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VULKAN_DESCRIPTOR_MANAGER_BLOCK_SETS * 2 },
         { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          VULKAN_DESCRIPTOR_MANAGER_BLOCK_SETS },
         { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         VULKAN_DESCRIPTOR_MANAGER_BLOCK_SETS },
      };

      vulkan_init_static_resources(vk);

      vk->num_swapchain_images = vk->context->num_swapchain_images;

      vulkan_init_render_pass(vk);
      vulkan_init_framebuffers(vk);
      vulkan_init_pipelines(vk);
      vulkan_init_samplers(vk);
      vulkan_init_textures(vk);

      for (i = 0; i < (int) vk->num_swapchain_images; i++)
      {
         VkCommandPoolCreateInfo pool_info;
         VkCommandBufferAllocateInfo info;

         vk->swapchain[i].descriptor_manager =
            vulkan_create_descriptor_manager(
                  vk->context->device,
                  pool_sizes, 4, vk->pipelines.set_layout);
         vk->swapchain[i].vbo                =
            vulkan_buffer_chain_init(
               VULKAN_BUFFER_BLOCK_SIZE, 16,
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
         vk->swapchain[i].ubo                =
            vulkan_buffer_chain_init(
               VULKAN_BUFFER_BLOCK_SIZE,
               vk->context->gpu_properties.limits.minUniformBufferOffsetAlignment,
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

         pool_info.sType            =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
         pool_info.pNext            = NULL;
         /* RESET_COMMAND_BUFFER_BIT allows command buffer to be reset. */
         pool_info.flags            =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
         pool_info.queueFamilyIndex = vk->context->graphics_queue_index;

         vkCreateCommandPool(vk->context->device,
               &pool_info, NULL, &vk->swapchain[i].cmd_pool);

         info.sType                 =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
         info.pNext                 = NULL;
         info.commandPool           = vk->swapchain[i].cmd_pool;
         info.level                 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
         info.commandBufferCount    = 1;

         vkAllocateCommandBuffers(vk->context->device,
               &info, &vk->swapchain[i].cmd);
      }
   }

   if (!vulkan_init_filter_chain(vk))
   {
      RARCH_ERR("[Vulkan] Failed to init filter chain.\n");
      goto error;
   }

   if (vk->ctx_driver->input_driver)
   {
      const char *joypad_name = settings->arrays.input_joypad_driver;
      vk->ctx_driver->input_driver(
            vk->ctx_data, joypad_name,
            input, input_data);
   }

   if (video->font_enable)
      font_driver_init_osd(vk,
            video,
            false,
            video->is_threaded,
            FONT_DRIVER_RENDER_VULKAN_API);

   /* The MoltenVK driver needs this, particularly after driver reinit
      Also it is required for HDR to not break during reinit, while not ideal it
      is the simplest solution unless reinit tracking is done */
   vk->flags |= VK_FLAG_SHOULD_RESIZE;

   vulkan_init_readback(vk, settings->bools.video_gpu_record);
   return vk;

   error:
      vulkan_free(vk);
      return NULL;
}


// void vulkan_context_destroy(gfx_ctx_vulkan_data_t *vk,
//       bool destroy_surface)
// {
//    video_driver_state_t *video_st = video_state_get_ptr();
//    uint32_t video_st_flags        = 0;
//    if (!vk->context.instance)
//       return;

//    if (vk->context.device)
//       vkDeviceWaitIdle(vk->context.device);

//    vulkan_destroy_swapchain(vk);

//    if (     destroy_surface
//          && (vk->vk_surface != VK_NULL_HANDLE))
//    {
//       vkDestroySurfaceKHR(vk->context.instance,
//             vk->vk_surface, NULL);
//       vk->vk_surface = VK_NULL_HANDLE;
//    }

//    video_st_flags              = video_st->flags;

//    if (video_st_flags & VIDEO_FLAG_CACHE_CONTEXT)
//    {
//       cached_device_vk         = vk->context.device;
//       cached_instance_vk       = vk->context.instance;
//       cached_destroy_device_vk = vk->context.destroy_device;
//    }
//    else
//    {
//       if (vk->context.device)
//       {
//          vkDestroyDevice(vk->context.device, NULL);
//          vk->context.device = NULL;
//       }

//       if (vk->context.instance)
//       {
//          if (vk->context.destroy_device)
//             vk->context.destroy_device();

//          vkDestroyInstance(vk->context.instance, NULL);
//          vk->context.instance = NULL;

//          if (vulkan_library)
//          {
//             dylib_close(vulkan_library);
//             vulkan_library = NULL;
//          }
//       }
//    }

//    video_driver_set_gpu_api_devices(GFX_CTX_VULKAN_API, NULL);
//    if (vk->gpu_list)
//    {
//       string_list_free(vk->gpu_list);
//       vk->gpu_list = NULL;
//    }
// }

// void vulkan_acquire_next_image(gfx_ctx_vulkan_data_t *vk)
// {
//    unsigned index;
//    VkFenceCreateInfo fence_info;
//    VkSemaphoreCreateInfo sem_info;
//    VkResult err                   = VK_SUCCESS;
//    VkFence fence                  = VK_NULL_HANDLE;
//    VkSemaphore semaphore          = VK_NULL_HANDLE;
//    bool is_retrying               = false;

//    fence_info.sType               = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
//    fence_info.pNext               = NULL;
//    fence_info.flags               = 0;

//    sem_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
//    sem_info.pNext                 = NULL;
//    sem_info.flags                 = 0;

// retry:
//    if (vk->swapchain == VK_NULL_HANDLE)
//    {
//       /* We don't have a swapchain, try to create one now. */
//       if (!vulkan_create_swapchain(vk, vk->context.swapchain_width,
//                vk->context.swapchain_height, vk->context.swap_interval))
//       {
//          retro_sleep(20);
//          return;
//       }

//       if (vk->swapchain == VK_NULL_HANDLE)
//       {
//          /* We still don't have a swapchain, so just fake it ... */
//          vk->context.current_swapchain_index = 0;
//          vk->context.current_frame_index     = 0;
//          vulkan_acquire_clear_fences(vk);
//          vulkan_acquire_wait_fences(vk);
//          vk->context.flags                  |= VK_CTX_FLAG_INVALID_SWAPCHAIN;
//          return;
//       }
//    }

//    retro_assert(!(vk->context.flags & VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN));

//    if (vk->flags & VK_DATA_FLAG_EMULATING_MAILBOX)
//    {
//       /* Non-blocking acquire. If we don't get a swapchain frame right away,
//        * just skip rendering to the swapchain this frame, similar to what
//        * MAILBOX would do. */
//       if (vk->mailbox.swapchain == VK_NULL_HANDLE)
//          err   = VK_ERROR_OUT_OF_DATE_KHR;
//       else
//          err   = vulkan_emulated_mailbox_acquire_next_image(
//                &vk->mailbox, &vk->context.current_swapchain_index);
//    }
//    else
//    {
//       if (vk->flags & VK_DATA_FLAG_USE_WSI_SEMAPHORE)
//           semaphore = vulkan_get_wsi_acquire_semaphore(&vk->context);
//       else
//           vkCreateFence(vk->context.device, &fence_info, NULL, &fence);

//       err = vkAcquireNextImageKHR(vk->context.device,
//             vk->swapchain, UINT64_MAX,
//             semaphore, fence, &vk->context.current_swapchain_index);
//    }

//    if (err == VK_SUCCESS || err == VK_SUBOPTIMAL_KHR)
//    {
//       if (fence != VK_NULL_HANDLE)
//          vkWaitForFences(vk->context.device, 1, &fence, true, UINT64_MAX);
//       vk->context.flags |= VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN;

//       if (vk->context.swapchain_acquire_semaphore)
//       {
// #ifdef HAVE_THREADS
//          slock_lock(vk->context.queue_lock);
// #endif
//          vkDeviceWaitIdle(vk->context.device);
//          vkDestroySemaphore(vk->context.device, vk->context.swapchain_acquire_semaphore, NULL);
// #ifdef HAVE_THREADS
//          slock_unlock(vk->context.queue_lock);
// #endif
//       }
//       vk->context.swapchain_acquire_semaphore = semaphore;
//    }
//    else
//    {
//       vk->context.flags &= ~VK_CTX_FLAG_HAS_ACQUIRED_SWAPCHAIN;
//       if (semaphore)
//       {
//          struct vulkan_context *ctx = &vk->context;
//          VkSemaphore sem            = semaphore;
//          assert(ctx->num_recycled_acquire_semaphores < VULKAN_MAX_SWAPCHAIN_IMAGES);
//          ctx->swapchain_recycled_semaphores[ctx->num_recycled_acquire_semaphores++] = sem;
//       }
//    }

// #ifdef WSI_HARDENING_TEST
//    trigger_spurious_error_vkresult(&err);
// #endif

//    if (fence != VK_NULL_HANDLE)
//       vkDestroyFence(vk->context.device, fence, NULL);

//    switch (err)
//    {
//       case VK_NOT_READY:
//       case VK_TIMEOUT:
//       case VK_SUBOPTIMAL_KHR:
//          /* Do nothing. */
//          break;
//       case VK_ERROR_OUT_OF_DATE_KHR:
//          /* Throw away the old swapchain and try again. */
//          vulkan_destroy_swapchain(vk);
//          /* Swapchain out of date, trying to create new one ... */
//          if (is_retrying)
//          {
//             retro_sleep(10);
//          }
//          else
//             is_retrying = true;
//          vulkan_acquire_clear_fences(vk);
//          goto retry;
//       default:
//          if (err != VK_SUCCESS)
//          {
//             /* We are screwed, don't try anymore. Maybe it will work later. */
//             vulkan_destroy_swapchain(vk);
//             printf("[Vulkan] Failed to acquire from swapchain (err = %d).\n",
//                   (int)err);
//             if (err == VK_ERROR_SURFACE_LOST_KHR)
//                printf("[Vulkan] Got VK_ERROR_SURFACE_LOST_KHR.\n");
//             /* Force driver to reset swapchain image handles. */
//             vk->context.flags |= VK_CTX_FLAG_INVALID_SWAPCHAIN;
//             vulkan_acquire_clear_fences(vk);
//             return;
//          }
//          break;
//    }

//    index = vk->context.current_swapchain_index;
//    if (vk->context.swapchain_semaphores[index] == VK_NULL_HANDLE)
//       vkCreateSemaphore(vk->context.device, &sem_info,
//             NULL, &vk->context.swapchain_semaphores[index]);
//    vulkan_acquire_wait_fences(vk);
// }