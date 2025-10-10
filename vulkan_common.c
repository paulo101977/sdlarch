#include "vulkan_common.h"
#include "vulkan_symbol_wrapper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "string_common.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

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

   PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices =
      (PFN_vkEnumeratePhysicalDevices) vkGetInstanceProcAddr(vk->context.instance, "vkEnumeratePhysicalDevices");
   PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceProperties");

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
   printf("g_iface cached_device_vk >>>>>>>>>>>>>>>>>>%d\n", cached_device_vk);

   if (!cached_device_vk && g_iface && g_iface->create_device)
   {
      printf("LINE %d\n", __LINE__);
      struct retro_vulkan_context context = { 0 };

      bool ret = false;

      if (     (g_iface->interface_version >= 2)
            &&  g_iface->create_device2)
      {
         printf("LINE %d\n", __LINE__);
         PFN_vkGetInstanceProcAddr fn = vulkan_symbol_wrapper_instance_proc_addr();

         ret = g_iface->create_device2(&context, vk->context.instance,
               vk->context.gpu,
               vk->vk_surface,
               fn,
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
         vulkan_symbol_wrapper_init(vkGetInstanceProcAddr);
         PFN_vkGetInstanceProcAddr fn = vulkan_symbol_wrapper_instance_proc_addr();

         printf("g_iface fn >>>>>>>>>>>>>>>>>> %p\n", fn);
         printf("g_iface vk->context.gpu >>>>>>>>>>>>>>>>>> %p\n", vk->context.gpu);
         printf("g_iface vk->vk_surface >>>>>>>>>>>>>>>>>> %p\n", vk->vk_surface);
         printf("g_iface vk->context.instance >>>>>>>>>>>>>>>>>> %p\n", vk->context.instance);
         printf("g_iface ARRAY_SIZE >>>>>>>>>>>>>>>>>> %d\n", ARRAY_SIZE(vulkan_device_extensions));
         printf("g_iface vulkan_device_extensions >>>>>>>>>>>>>>>>>> %s\n", vulkan_device_extensions[0]);

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

   PFN_vkGetDeviceProcAddr vkGetPhysicalDeviceMemoryProperties =
      (PFN_vkGetDeviceProcAddr) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceMemoryProperties");
   PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties) vkGetInstanceProcAddr(vk->context.instance, "vkGetPhysicalDeviceProperties");

   vkGetPhysicalDeviceProperties(vk->context.gpu,
         &vk->context.gpu_properties);
   vkGetPhysicalDeviceMemoryProperties(vk->context.gpu,
         &vk->context.memory_properties);


   printf("[Vulkan] Using GPU: \"%s\".\n", vk->context.gpu_properties.deviceName);

   {
      char version_str[128];
      size_t _len            = snprintf(version_str      , sizeof(version_str)      , "%u", VK_VERSION_MAJOR(vk->context.gpu_properties.apiVersion));
      version_str[  _len]    = '.';
      version_str[++_len]    = '\0';
      _len                  += snprintf(version_str + _len, sizeof(version_str) - _len, "%u", VK_VERSION_MINOR(vk->context.gpu_properties.apiVersion));
      version_str[  _len]    = '.';
      version_str[++_len]    = '\0';
      snprintf(version_str + _len, sizeof(version_str) - _len, "%u", VK_VERSION_PATCH(vk->context.gpu_properties.apiVersion));
      video_driver_set_gpu_api_version_string(version_str);
   }

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

   return true;
}