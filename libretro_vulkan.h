/* Copyright (C) 2010-2016 The RetroArch team
 *
 * ---------------------------------------------------------------------------------------------
 * The following license statement only applies to this libretro API header (libretro_vulkan.h)
 * ---------------------------------------------------------------------------------------------
 *
 * Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation files (the
 * "Software"),
 * to deal in the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIBRETRO_VULKAN_H__
#define LIBRETRO_VULKAN_H__

#include <vulkan/vulkan.h>
#include <libretro.h>

#define RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION 5
#define RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION 1

struct retro_vulkan_image
{
  VkImageView image_view;
  VkImageLayout image_layout;
  VkImageViewCreateInfo create_info;
};

typedef void (*retro_vulkan_set_image_t)(void* handle, const struct retro_vulkan_image* image,
                                         uint32_t num_semaphores, const VkSemaphore* semaphores,
                                         uint32_t src_queue_family);

typedef uint32_t (*retro_vulkan_get_sync_index_t)(void* handle);
typedef uint32_t (*retro_vulkan_get_sync_index_mask_t)(void* handle);
typedef void (*retro_vulkan_set_command_buffers_t)(void* handle, uint32_t num_cmd,
                                                   const VkCommandBuffer* cmd);
typedef void (*retro_vulkan_wait_sync_index_t)(void* handle);
typedef void (*retro_vulkan_lock_queue_t)(void* handle);
typedef void (*retro_vulkan_unlock_queue_t)(void* handle);
typedef void (*retro_vulkan_set_signal_semaphore_t)(void* handle, VkSemaphore semaphore);

typedef const VkApplicationInfo* (*retro_vulkan_get_application_info_t)(void);

struct retro_vulkan_context
{
  VkPhysicalDevice gpu;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;
  VkQueue presentation_queue;
  uint32_t presentation_queue_family_index;
};

typedef bool (*retro_vulkan_create_device_t)(
    struct retro_vulkan_context* context, VkInstance instance, VkPhysicalDevice gpu,
    VkSurfaceKHR surface, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const char** required_device_extensions, unsigned num_required_device_extensions,
    const char** required_device_layers, unsigned num_required_device_layers,
    const VkPhysicalDeviceFeatures* required_features);

typedef void (*retro_vulkan_destroy_device_t)(void);

/* Note on thread safety:
 * The Vulkan API is heavily designed around multi-threading, and
 * the libretro interface for it should also be threading friendly.
 * A core should be able to build command buffers and submit
 * command buffers to the GPU from any thread.
 */

struct retro_hw_render_context_negotiation_interface_vulkan
{
  /* Must be set to RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN. */
  enum retro_hw_render_context_negotiation_interface_type interface_type;
  /* Must be set to RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION. */
  unsigned interface_version;

  /* If non-NULL, returns a VkApplicationInfo struct that the frontend can use instead of
   * its "default" application info.
   */
  retro_vulkan_get_application_info_t get_application_info;

  /* If non-NULL, the libretro core will choose one or more physical devices,
   * create one or more logical devices and create one or more queues.
   * The core must prepare a designated PhysicalDevice, Device, Queue and queue family index
   * which the frontend will use for its internal operation.
   *
   * If gpu is not VK_NULL_HANDLE, the physical device provided to the frontend must be this
   * PhysicalDevice.
   * The core is still free to use other physical devices.
   *
   * The frontend will request certain extensions and layers for a device which is created.
   * The core must ensure that the queue and queue_family_index support GRAPHICS and COMPUTE.
   *
   * If surface is not VK_NULL_HANDLE, the core must consider presentation when creating the queues.
   * If presentation to "surface" is supported on the queue, presentation_queue must be equal to
   * queue.
   * If not, a second queue must be provided in presentation_queue and presentation_queue_index.
   * If surface is not VK_NULL_HANDLE, the instance from frontend will have been created with
   * supported for
   * VK_KHR_surface extension.
   *
   * The core is free to set its own queue priorities.
   * Device provided to frontend is owned by the frontend, but any additional device resources must
   * be freed by core
   * in destroy_device callback.
   *
   * If this function returns true, a PhysicalDevice, Device and Queues are initialized.
   * If false, none of the above have been initialized and the frontend will attempt
   * to fallback to "default" device creation, as if this function was never called.
   */
  retro_vulkan_create_device_t create_device;

  /* If non-NULL, this callback is called similar to context_destroy for HW_RENDER_INTERFACE.
   * However, it will be called even if context_reset was not called.
   * This can happen if the context never succeeds in being created.
   * destroy_device will always be called before the VkInstance
   * of the frontend is destroyed if create_device was called successfully so that the core has a
   * chance of
   * tearing down its own device resources.
   *
   * Only auxillary resources should be freed here, i.e. resources which are not part of
   * retro_vulkan_context.
   */
  retro_vulkan_destroy_device_t destroy_device;
};

// struct retro_hw_render_interface_vulkan
// {
//   enum retro_hw_render_interface_type interface_type;
//   unsigned interface_version;
//   void* handle;
//   VkInstance instance;
//   VkPhysicalDevice gpu;
//   VkDevice device;
//   PFN_vkGetDeviceProcAddr get_device_proc_addr;
//   PFN_vkGetInstanceProcAddr get_instance_proc_addr;
//   VkQueue queue;
//   unsigned queue_index;
//   retro_vulkan_set_image_t set_image;
//   retro_vulkan_get_sync_index_t get_sync_index;
//   retro_vulkan_get_sync_index_mask_t get_sync_index_mask;
//   retro_vulkan_set_command_buffers_t set_command_buffers;
//   retro_vulkan_wait_sync_index_t wait_sync_index;
//   retro_vulkan_lock_queue_t lock_queue;
//   retro_vulkan_unlock_queue_t unlock_queue;
//   retro_vulkan_set_signal_semaphore_t set_signal_semaphore;
// };

#endif