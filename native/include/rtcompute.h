// rtcompute — plain C API over Slang + slang-rhi, consumed by Swift (module CRTCompute).
//
// Conventions:
// - Functions returning rtc_status set a thread-local message readable with rtc_last_error() on failure.
// - Handles are owned by the caller and released with the matching *_destroy function.
// - Dispatches are asynchronous; rtc_buffer_download and rtc_device_wait synchronise with the device.
#ifndef RTCOMPUTE_H
#define RTCOMPUTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define RTC_API __attribute__((visibility("default")))
#else
#define RTC_API
#endif

typedef struct rtc_device_t* rtc_device;
typedef struct rtc_buffer_t* rtc_buffer;
typedef struct rtc_kernel_t* rtc_kernel;

typedef enum rtc_status {
    RTC_OK = 0,
    RTC_ERROR = 1,
} rtc_status;

typedef enum rtc_backend {
    RTC_BACKEND_CPU = 1,
    RTC_BACKEND_CUDA = 2,
} rtc_backend;

typedef struct rtc_device_desc {
    rtc_backend backend;
    /// Directories searched for `import`ed / loaded Slang modules.
    const char* const* search_paths;
    uint32_t search_path_count;
    /// Directory for the persistent compiled-kernel cache; NULL disables it.
    const char* cache_dir;
    /// Forward slang-rhi warnings and errors to stderr.
    int verbose;
} rtc_device_desc;

RTC_API const char* rtc_last_error(void);

RTC_API rtc_status rtc_device_create(const rtc_device_desc* desc, rtc_device* out_device);
RTC_API void rtc_device_destroy(rtc_device device);
RTC_API rtc_backend rtc_device_backend(rtc_device device);
/// Adapter name, e.g. "NVIDIA GeForce RTX 3090". Valid for the device's lifetime.
RTC_API const char* rtc_device_name(rtc_device device);
/// Blocks until all submitted work has finished.
RTC_API rtc_status rtc_device_wait(rtc_device device);

/// Structured buffer of `size` bytes with elements of `element_size` bytes. `initial_data` may be NULL.
RTC_API rtc_status rtc_buffer_create(
    rtc_device device, uint64_t size, uint32_t element_size, const void* initial_data, rtc_buffer* out_buffer);
RTC_API void rtc_buffer_destroy(rtc_buffer buffer);
RTC_API uint64_t rtc_buffer_size(rtc_buffer buffer);
RTC_API rtc_status rtc_buffer_upload(rtc_buffer buffer, uint64_t offset, uint64_t size, const void* data);
/// Waits for pending work that writes the buffer, then copies to host memory.
RTC_API rtc_status rtc_buffer_download(rtc_buffer buffer, uint64_t offset, uint64_t size, void* out_data);

/// Loads compute entry point `entry_point` from module `module_name`, found on the search paths.
RTC_API rtc_status rtc_kernel_load(
    rtc_device device, const char* module_name, const char* entry_point, rtc_kernel* out_kernel);
/// Same, compiling `source` directly. `path` is only used for diagnostics and relative imports.
RTC_API rtc_status rtc_kernel_load_source(
    rtc_device device, const char* module_name, const char* path, const char* source, const char* entry_point,
    rtc_kernel* out_kernel);
RTC_API void rtc_kernel_destroy(rtc_kernel kernel);
RTC_API void rtc_kernel_thread_group_size(rtc_kernel kernel, uint32_t out_size[3]);

typedef enum rtc_arg_kind {
    RTC_ARG_BUFFER = 0, ///< binds `buffer`
    RTC_ARG_DATA = 1,   ///< copies `size` bytes from `data` (scalars, vectors, plain structs)
} rtc_arg_kind;

/// One shader parameter, looked up by name among globals first, then entry-point parameters.
typedef struct rtc_arg {
    const char* name;
    rtc_arg_kind kind;
    rtc_buffer buffer;
    const void* data;
    uint64_t size;
} rtc_arg;

/// Launches enough thread groups to cover `threads_x * threads_y * threads_z` threads.
RTC_API rtc_status rtc_dispatch(
    rtc_device device, rtc_kernel kernel, const rtc_arg* args, uint32_t arg_count, uint32_t threads_x,
    uint32_t threads_y, uint32_t threads_z);

#ifdef __cplusplus
}
#endif

#endif // RTCOMPUTE_H
