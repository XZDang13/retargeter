// rtcompute — implementation of the C API in rtcompute.h on top of slang-rhi.

#include "rtcompute.h"

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include <dlfcn.h>
#include <pthread.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

using namespace rhi;
namespace fs = std::filesystem;

namespace {

thread_local std::string g_lastError;

rtc_status fail(std::string message)
{
    g_lastError = std::move(message);
    return RTC_ERROR;
}

std::string blobString(ISlangBlob* blob)
{
    if (!blob)
        return {};
    return std::string(static_cast<const char*>(blob->getBufferPointer()), blob->getBufferSize());
}

rtc_status failWithDiagnostics(const std::string& what, ISlangBlob* diagnostics)
{
    std::string message = what;
    std::string diag = blobString(diagnostics);
    if (!diag.empty())
        message += ":\n" + diag;
    return fail(message);
}

// Compiled kernels on disk, one file per key: <fnv1a64(key)>.bin = [u64 key size][key][data].
// The stored key is compared on read, so hash collisions are harmless.
class FileCache : public IPersistentCache
{
public:
    explicit FileCache(fs::path dir)
        : m_dir(std::move(dir))
    {
        std::error_code ec;
        fs::create_directories(m_dir, ec);
    }

    SLANG_NO_THROW Result SLANG_MCALL writeCache(ISlangBlob* key, ISlangBlob* data) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        fs::path file = pathFor(key);
        fs::path tmp = file;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                return SLANG_FAIL;
            uint64_t keySize = key->getBufferSize();
            out.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));
            out.write(static_cast<const char*>(key->getBufferPointer()), std::streamsize(keySize));
            out.write(static_cast<const char*>(data->getBufferPointer()), std::streamsize(data->getBufferSize()));
            if (!out)
                return SLANG_FAIL;
        }
        std::error_code ec;
        fs::rename(tmp, file, ec);
        return ec ? SLANG_FAIL : SLANG_OK;
    }

    SLANG_NO_THROW Result SLANG_MCALL queryCache(ISlangBlob* key, ISlangBlob** outData) override
    {
        *outData = nullptr;
        std::lock_guard<std::mutex> lock(m_mutex);
        std::ifstream in(pathFor(key), std::ios::binary);
        if (!in)
            return SLANG_E_NOT_FOUND;
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        uint64_t keySize = 0;
        if (bytes.size() < sizeof(keySize))
            return SLANG_E_NOT_FOUND;
        std::memcpy(&keySize, bytes.data(), sizeof(keySize));
        size_t header = sizeof(keySize) + keySize;
        if (keySize != key->getBufferSize() || bytes.size() < header ||
            std::memcmp(bytes.data() + sizeof(keySize), key->getBufferPointer(), keySize) != 0)
            return SLANG_E_NOT_FOUND;
        return getRHI()->createBlob(bytes.data() + header, bytes.size() - header, outData);
    }

    SLANG_NO_THROW Result SLANG_MCALL queryInterface(const SlangUUID& uuid, void** outObject) override
    {
        if (uuid == IPersistentCache::getTypeGuid() || uuid == ISlangUnknown::getTypeGuid())
        {
            *outObject = static_cast<IPersistentCache*>(this);
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }

    // Lifetime is owned by rtc_device_t, which outlives the rhi device.
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 2; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 2; }

private:
    fs::path pathFor(ISlangBlob* key) const
    {
        uint64_t hash = 1469598103934665603ull;
        auto bytes = static_cast<const uint8_t*>(key->getBufferPointer());
        for (size_t i = 0; i < key->getBufferSize(); ++i)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
        char name[32];
        std::snprintf(name, sizeof(name), "%016llx.bin", static_cast<unsigned long long>(hash));
        return m_dir / name;
    }

    fs::path m_dir;
    std::mutex m_mutex;
};

class StderrDebugCallback : public IDebugCallback
{
public:
    SLANG_NO_THROW void SLANG_MCALL handleMessage(DebugMessageType type, DebugMessageSource, const char* message)
        override
    {
        if (type == DebugMessageType::Info)
            return;
        std::fprintf(stderr, "[rtcompute] %s: %s\n", type == DebugMessageType::Error ? "error" : "warning", message);
    }
};

StderrDebugCallback g_stderrCallback;

// CUDA contexts are current per thread, and slang-rhi does not make its context current on every path
// (e.g. queue polling). Every API call on a CUDA device therefore pushes the device's context for its
// duration, which makes the C API usable from any thread (one call at a time per device).
struct CudaDriver
{
    using PushFn = int (*)(void*);
    using PopFn = int (*)(void**);
    using CopyFn = int (*)(unsigned long long, const void*, size_t, void*);
    PushFn push = nullptr;
    PopFn pop = nullptr;
    CopyFn copyToDeviceAsync = nullptr;   // cuMemcpyHtoDAsync_v2

    CudaDriver()
    {
        if (void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL))
        {
            push = reinterpret_cast<PushFn>(dlsym(lib, "cuCtxPushCurrent_v2"));
            pop = reinterpret_cast<PopFn>(dlsym(lib, "cuCtxPopCurrent_v2"));
            copyToDeviceAsync = reinterpret_cast<CopyFn>(dlsym(lib, "cuMemcpyHtoDAsync_v2"));
        }
    }
};

CudaDriver& cudaDriver()
{
    static CudaDriver driver;
    return driver;
}

class ContextScope
{
public:
    explicit ContextScope(void* context)
        : m_active(context && cudaDriver().push && cudaDriver().pop && cudaDriver().push(context) == 0)
    {
    }
    ~ContextScope()
    {
        if (m_active)
        {
            void* popped = nullptr;
            cudaDriver().pop(&popped);
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;

private:
    bool m_active;
};

// Slang compilation recurses deeply; worker threads (e.g. an HTTP server's) have small stacks. Kernel
// loading therefore always runs on a dedicated thread with a large stack, whatever thread calls it.
void runWithLargeStack(const std::function<void()>& body)
{
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 64u << 20);
    auto trampoline = [](void* arg) -> void* {
        (*static_cast<const std::function<void()>*>(arg))();
        return nullptr;
    };
    pthread_t thread;
    if (pthread_create(&thread, &attr, trampoline, const_cast<std::function<void()>*>(&body)) == 0)
        pthread_join(thread, nullptr);
    else
        body();
    pthread_attr_destroy(&attr);
}

} // namespace

struct rtc_device_t
{
    rtc_backend backend;
    void* cudaContext = nullptr; // CUcontext for CUDA devices
    void* cudaStream = nullptr;  // CUstream of the queue (null: the default stream)
    bool cudaStreamKnown = false;
    std::unique_ptr<FileCache> cache;
    ComPtr<IDevice> device;
    ComPtr<ICommandQueue> queue;
    std::string name;
};

struct rtc_buffer_t
{
    rtc_device device;
    ComPtr<IBuffer> buffer;
    uint64_t size;
};

struct rtc_kernel_t
{
    rtc_device device;
    ComPtr<IShaderProgram> program;
    ComPtr<IComputePipeline> pipeline;
    uint32_t groupSize[3];
};

extern "C" {

const char* rtc_last_error(void)
{
    return g_lastError.c_str();
}

rtc_status rtc_device_create(const rtc_device_desc* desc, rtc_device* outDevice)
{
    *outDevice = nullptr;
    DeviceType type;
    switch (desc->backend)
    {
    case RTC_BACKEND_CPU:
        type = DeviceType::CPU;
        break;
    case RTC_BACKEND_CUDA:
        type = DeviceType::CUDA;
        break;
    default:
        return fail("unknown backend");
    }

    auto handle = std::make_unique<rtc_device_t>();
    handle->backend = desc->backend;

    DeviceDesc deviceDesc = {};
    deviceDesc.deviceType = type;
    deviceDesc.slang.searchPaths = desc->search_paths;
    deviceDesc.slang.searchPathCount = desc->search_path_count;
    deviceDesc.enableRayTracing = false;
    if (desc->cache_dir)
    {
        handle->cache = std::make_unique<FileCache>(fs::path(desc->cache_dir) / (type == DeviceType::CUDA ? "cuda" : "cpu"));
        deviceDesc.persistentShaderCache = handle->cache.get();
    }
    if (desc->verbose)
        deviceDesc.debugCallback = &g_stderrCallback;

    if (SLANG_FAILED(getRHI()->createDevice(deviceDesc, handle->device.writeRef())) || !handle->device)
        return fail(desc->backend == RTC_BACKEND_CUDA ? "failed to create CUDA device" : "failed to create CPU device");
    if (SLANG_FAILED(handle->device->getQueue(QueueType::Graphics, handle->queue.writeRef())))
        return fail("failed to get device queue");

    if (type == DeviceType::CUDA)
    {
        DeviceNativeHandles handles = {};
        if (SLANG_SUCCEEDED(handle->device->getNativeDeviceHandles(&handles)))
            for (const NativeHandle& h : handles.handles)
                if (h.type == NativeHandleType::CUcontext)
                    handle->cudaContext = reinterpret_cast<void*>(h.value);
        NativeHandle stream = {};
        if (SLANG_SUCCEEDED(handle->queue->getNativeHandle(&stream)) && stream.type == NativeHandleType::CUstream)
        {
            handle->cudaStream = reinterpret_cast<void*>(stream.value);
            handle->cudaStreamKnown = true;
        }
    }

    const DeviceInfo& info = handle->device->getInfo();
    handle->name = info.adapterName ? info.adapterName : (info.apiName ? info.apiName : "unknown");
    *outDevice = handle.release();
    return RTC_OK;
}

void rtc_device_destroy(rtc_device device)
{
    if (!device)
        return;
    {
        // Drain with the context current; it is released together with the device below.
        ContextScope scope(device->cudaContext);
        if (device->queue)
            device->queue->waitOnHost();
    }
    device->queue = nullptr;
    device->device = nullptr;
    delete device;
}

rtc_backend rtc_device_backend(rtc_device device)
{
    return device->backend;
}

const char* rtc_device_name(rtc_device device)
{
    return device->name.c_str();
}

rtc_status rtc_device_wait(rtc_device device)
{
    ContextScope scope(device->cudaContext);
    if (SLANG_FAILED(device->queue->waitOnHost()))
        return fail("waitOnHost failed");
    return RTC_OK;
}

rtc_status rtc_buffer_create(
    rtc_device device, uint64_t size, uint32_t elementSize, const void* initialData, rtc_buffer* outBuffer)
{
    ContextScope scope(device->cudaContext);
    *outBuffer = nullptr;
    if (size == 0)
        return fail("buffer size must be > 0");
    BufferDesc desc = {};
    desc.size = size;
    desc.elementSize = elementSize;
    desc.format = Format::Undefined;
    desc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource |
                 BufferUsage::CopyDestination;
    desc.defaultState = ResourceState::UnorderedAccess;
    desc.memoryType = MemoryType::DeviceLocal;

    auto handle = std::make_unique<rtc_buffer_t>();
    handle->device = device;
    handle->size = size;
    if (SLANG_FAILED(device->device->createBuffer(desc, initialData, handle->buffer.writeRef())))
        return fail("createBuffer failed (" + std::to_string(size) + " bytes)");
    *outBuffer = handle.release();
    return RTC_OK;
}

void rtc_buffer_destroy(rtc_buffer buffer)
{
    ContextScope scope(buffer ? buffer->device->cudaContext : nullptr);
    delete buffer;
}

uint64_t rtc_buffer_size(rtc_buffer buffer)
{
    return buffer->size;
}

rtc_status rtc_buffer_upload(rtc_buffer buffer, uint64_t offset, uint64_t size, const void* data)
{
    ContextScope scope(buffer->device->cudaContext);
    if (offset + size > buffer->size)
        return fail("upload out of range");
    // CUDA: one asynchronous copy on the queue's stream, ordered with the kernels around it (from pageable memory
    // the call returns once the data is staged, so the caller may reuse it). slang-rhi's encoded upload stages
    // through its own allocations and costs ~1 ms per call, which dominated per-frame (teleop) dispatches.
    if (buffer->device->cudaStreamKnown && cudaDriver().copyToDeviceAsync)
    {
        uint64_t address = buffer->buffer->getDeviceAddress();
        if (address && cudaDriver().copyToDeviceAsync(address + offset, data, size, buffer->device->cudaStream) == 0)
            return RTC_OK;
    }
    auto encoder = buffer->device->queue->createCommandEncoder();
    encoder->uploadBufferData(buffer->buffer, offset, size, const_cast<void*>(data));
    if (SLANG_FAILED(buffer->device->queue->submit(encoder->finish())))
        return fail("upload submit failed");
    return RTC_OK;
}

rtc_status rtc_buffer_download(rtc_buffer buffer, uint64_t offset, uint64_t size, void* outData)
{
    ContextScope scope(buffer->device->cudaContext);
    if (offset + size > buffer->size)
        return fail("download out of range");
    buffer->device->queue->waitOnHost();
    if (SLANG_FAILED(buffer->device->device->readBuffer(buffer->buffer, offset, size, outData)))
        return fail("readBuffer failed");
    return RTC_OK;
}

static rtc_status finishKernel(
    rtc_device device, slang::IModule* module, const char* entryPoint, rtc_kernel* outKernel)
{
    ComPtr<slang::IEntryPoint> entry;
    if (SLANG_FAILED(module->findEntryPointByName(entryPoint, entry.writeRef())))
        return fail(std::string("entry point not found: ") + entryPoint);

    slang::IComponentType* components[] = {module, entry.get()};
    ComPtr<slang::IComponentType> composite;
    ComPtr<slang::IBlob> diagnostics;
    if (SLANG_FAILED(device->device->getSlangSession()->createCompositeComponentType(
            components, 2, composite.writeRef(), diagnostics.writeRef())))
        return failWithDiagnostics("composing program failed", diagnostics);
    ComPtr<slang::IComponentType> linked;
    if (SLANG_FAILED(composite->link(linked.writeRef(), diagnostics.writeRef())))
        return failWithDiagnostics("linking program failed", diagnostics);

    auto handle = std::make_unique<rtc_kernel_t>();
    handle->device = device;

    ShaderProgramDesc programDesc = {};
    programDesc.slangGlobalScope = linked;
    if (SLANG_FAILED(device->device->createShaderProgram(programDesc, handle->program.writeRef(), diagnostics.writeRef())))
        return failWithDiagnostics(std::string("compiling ") + entryPoint + " failed", diagnostics);

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = handle->program;
    if (SLANG_FAILED(device->device->createComputePipeline(pipelineDesc, handle->pipeline.writeRef())))
        return fail(std::string("createComputePipeline failed for ") + entryPoint);

    SlangUInt size[3] = {1, 1, 1};
    if (auto layout = linked->getLayout())
        if (auto ep = layout->getEntryPointByIndex(0))
            ep->getComputeThreadGroupSize(3, size);
    for (int i = 0; i < 3; ++i)
        handle->groupSize[i] = uint32_t(size[i] ? size[i] : 1);

    *outKernel = handle.release();
    return RTC_OK;
}

rtc_status rtc_kernel_load(rtc_device device, const char* moduleName, const char* entryPoint, rtc_kernel* outKernel)
{
    *outKernel = nullptr;
    rtc_status status = RTC_ERROR;
    std::string error;
    runWithLargeStack([&] {
        ContextScope scope(device->cudaContext);
        ComPtr<slang::IBlob> diagnostics;
        slang::IModule* module = device->device->getSlangSession()->loadModule(moduleName, diagnostics.writeRef());
        status = module ? finishKernel(device, module, entryPoint, outKernel)
                        : failWithDiagnostics(std::string("loading module ") + moduleName + " failed", diagnostics);
        error = g_lastError; // thread-local on the compile thread
    });
    if (status != RTC_OK)
        g_lastError = error;
    return status;
}

rtc_status rtc_kernel_load_source(
    rtc_device device, const char* moduleName, const char* path, const char* source, const char* entryPoint,
    rtc_kernel* outKernel)
{
    *outKernel = nullptr;
    rtc_status status = RTC_ERROR;
    std::string error;
    runWithLargeStack([&] {
        ContextScope scope(device->cudaContext);
        ComPtr<slang::IBlob> diagnostics;
        slang::IModule* module = device->device->getSlangSession()->loadModuleFromSourceString(
            moduleName, path, source, diagnostics.writeRef());
        status = module ? finishKernel(device, module, entryPoint, outKernel)
                        : failWithDiagnostics(std::string("compiling module ") + moduleName + " failed", diagnostics);
        error = g_lastError;
    });
    if (status != RTC_OK)
        g_lastError = error;
    return status;
}

void rtc_kernel_destroy(rtc_kernel kernel)
{
    ContextScope scope(kernel ? kernel->device->cudaContext : nullptr);
    delete kernel;
}

void rtc_kernel_thread_group_size(rtc_kernel kernel, uint32_t outSize[3])
{
    for (int i = 0; i < 3; ++i)
        outSize[i] = kernel->groupSize[i];
}

rtc_status rtc_dispatch(
    rtc_device device, rtc_kernel kernel, const rtc_arg* args, uint32_t argCount, uint32_t threadsX, uint32_t threadsY,
    uint32_t threadsZ)
{
    ContextScope scope(device->cudaContext);
    auto encoder = device->queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    IShaderObject* root = pass->bindPipeline(kernel->pipeline);
    ShaderCursor globals(root);
    ShaderCursor entry(root->getEntryPoint(0));

    for (uint32_t i = 0; i < argCount; ++i)
    {
        const rtc_arg& arg = args[i];
        ShaderCursor cursor = globals[arg.name];
        if (!cursor.isValid())
            cursor = entry[arg.name];
        if (!cursor.isValid())
        {
            pass->end();
            return fail(std::string("unknown shader parameter: ") + arg.name);
        }
        Result result = arg.kind == RTC_ARG_BUFFER ? cursor.setBinding(arg.buffer->buffer)
                                                   : cursor.setData(arg.data, arg.size);
        if (SLANG_FAILED(result))
        {
            pass->end();
            return fail(std::string("binding shader parameter failed: ") + arg.name);
        }
    }

    auto groups = [](uint32_t threads, uint32_t group) { return (threads + group - 1) / group; };
    pass->dispatchCompute(
        groups(threadsX, kernel->groupSize[0]), groups(threadsY, kernel->groupSize[1]),
        groups(threadsZ, kernel->groupSize[2]));
    pass->end();
    if (SLANG_FAILED(device->queue->submit(encoder->finish())))
        return fail("dispatch submit failed");
    return RTC_OK;
}

} // extern "C"
