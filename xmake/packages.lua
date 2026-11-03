-- Third-party packages, pinned. Slang is the prebuilt release that slang-rhi pins and hashes.

-- Prebuilt Slang release per platform: the release asset name and its sha256 (GitHub's published digests).
local slang_releases = {
    ["linux|x86_64"]  = {"linux-x86_64",   "31d7e53377dd9a80a1b7b9ec86f800562db91ca4ed2fb4c77573cba4d3e5095d"},
    ["linux|arm64"]   = {"linux-aarch64",  "9fde649a04957d854292f315a77fea44365943af6fc0854ab6a01a72cbc24cd7"},
    ["macosx|arm64"]  = {"macos-aarch64",  "ab82adcbeb19ec3246fc3746499f64608fb8e640cbc9d6a55628b9c972abee2c"},
    ["macosx|x86_64"] = {"macos-x86_64",   "2b702d8a0e6571cee32f3d67db0e4eb7847ac99ecb819e0c22592b4df1941f47"},
}

package("slang")
    set_homepage("https://shader-slang.org")
    set_description("Slang shading language compiler (prebuilt release)")
    for key, release in pairs(slang_releases) do
        local plat, arch = key:match("^([^|]+)|([^|]+)$")
        if is_plat(plat) and (is_arch(arch) or (arch == "arm64" and is_arch("aarch64"))) then
            set_urls("https://github.com/shader-slang/slang/releases/download/v$(version)/slang-$(version)-" .. release[1]
                .. ".tar.gz")
            add_versions("2026.17.1", release[2])
        end
    end
    add_links("slang-compiler")
    on_install("linux|x86_64", "linux|arm64", "linux|aarch64", "macosx|arm64", "macosx|x86_64", function (package)
        -- Keep the release layout (include/, lib/, bin/): slang-rhi expects it.
        os.cp("include", package:installdir())
        os.cp("lib", package:installdir())
        os.cp("bin", package:installdir())
    end)
package_end()

package("slang-rhi")
    set_homepage("https://github.com/shader-slang/slang-rhi")
    set_description("Slang rendering hardware interface (CUDA or Metal, plus CPU backends)")
    set_urls("https://github.com/shader-slang/slang-rhi.git")
    add_versions("2026.10.23", "7da55ecd0145543ffaa25eaf537e5727875221f2")
    add_deps("cmake", "slang")
    add_links("slang-rhi", "slang-rhi-resources")
    if is_plat("macosx") then
        add_frameworks("Metal", "Foundation", "QuartzCore")
    else
        add_syslinks("dl", "pthread")
    end
    on_install("linux|x86_64", "linux|arm64", "linux|aarch64", "macosx|arm64", "macosx|x86_64", function (package)
        -- GPU backend: Metal on macOS, CUDA elsewhere (the CPU backend is always built).
        local metal = package:is_plat("macosx")
        local slang = package:dep("slang")
        local configs = {
            "-DSLANG_RHI_BUILD_SHARED=OFF",
            "-DSLANG_RHI_BUILD_TESTS=OFF",
            "-DSLANG_RHI_BUILD_TESTS_WITH_GLFW=OFF",
            "-DSLANG_RHI_BUILD_EXAMPLES=OFF",
            "-DSLANG_RHI_INSTALL=ON",
            "-DSLANG_RHI_FETCH_SLANG=OFF",
            "-DSLANG_RHI_SLANG_INCLUDE_DIR=" .. slang:installdir("include"),
            "-DSLANG_RHI_SLANG_BINARY_DIR=" .. slang:installdir(),
            "-DSLANG_RHI_ENABLE_CUDA=" .. (metal and "OFF" or "ON"),
            "-DSLANG_RHI_ENABLE_METAL=" .. (metal and "ON" or "OFF"),
            "-DSLANG_RHI_ENABLE_CPU=ON",
            "-DSLANG_RHI_ENABLE_OPTIX=OFF",
            "-DSLANG_RHI_ENABLE_VULKAN=OFF",
            "-DSLANG_RHI_ENABLE_WGPU=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"),
        }
        import("package.tools.cmake").install(package, configs, {builddir = "build"})
        -- Embedded shader resources (CMakeRC) are a separate static library that slang-rhi does not install.
        os.cp(path.join("build", "libslang-rhi-resources.a"), package:installdir("lib"))
    end)
package_end()
