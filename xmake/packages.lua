-- Third-party packages, pinned. Slang is the prebuilt release that slang-rhi pins and hashes.

package("slang")
    set_homepage("https://shader-slang.org")
    set_description("Slang shading language compiler (prebuilt release)")
    set_urls("https://github.com/shader-slang/slang/releases/download/v$(version)/slang-$(version)-linux-x86_64.tar.gz")
    add_versions("2026.17.1", "31d7e53377dd9a80a1b7b9ec86f800562db91ca4ed2fb4c77573cba4d3e5095d")
    add_links("slang-compiler")
    on_install("linux|x86_64", function (package)
        -- Keep the release layout (include/, lib/, bin/): slang-rhi expects it.
        os.cp("include", package:installdir())
        os.cp("lib", package:installdir())
        os.cp("bin", package:installdir())
    end)
package_end()

package("slang-rhi")
    set_homepage("https://github.com/shader-slang/slang-rhi")
    set_description("Slang rendering hardware interface (CUDA + CPU backends)")
    set_urls("https://github.com/shader-slang/slang-rhi.git")
    add_versions("2026.10.23", "7da55ecd0145543ffaa25eaf537e5727875221f2")
    add_deps("cmake", "slang")
    add_links("slang-rhi", "slang-rhi-resources")
    add_syslinks("dl", "pthread")
    on_install("linux|x86_64", function (package)
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
            "-DSLANG_RHI_ENABLE_CUDA=ON",
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
