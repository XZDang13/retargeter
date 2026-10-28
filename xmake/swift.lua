-- Whole-module Swift build driven by swiftc (xmake's built-in Swift support does not emit modules).
rule("swiftpkg")
    set_extensions(".swift")
    on_build(function (target, opt)
        import("core.project.depend")
        import("lib.detect.find_tool")
        local swiftc = assert(find_tool("swiftc"), "swiftc not found").program
        local builddir = get_config("builddir") or "build"
        local moddir = path.absolute(path.join(builddir, "swiftmodules"))
        os.mkdir(moddir)
        local argv = {"-module-name", target:name(), "-I", moddir, "-wmo", "-swift-version", "6"}
        if is_mode("debug") then table.join2(argv, {"-Onone", "-g"}) else table.join2(argv, {"-O"}) end
        -- C module maps: own include dirs plus those of every dependency (transitive imports).
        local incdirs = table.wrap(target:get("includedirs"))
        for _, dep in ipairs(target:orderdeps()) do table.join2(incdirs, table.wrap(dep:get("includedirs"))) end
        for _, dir in ipairs(table.unique(incdirs)) do table.join2(argv, {"-I", path.absolute(dir)}) end
        local depfiles = {}
        -- orderdeps() lists dependencies before their dependents; static archives must be linked the other
        -- way round, so walk it backwards.
        local deps = target:orderdeps()
        for i = #deps, 1, -1 do
            local dep = deps[i]
            local f = path.absolute(dep:targetfile())
            table.insert(depfiles, f)
            if target:kind() == "binary" then
                if dep:kind() == "static" then table.insert(argv, f)
                elseif dep:kind() == "shared" then
                    table.join2(argv, {"-L", path.directory(f), "-l" .. dep:basename(),
                                       "-Xlinker", "-rpath", "-Xlinker", path.directory(f)})
                end
            end
        end
        for _, flag in ipairs(target:get("swiftflags")) do table.insert(argv, flag) end
        local out = path.absolute(target:targetfile())
        os.mkdir(path.directory(out))
        if target:kind() == "static" then
            table.join2(argv, {"-parse-as-library", "-emit-library", "-static", "-emit-module",
                               "-emit-module-path", path.join(moddir, target:name() .. ".swiftmodule")})
        elseif target:kind() == "binary" then
            table.insert(argv, "-emit-executable")
        end
        table.join2(argv, {"-o", out})
        local files = {}
        for _, f in ipairs(target:sourcefiles()) do table.insert(files, path.absolute(f)) end
        table.join2(argv, files)
        depend.on_changed(function ()
            print("swiftc %s", target:name())
            os.vrunv(swiftc, argv)
        end, {dependfile = target:dependfile(), files = table.join(files, depfiles), values = argv,
              -- also rebuild when the product was removed (xmake -r / clean) behind the depend cache's back
              changed = target:is_rebuilt() or not os.isfile(out)})
    end)
rule_end()
