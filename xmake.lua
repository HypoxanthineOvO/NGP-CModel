add_languages("c++17")

add_rules("mode.release")

local depends = {
    "eigen3", "stb", "nlohmann_json"
}
add_requires(depends)
add_requires("openmp")

target("Utils")
    set_kind("static")
    add_packages(depends, {public = true})
    add_includedirs("Utils", {public = true})

target("NGP_Runner")
    set_kind("static")
    add_includedirs({
        "Modules/Camera",
        "Modules/HashEncoding",
        "Modules/IO/Config", "Modules/IO/Image",
        "Modules/MLP",
        "Modules/SHEncoding"
        }, {public = true}
    )
    add_files({
        "Modules/Camera/*.cpp",
        "Modules/HashEncoding/*.cpp",
        "Modules/MLP/*.cpp",
        "Modules/IO/Image/image.cpp",
        "Modules/SHEncoding/*.cpp",
        "NGP_Runner/*.cpp"
    })
    add_deps("Utils")

target("Instant-NGP")
    add_deps("NGP_Runner")
    add_includedirs("NGP_Runner")
    add_files("main.cpp")
    set_kind("binary")
    set_targetdir(".")
