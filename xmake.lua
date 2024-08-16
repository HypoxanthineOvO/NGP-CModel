add_languages("c++17")

add_rules("mode.release")
local depends = {
    "eigen", "stb", "nlohmann_json", "openmp"
}
add_requires(depends)

target("Utils")
    set_kind("static")
    add_packages(depends, {public = true})
    add_includedirs("Utils", {public = true})
    --add_defines("EIGEN_DONT_PARALLELIZE", {public = true})


target("NGP_Runner")
    set_kind("static")
    add_includedirs({
        "Modules/Camera",
        "Modules/HashEncoding",
        "Modules/MLP",
        "Modules/SHEncoding",
        "Utils/Image"
        }, {public = true}
    )
    add_files({
        "Modules/Camera/*.cpp",
        "Modules/HashEncoding/*.cpp",
        "Modules/MLP/*.cpp",
        "Utils/Image/image.cpp",
        "Modules/SHEncoding/*.cpp",
        "NGP_Runner/*.cpp"
    })
    add_deps("Utils")
    add_packages("openmp")

target("PCAccNR_Runner")
    set_kind("static")
    add_includedirs({
        "Modules/Camera",
        "Modules/HashEncoding",
        "Modules/MLP",
        "Modules/SHEncoding",
        "Utils/Image"
        }, {public = true}
    )
    add_files({
        "Modules/Camera/*.cpp",
        "Modules/HashEncoding/*.cpp",
        "Modules/MLP/*.cpp",
        "Utils/Image/image.cpp",
        "Modules/SHEncoding/*.cpp",
        "PCAccNR_Runner/*.cpp"
    })
    add_deps("Utils")
    add_packages("openmp")

target("Instant-NGP")
    add_deps("NGP_Runner")
    add_includedirs("NGP_Runner")
    add_files("main.cpp")
    set_kind("binary")

    add_packages("openmp")
    set_targetdir(".")

target("PCAccNR")
    add_deps("PCAccNR_Runner")
    add_includedirs("PCAccNR_Runner")
    add_files("main_pcaccnr.cpp")
    set_kind("binary")

    add_packages("openmp")
    set_targetdir(".")