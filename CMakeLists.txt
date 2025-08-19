cmake_minimum_required(VERSION 3.10)
project(pcsx2_headless C CXX)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)


add_library(pcsx2_headless SHARED sdlarch.cpp glad.c)

set_source_files_properties(glad.c PROPERTIES
    LANGUAGE C
    C_STANDARD 11
)

target_include_directories(pcsx2_headless PRIVATE 
    ${SDL2_INCLUDE_DIRS}
    .
)

target_link_libraries(pcsx2_headless PRIVATE 
    SDL2::SDL2 
    OpenGL::GL
    dl
)

set_target_properties(pcsx2_headless PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    C_STANDARD 11
)