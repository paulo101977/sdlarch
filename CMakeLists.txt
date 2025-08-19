cmake_minimum_required(VERSION 3.12)
project(pcsx2_headless LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-D_WINDOWS -D_USRDLL)

    set(SDL2_POSSIBLE_PATHS
        "C:/SDL2"
        "C:/Program Files/SDL2"
        "D:/SDL2"
        "C:/Program Files (x86)/SDL2"
        "${CMAKE_SOURCE_DIR}/SDL2"
        "${CMAKE_SOURCE_DIR}/../SDL2"
    )
    
    find_path(SDL2_INCLUDE_DIR SDL.h
        PATHS ${SDL2_POSSIBLE_PATHS}
        PATH_SUFFIXES include
    )
    
    find_library(SDL2_LIBRARY
        NAMES SDL2
        PATHS ${SDL2_POSSIBLE_PATHS}
        PATH_SUFFIXES lib/x64 lib lib64
    )
    
    if(SDL2_INCLUDE_DIR AND SDL2_LIBRARY)
        message(STATUS "Found SDL2: ${SDL2_LIBRARY}")
        set(SDL2_FOUND TRUE)
    else()
        message(FATAL_ERROR "SDL2 not found! Please install SDL2 or set SDL2_ROOT")
    endif()

else()
    # Linux/WSL
    find_package(SDL2 REQUIRED)
    find_package(OpenGL REQUIRED)
    set(SDL2_FOUND TRUE)
endif()

add_library(pcsx2_headless SHARED
    sdlarch.cpp
    glad.c
)


target_include_directories(pcsx2_headless PRIVATE
    ${SDL2_INCLUDE_DIR}
    .
)

if(WIN32)
    target_link_libraries(pcsx2_headless PRIVATE
        ${SDL2_LIBRARY}
        opengl32.lib
    )
    
    set_target_properties(pcsx2_headless PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
    )
    
    add_custom_command(TARGET pcsx2_headless POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDL2_LIBRARY_DIR}/../bin/SDL2.dll"
            $<TARGET_FILE_DIR:pcsx2_headless>
        COMMENT "Copying SDL2.dll to output directory"
    )
    
else()
    target_link_libraries(pcsx2_headless PRIVATE
        SDL2::SDL2
        OpenGL::GL
        dl
        pthread
    )
    
    set_target_properties(pcsx2_headless PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
endif()

set_source_files_properties(glad.c PROPERTIES
    LANGUAGE C
    C_STANDARD 11
)

message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Building for: ${CMAKE_SYSTEM}")
message(STATUS "SDL2 include: ${SDL2_INCLUDE_DIR}")
message(STATUS "SDL2 library: ${SDL2_LIBRARY}")