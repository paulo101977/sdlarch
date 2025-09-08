cmake_minimum_required(VERSION 3.12)
project(pcsx2_headless LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DGLAD_GLAPI_EXPORT)
add_definitions(-DGLAD_GLAPI_EXPORT_BUILD)

add_subdirectory("third-party/pybind11")

find_package(Python3 COMPONENTS Development REQUIRED)
# find_package(pybind11 REQUIRED)

if(UNIX AND NOT APPLE AND NOT PYTHON3_EXTENSION_SUFFIX)
    set(PYTHON3_EXTENSION_SUFFIX ".so")
    message(STATUS "Setting manual Python extension suffix: ${PYTHON3_EXTENSION_SUFFIX}")
endif()

message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")

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

add_library(glad STATIC glad.c)
set_target_properties(glad PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(pcsx2_headless SHARED
    sdlarch.cpp
)

pybind11_add_module(_retro
    "sdlarch.cpp"
    "glad.c"
)

target_include_directories(pcsx2_headless PRIVATE
    ${SDL2_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    .
)



target_include_directories(_retro PRIVATE
    ${SDL2_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    .
)

target_include_directories(glad PRIVATE
    .
)

if(WIN32)
    target_link_libraries(pcsx2_headless PRIVATE
        glad  
        ${SDL2_LIBRARY}
        opengl32.lib
        ${Python3_LIBRARIES}
    )
    
    target_link_libraries(_retro PRIVATE
        pcsx2_headless
        ${SDL2_LIBRARY}
        opengl32.lib
        ${Python3_LIBRARIES}
        pybind11::module
    )
    
    # set_target_properties(pcsx2_headless PROPERTIES
    #     PREFIX ""
    #     SUFFIX ".dll"
    # )
    
    set_target_properties(_retro PROPERTIES
        OUTPUT_NAME "_retro"
        SUFFIX ".pyd"
        PREFIX ""
    )
    
else()
    # Linux/WSL
    target_link_libraries(pcsx2_headless PRIVATE
        SDL2::SDL2
        OpenGL::GL
        dl
        pthread
        ${Python3_LIBRARIES}
    )
    
    target_link_libraries(_retro PRIVATE
        pcsx2_headless
        SDL2::SDL2
        OpenGL::GL
        dl
        pthread
        ${Python3_LIBRARIES}
        pybind11::module
    )
    
    # set_target_properties(pcsx2_headless PROPERTIES
    #     PREFIX ""
    #     SUFFIX ".so"
    # )
    
    set_target_properties(_retro PROPERTIES
        OUTPUT_NAME "_retro"
        SUFFIX "${PYTHON3_EXTENSION_SUFFIX}"
        PREFIX ""
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
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "Python extension suffix: ${PYTHON3_EXTENSION_SUFFIX}")

# install(TARGETS pcsx2_headless
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )

# install(TARGETS _retro
#     LIBRARY DESTINATION lib
# )