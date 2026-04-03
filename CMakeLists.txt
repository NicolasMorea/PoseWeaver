cmake_minimum_required(VERSION 3.16)

project(PoseWeaver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POSEWEAVER_BUILD_APP "Build PoseWeaver app" ON)

find_package(OpenGL REQUIRED)

# GLAD (same pinned commit as libigl recipe)
include(FetchContent)
FetchContent_Declare(
	glad
	GIT_REPOSITORY https://github.com/libigl/libigl-glad.git
	GIT_TAG        651a425101365aa6e8504988ef9bb363d066c5ee
)
FetchContent_MakeAvailable(glad)
add_library(glad::glad ALIAS glad)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GLFW_BUILD_WAYLAND)
	set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "")
endif()
if(NOT DEFINED GLFW_BUILD_X11)
	set(GLFW_BUILD_X11 ON CACHE BOOL "")
endif()
add_subdirectory(external/glfw)

# Dear ImGui (as a submodule)
add_library(imgui STATIC
	external/imgui/imgui.cpp
	external/imgui/imgui_demo.cpp
	external/imgui/imgui_draw.cpp
	external/imgui/imgui_tables.cpp
	external/imgui/imgui_widgets.cpp
	external/imgui/backends/imgui_impl_glfw.cpp
	external/imgui/backends/imgui_impl_opengl3.cpp
)


target_include_directories(imgui PUBLIC
	external/imgui
	external/imgui/backends
	external/glfw/include
)

target_compile_definitions(imgui PUBLIC GLFW_INCLUDE_NONE)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL glad::glad)

if(MSVC)
	target_compile_options(imgui PRIVATE /W4)
else()
	target_compile_options(imgui PRIVATE -Wall -Wextra -Wpedantic)
endif()

# App

if(POSEWEAVER_BUILD_APP)
	add_executable(poseweaver
		src/app/main.cpp
		src/app/AppState.h
		src/core/Assets.cpp
		src/core/Assets.h
		src/core/Paths.h
		src/render/Renderer.cpp
		src/render/Renderer.h
		src/ui/Ui.cpp
		src/ui/Ui.h
	)

	target_include_directories(poseweaver PRIVATE src)
	target_compile_definitions(poseweaver PRIVATE GLFW_INCLUDE_NONE)
	target_link_libraries(poseweaver PRIVATE imgui glfw OpenGL::GL glad::glad)

	target_compile_definitions(poseweaver PRIVATE
		POSEWEAVER_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
	)

	if(MSVC)
		target_compile_options(poseweaver PRIVATE /W4)
	else()
		target_compile_options(poseweaver PRIVATE -Wall -Wextra -Wpedantic)
	endif()
endif()
