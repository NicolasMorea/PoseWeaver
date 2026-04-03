cmake_minimum_required(VERSION 3.16)

project(PoseWeaver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POSEWEAVER_BUILD_APP "Build PoseWeaver app" ON)

include(FetchContent)

set(LIBIGL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(LIBIGL_INSTALL OFF CACHE BOOL "" FORCE)

set(LIBIGL_GLFW ON CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI ON CACHE BOOL "" FORCE)
set(LIBIGL_OPENGL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
	libigl
	GIT_REPOSITORY https://github.com/libigl/libigl.git
	GIT_TAG        v2.6.0
)
FetchContent_MakeAvailable(libigl)

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

	target_compile_definitions(poseweaver PRIVATE
		GLFW_INCLUDE_NONE
		IMGUI_IMPL_OPENGL_LOADER_GLAD
		POSEWEAVER_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
	)

	target_link_libraries(poseweaver PRIVATE igl::glfw igl::imgui)

	if(MSVC)
		target_compile_options(poseweaver PRIVATE /W4)
	else()
		target_compile_options(poseweaver PRIVATE -Wall -Wextra -Wpedantic)
	endif()

	# Treat libigl headers as system headers to suppress warnings from them
	get_target_property(IGL_GLFW_INCLUDES igl::glfw INTERFACE_INCLUDE_DIRECTORIES)
	if(IGL_GLFW_INCLUDES)
		target_include_directories(poseweaver SYSTEM PRIVATE ${IGL_GLFW_INCLUDES})
	endif()

	get_target_property(IGL_IMGUI_INCLUDES igl::imgui INTERFACE_INCLUDE_DIRECTORIES)
	if(IGL_IMGUI_INCLUDES)
		target_include_directories(poseweaver SYSTEM PRIVATE ${IGL_IMGUI_INCLUDES})
	endif()
endif()