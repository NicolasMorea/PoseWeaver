cmake_minimum_required(VERSION 3.16)

project(PoseWeaver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POSEWEAVER_BUILD_APP "Build PoseWeaver app" ON)

include(FetchContent)

# cnpy allows to read .npy files (to read SMPL models and poses)
FetchContent_Declare(
	cnpy
	GIT_REPOSITORY https://github.com/rogersce/cnpy.git
	GIT_TAG        4e8810b1a8637695171ed346ce68f6984e585ef4
	PATCH_COMMAND  ${CMAKE_COMMAND}
		-DSOURCE_DIR=<SOURCE_DIR>
		-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/cmake/patches/cnpy-zip64.patch
		-P ${CMAKE_CURRENT_LIST_DIR}/cmake/patch_cnpy_zip64.cmake
)

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

FetchContent_GetProperties(cnpy)
if(NOT cnpy_POPULATED)
	FetchContent_Populate(cnpy)
	add_library(cnpy STATIC
		${cnpy_SOURCE_DIR}/cnpy.cpp
	)
	target_include_directories(cnpy PUBLIC ${cnpy_SOURCE_DIR})
	find_package(ZLIB REQUIRED)
	target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)
endif()

# Ensure ZIP64 .npz files load correctly (idempotent).
if(EXISTS "${cnpy_SOURCE_DIR}/cnpy.cpp")
	execute_process(
		COMMAND ${CMAKE_COMMAND}
			-DSOURCE_DIR=${cnpy_SOURCE_DIR}
			-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/cmake/patches/cnpy-zip64.patch
			-P ${CMAKE_CURRENT_LIST_DIR}/cmake/patch_cnpy_zip64.cmake
		RESULT_VARIABLE CNPY_PATCH_RESULT
	)
	if(NOT CNPY_PATCH_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to apply cnpy ZIP64 patch")
	endif()
endif()

if(POSEWEAVER_BUILD_APP)
	add_executable(poseweaver
		src/app/main.cpp
		src/app/AppState.h
		src/core/Assets.cpp
		src/core/Assets.h
		src/core/Paths.h
		src/simulation/ClothMeshProcessor.cpp
		src/simulation/ClothMeshProcessor.h
		src/simulation/ClothToSmplMapping.cpp
		src/simulation/ClothToSmplMapping.h
		src/smpl/Smplh.cpp
		src/smpl/Smplh.h
		src/render/Renderer.cpp
		src/render/RendererCommands.cpp
		src/render/RendererGarmentCommands.cpp
		src/render/Renderer.h
		src/render/RendererGarment.cpp
		src/render/RendererSmpl.cpp
		src/render/RendererUv.cpp
		src/render/RendererUvCommands.cpp
		src/render/RendererInternal.h
		src/ui/Ui.cpp
		src/ui/Ui.h
		src/ui/UiControls.cpp
		src/ui/UiConsole.cpp
		src/ui/UiMenu.cpp
		src/ui/UiSceneView.cpp
		src/ui/UiInternal.h
	)

	target_include_directories(poseweaver PRIVATE src)

	target_compile_definitions(poseweaver PRIVATE
		GLFW_INCLUDE_NONE
		IMGUI_IMPL_OPENGL_LOADER_GLAD
		POSEWEAVER_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
	)

	target_link_libraries(poseweaver PRIVATE igl::glfw igl::imgui cnpy)

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