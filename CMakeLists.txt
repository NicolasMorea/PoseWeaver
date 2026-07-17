cmake_minimum_required(VERSION 3.16)

project(PoseWeaver LANGUAGES C CXX)

# If the active CMake preset requests ccache but it's not installed,
# disable the compiler launcher so builds still work out of the box.
if(CMAKE_C_COMPILER_LAUNCHER STREQUAL "ccache" OR CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "ccache")
	find_program(POSEWEAVER_CCACHE_PROGRAM ccache)
	if(NOT POSEWEAVER_CCACHE_PROGRAM)
		message(STATUS "ccache not found; disabling C/C++ compiler launcher")
		if(CMAKE_C_COMPILER_LAUNCHER STREQUAL "ccache")
			set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
		endif()
		if(CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "ccache")
			set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
		endif()
	else()
		# Normalize to full path to avoid PATH surprises.
		set(CMAKE_C_COMPILER_LAUNCHER "${POSEWEAVER_CCACHE_PROGRAM}" CACHE STRING "" FORCE)
		set(CMAKE_CXX_COMPILER_LAUNCHER "${POSEWEAVER_CCACHE_PROGRAM}" CACHE STRING "" FORCE)
	endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POSEWEAVER_BUILD_APP "Build PoseWeaver app" ON)
option(POSEWEAVER_ENABLE_CUDA "Build the CUDA cloth simulation backend when a CUDA compiler is available" ON)
set(POSEWEAVER_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures used by PoseWeaver")

set(POSEWEAVER_GPU_SIMULATION_SOURCE src/simulation/gpu/ClothCudaJacobiStub.cpp)
set(POSEWEAVER_HAS_CUDA OFF)
if(POSEWEAVER_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		find_package(CUDAToolkit REQUIRED)
		if(UNIX AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6)
			set(POSEWEAVER_CUDA_LEGACY_COMPILER ON)
		endif()
		set(POSEWEAVER_GPU_SIMULATION_SOURCE "")
		set(POSEWEAVER_HAS_CUDA ON)
	endif()
endif()

include(FetchContent)
find_package(Threads REQUIRED)

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

if(POSEWEAVER_CUDA_LEGACY_COMPILER)
	set(POSEWEAVER_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/ClothCudaJacobi.cu.o")
	add_custom_command(
		OUTPUT "${POSEWEAVER_CUDA_OBJECT}"
		COMMAND "${CMAKE_CUDA_COMPILER}"
			-std=c++14
			--expt-relaxed-constexpr
			-D_GLIBCXX_STD_FUNCTION_H=1
			-O3
			"--generate-code=arch=compute_${POSEWEAVER_CUDA_ARCHITECTURES},code=[compute_${POSEWEAVER_CUDA_ARCHITECTURES},sm_${POSEWEAVER_CUDA_ARCHITECTURES}]"
			"-I${CMAKE_CURRENT_SOURCE_DIR}/src"
			"-I${FETCHCONTENT_BASE_DIR}/eigen-src"
			-c "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/gpu/ClothCudaJacobi.cu"
			-o "${POSEWEAVER_CUDA_OBJECT}"
		DEPENDS
			src/simulation/gpu/ClothCudaJacobi.cu
			src/simulation/gpu/ClothCudaJacobi.h
			src/simulation/xpbd/ClothXpbd.h
		VERBATIM
	)
	set_source_files_properties("${POSEWEAVER_CUDA_OBJECT}" PROPERTIES
		EXTERNAL_OBJECT TRUE
		GENERATED TRUE
	)
	add_library(poseweaver_cuda_backend STATIC "${POSEWEAVER_CUDA_OBJECT}")
	set_target_properties(poseweaver_cuda_backend PROPERTIES LINKER_LANGUAGE CXX)
	target_link_libraries(poseweaver_cuda_backend PUBLIC CUDA::cudart)
elseif(POSEWEAVER_HAS_CUDA)
	add_library(poseweaver_cuda_backend STATIC
		src/simulation/gpu/ClothCudaJacobi.cu
		src/simulation/gpu/ClothCudaJacobi.h
	)
	set_target_properties(poseweaver_cuda_backend PROPERTIES
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
		CUDA_ARCHITECTURES "${POSEWEAVER_CUDA_ARCHITECTURES}"
	)
	target_compile_options(poseweaver_cuda_backend PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
	)
	target_include_directories(poseweaver_cuda_backend PRIVATE src)
	target_link_libraries(poseweaver_cuda_backend PRIVATE Eigen3::Eigen)
	target_link_libraries(poseweaver_cuda_backend PUBLIC CUDA::cudart)
endif()

# stb provides stb_image.h (PNG/JPG/etc image loading) for the canva PNG import.
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        5c205738c191bcb0abc65c4febfa9bd25ff35234
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
	FetchContent_Populate(stb)
endif()

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

# tinyfiledialogs provides a small native file-open dialog used by the canva "Load" button.
FetchContent_Declare(
	tinyfiledialogs
	GIT_REPOSITORY https://github.com/native-toolkit/tinyfiledialogs.git
	GIT_TAG        2.9.3
)
FetchContent_GetProperties(tinyfiledialogs)
if(NOT tinyfiledialogs_POPULATED)
	FetchContent_Populate(tinyfiledialogs)
	add_library(tinyfiledialogs STATIC
		${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c
	)
	target_include_directories(tinyfiledialogs PUBLIC ${tinyfiledialogs_SOURCE_DIR})
	if(UNIX AND NOT APPLE)
		target_link_libraries(tinyfiledialogs PUBLIC dl)
	endif()
endif()

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

if(POSEWEAVER_HAS_CUDA)
	enable_testing()
	add_executable(poseweaver_cuda_smoke
		tests/ClothCudaSmoke.cpp
		src/simulation/gpu/ClothCudaJacobi.h
	)
	target_include_directories(poseweaver_cuda_smoke PRIVATE src)
	target_link_libraries(poseweaver_cuda_smoke PRIVATE Eigen3::Eigen poseweaver_cuda_backend)
	target_compile_features(poseweaver_cuda_smoke PRIVATE cxx_std_17)
	add_test(NAME poseweaver_cuda_smoke COMMAND poseweaver_cuda_smoke)
endif()

if(POSEWEAVER_BUILD_APP)
	add_executable(poseweaver
		src/app/main.cpp
		src/app/AppConfig.cpp
		src/app/AppConfig.h
		src/app/AppState.h
		src/third_party/stb_image_impl.cpp
		src/third_party/stb_image_write_impl.cpp
		src/core/Assets.cpp
		src/core/Assets.h
		src/core/Geometry.h
		src/core/Paths.h
		src/simulation/ClothMeshProcessor.cpp
		src/simulation/ClothMeshProcessor.h
		src/simulation/SmplSimplifier.cpp
		src/simulation/SmplSimplifier.h
		src/simulation/SmplCollisionProxy.cpp
		src/simulation/SmplCollisionProxy.h
		src/simulation/collision/TriangleBvh.cpp
		src/simulation/collision/TriangleBvh.h
		src/simulation/collision/ClothContacts.h
		src/simulation/xpbd/ClothXpbd.cpp
		src/simulation/xpbd/ClothXpbd.h
		${POSEWEAVER_GPU_SIMULATION_SOURCE}
		src/simulation/gpu/ClothCudaJacobi.h
		src/simulation/ClothToSmplMapping.cpp
		src/simulation/ClothToSmplMapping.h
		src/services/ModelInferenceService.cpp
		src/services/ModelInferenceService.h
		src/sketch2depth/inference_worker.py
		src/vitpose/inference_worker.py
		src/smpl/Smplh.cpp
		src/smpl/Smplh.h
		src/render/Renderer.cpp
		src/render/RendererCommands.cpp
		src/render/RendererClothSimulationCommands.cpp
		src/render/RendererClothConstraints.cpp
		src/render/RendererDisplacement.cpp
		src/render/RendererGarmentCommands.cpp
		src/render/Renderer.h
		src/render/RendererGarment.cpp
		src/render/RendererIso.cpp
		src/render/RendererRig.cpp
		src/render/RendererSmpl.cpp
		src/render/RendererSmplPreview.cpp
		src/render/RendererUv.cpp
		src/render/RendererUvCommands.cpp
		src/render/RendererInternal.h
		src/ui/Ui.cpp
		src/ui/Ui.h
		src/ui/UiControls.cpp
		src/ui/UiConsole.cpp
		src/ui/UiMenu.cpp
		src/ui/UiSceneView.cpp
		src/ui/canva/CanvaPreprocess.cpp
		src/ui/canva/CanvaPreprocess.h
		src/ui/canva/SketchPose.cpp
		src/ui/canva/SketchPose.h
		src/ui/canva/CanvaPanel.cpp
		src/ui/canva/CanvaPanel.h
		src/ui/canva/CanvaPanelInternal.h
		src/ui/canva/CanvaPanelCanvasEditing.cpp
		src/ui/canva/CanvaPanelCanvasView.cpp
		src/ui/canva/CanvaPanelClothFitting.cpp
		src/ui/canva/CanvaPanelControls.cpp
		src/ui/canva/CanvaPanelFrameUpdates.cpp
		src/ui/canva/CanvaPanelLayerIO.cpp
		src/ui/canva/CanvaPanelPipelineRequests.cpp
		src/ui/canva/CanvaPanelPipelineUi.cpp
		src/ui/canva/CanvaPanelPoseLayers.cpp
		src/ui/canva/CanvaPanelSequence.cpp
		src/ui/UiInternal.h
	)

	target_include_directories(poseweaver PRIVATE src)

	target_compile_definitions(poseweaver PRIVATE
		GLFW_INCLUDE_NONE
		IMGUI_IMPL_OPENGL_LOADER_GLAD
		POSEWEAVER_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
		POSEWEAVER_SKETCH2RIG_WORKER="${CMAKE_SOURCE_DIR}/src/sketch2rig/inference_worker.py"
		POSEWEAVER_SKETCH2BETAS_WORKER="${CMAKE_SOURCE_DIR}/src/sketch2betas/inference_worker.py"
		POSEWEAVER_SKETCH2UV_WORKER="${CMAKE_SOURCE_DIR}/src/sketch2uv/inference_worker.py"
		POSEWEAVER_SKETCH2ISO_WORKER="${CMAKE_SOURCE_DIR}/src/sketch2iso/inference_worker.py"
		POSEWEAVER_SKETCH2DEPTH_WORKER="${CMAKE_SOURCE_DIR}/src/sketch2depth/inference_worker.py"
		POSEWEAVER_VITPOSE_WORKER="${CMAKE_SOURCE_DIR}/src/vitpose/inference_worker.py"
		POSEWEAVER_RIG_OPTIMIZER_WORKER="${CMAKE_SOURCE_DIR}/src/rigopt/optimize_rig.py"
		POSEWEAVER_HAS_CUDA=$<BOOL:${POSEWEAVER_HAS_CUDA}>
	)

	if(POSEWEAVER_HAS_CUDA)
		target_link_libraries(poseweaver PRIVATE poseweaver_cuda_backend)
	endif()

	target_link_libraries(poseweaver PRIVATE igl::glfw igl::imgui cnpy stb_image tinyfiledialogs Threads::Threads)

	if(MSVC)
		target_compile_options(poseweaver PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
	else()
		target_compile_options(poseweaver PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
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
