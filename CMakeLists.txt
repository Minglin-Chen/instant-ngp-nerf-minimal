cmake_minimum_required(VERSION 3.19)
PROJECT(instant-ngp
	VERSION 1.0
	DESCRIPTION "Instant neural graphics primitives"
	LANGUAGES C CXX CUDA
)

###############################################################################
# Build type and C++ compiler setup
###############################################################################
# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-fPIC")
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies (ordered alphabetically)
###############################################################################
include_directories("dependencies")

#########
# Eigen #
#########
include_directories("dependencies/eigen")
include_directories("dependencies/filesystem")

##########
# OpenMP #
##########
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

##########
# Python #
##########
find_package(Python 3.7 COMPONENTS Interpreter Development)
if (Python_FOUND)
	add_subdirectory("dependencies/pybind11")
endif()

################
# tiny-cuda-nn #
################
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(dependencies/tiny-cuda-nn)
include_directories("dependencies/tiny-cuda-nn/include")
include_directories("dependencies/tiny-cuda-nn/dependencies")

##############
# tinylogger #
##############
include_directories("dependencies/tinylogger")

###############################################################################
# Program
###############################################################################
add_definitions(-DNGP_VERSION="${CMAKE_PROJECT_VERSION}")

include_directories(".")

set(SOURCES
	internal/dataset/nerf_synthetic.cu
	internal/sampler/occupancy_sampler.cu
	internal/render/ray_marcher.cu
	internal/utils/common_device.cu
	internal/utils/loss.cu
	internal/utils/render_buffer.cu
	internal/testbed.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

add_library(ngp STATIC ${SOURCES})
set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ngp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ngp PUBLIC ${GL_LIBRARIES} tiny-cuda-nn)
target_compile_options(ngp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

if (Python_FOUND)
	add_library(pyngp SHARED internal/python_api.cu)
	set_target_properties(pyngp PROPERTIES CXX_VISIBILITY_PRESET "hidden")
	set_target_properties(pyngp PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pyngp PUBLIC ngp ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_options(pyngp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
	target_compile_definitions(pyngp PUBLIC -DNGP_PYTHON)
	pybind11_extension(pyngp)

	if (MSVC)
		# Copy dlls needed at runtime
		file(GLOB CUBLAS_DLLS "${CUDA_COMPILER_BIN}/cublas*.dll")
		if (CUBLAS_DLLS)
			add_custom_command(TARGET pyngp POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy ${CUBLAS_DLLS} $<TARGET_FILE_DIR:pyngp>
				COMMAND_EXPAND_LISTS
			)
		endif()
	endif()
endif()
