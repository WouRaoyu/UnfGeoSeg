cmake_minimum_required(VERSION 3.18)

project(dt_pipeline LANGUAGES CXX)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# ---------------------------------------------------------------------------
# Standalone build for the merged infer + export tool. Has no dependency on
# the DTGeoStudio source tree; it only needs the third-party libraries below.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Position-independent code is required because we embed CPython.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Dependencies ---------------------------------------------------------
# OpenVDB ships its own FindOpenVDB.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/lib/cmake/OpenVDB")

# Let VTK import conda's complete TBB target set before OpenVDB's finder can
# create a smaller partial set. Some conda TBB packages are not re-entrant.
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOImage)
find_package(OpenVDB REQUIRED COMPONENTS openvdb)
if(NOT TARGET TBB::tbb)
    find_package(TBB REQUIRED)
endif()
find_package(nlohmann_json REQUIRED)

# Embedded Python interpreter for running the trained model (model_utils.py).
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# ---- Target ---------------------------------------------------------------
add_executable(dt_pipeline dt_pipeline.cpp)

target_link_libraries(dt_pipeline
    PRIVATE
        OpenVDB::openvdb
        TBB::tbb
        nlohmann_json::nlohmann_json
        pybind11::embed
        ${VTK_LIBRARIES}
)

# VTK 9 module auto-initialisation (object factories for the NIFTI writer).
vtk_module_autoinit(
    TARGETS dt_pipeline
    MODULES ${VTK_LIBRARIES}
)

# Make the Python helper available next to the executable so it can be passed
# via `--scripts $<TARGET_FILE_DIR:dt_pipeline>` at runtime.
add_custom_command(TARGET dt_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/model_utils.py"
            "$<TARGET_FILE_DIR:dt_pipeline>/model_utils.py"
    COMMENT "Copying model_utils.py next to dt_pipeline"
)

install(TARGETS dt_pipeline RUNTIME DESTINATION bin)
install(FILES model_utils.py DESTINATION bin)
