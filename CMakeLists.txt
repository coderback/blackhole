cmake_minimum_required(VERSION 3.21)
project(BlackHoleSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use CLion's built-in vcpkg (which you're already using)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Common dependencies
set(DEPS glfw GLEW::GLEW glm::glm opengl32)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main 3D Black Hole Simulation
add_executable(BlackHole3D src/black_hole.cpp)
target_link_libraries(BlackHole3D PRIVATE ${DEPS})
target_include_directories(BlackHole3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Example programs
add_executable(Lensing2D examples/2D_lensing.cpp)
target_link_libraries(Lensing2D PRIVATE ${DEPS})
target_include_directories(Lensing2D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(CPUGeodesic examples/CPU-geodesic.cpp)
target_link_libraries(CPUGeodesic PRIVATE ${DEPS})
target_include_directories(CPUGeodesic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(RayTracing examples/ray_tracing.cpp)
target_link_libraries(RayTracing PRIVATE ${DEPS})
target_include_directories(RayTracing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Shader files (copy to output dir)
file(GLOB SHADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/geodesic.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid.frag"
)

foreach(shader ${SHADERS})
    add_custom_command(TARGET BlackHole3D POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${shader}
            $<TARGET_FILE_DIR:BlackHole3D>
    )
    add_custom_command(TARGET Lensing2D POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${shader}
            $<TARGET_FILE_DIR:Lensing2D>
    )
    add_custom_command(TARGET CPUGeodesic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${shader}
            $<TARGET_FILE_DIR:CPUGeodesic>
    )
    add_custom_command(TARGET RayTracing POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${shader}
            $<TARGET_FILE_DIR:RayTracing>
    )
endforeach()

# Windows-specific settings
if(WIN32)
    # Set subsystem to console to see output for all targets
    set_target_properties(BlackHole3D Lensing2D CPUGeodesic RayTracing PROPERTIES
            WIN32_EXECUTABLE FALSE
    )
endif()

# Set as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT BlackHole3D)