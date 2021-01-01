# Compile LodePNG as a static lib
file(GLOB SRC_LODE "${CMAKE_SOURCE_DIR}/source/vendor/lodepng/*.cpp")
add_library(lodepnd STATIC)
target_sources(lodepnd
    PRIVATE
        ${SRC_LODE}
    )
target_include_directories(lodepnd
    PRIVATE
        "${CMAKE_SOURCE_DIR}/source/vendor/lodepng/"
    )
set_target_properties(lodepnd
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lodepnd
    PUBLIC
        png
)
cotire(lodepnd)


file(GLOB_RECURSE SRC_APP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(pencel)
target_sources(pencel
    PRIVATE
        ${SRC_APP}
    )
target_include_directories(pencel
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/source/vendor"
    )
set_target_properties(pencel
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
target_compile_definitions(pencel PRIVATE -DLOGGING_ENABLED=1)
target_link_libraries(pencel
    PRIVATE
        project_options
        project_warnings
        pthread
        stdc++fs
        lodepnd
        kibble # My utility lib
)

cotire(pencel)