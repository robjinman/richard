find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

function(compile_shaders targetName shaderSources shaderBinaryDir)
  set(shaderBinaries "")
  foreach(shaderSource ${shaderSources})
    get_filename_component(shaderFilename ${shaderSource} NAME)
    string(REGEX REPLACE "[.]glsl$" ".spv" shaderBinaryName ${shaderFilename})
    set(shaderBinary "${shaderBinaryDir}/${shaderBinaryName}")
    list(APPEND shaderBinaries ${shaderBinary})
    add_custom_command(
      OUTPUT ${shaderBinary}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${shaderBinaryDir}"
      COMMAND ${glslc_executable} -fshader-stage=compute ${shaderSource} -o ${shaderBinary}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      MAIN_DEPENDENCY ${shaderSource}
    )
  endforeach()
  add_custom_target(${targetName} DEPENDS ${shaderBinaries})
endfunction()
