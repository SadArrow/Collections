# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\Meowww_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\Meowww_autogen.dir\\ParseCache.txt"
  "Meowww_autogen"
  )
endif()
