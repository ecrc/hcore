set( TESTINGS_SRC 
    testing_zsyrk.c
    testing_zgemm.c
    )

list(REMOVE_DUPLICATES HCORE_DEP) # WARNING: is it safe, respect order?
string(REPLACE ";" " " CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
#message(${CMAKE_EXE_LINKER_FLAGS})
#list(APPEND CMAKE_EXE_LINKER_FLAGS "${LAPACKE_LINKER_FLAGS}")
include_directories( ${CMAKE_SOURCE_DIR}/src/misc/include )
include_directories( ${CMAKE_SOURCE_DIR}/src/include )
foreach(_testing ${TESTINGS_SRC})
    get_filename_component(_name_exe ${_testing} NAME_WE)
    add_executable(${_name_exe} ${_testing}) 
    #    add_dependencies(${_name_exe} testing_include)
    #target_link_libraries(${_name_exe} ${libs_for_testings})
    target_link_libraries(${_name_exe} hcore ${HCORE_DEP})
    install(TARGETS ${_name_exe} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/hcore/testing)

    add_test( NAME ${_name_exe} COMMAND ${_name_exe})
    # Test
    #if( _name_exe MATCHES "zposv")
        #add_test( NAME ${_name_exe}
                  #COMMAND ${_name_exe} 1 0 posv 1000 1000 200 1000 250 1e-7 0 250 250 1 1 1)
        #set_tests_properties( ${_name_exe} PROPERTIES ENVIRONMENT "MORSE_TESTING_VERBOSE=1" )
        #set_tests_properties( ${_name_exe} PROPERTIES LABELS "testing" )
    #endif()

endforeach()
