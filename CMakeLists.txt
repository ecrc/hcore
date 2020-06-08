cmake_minimum_required(VERSION 3.2.3)
project( HCORE C )

set(HCORE_VERSION_MAJOR 0)
set(HCORE_VERSION_MINOR 1)
set(HCORE_VERSION_PATCH 0)
set(HCORE_VERSION "${HCORE_VERSION_MAJOR}.${HCORE_VERSION_MINOR}.${HCORE_VERSION_PATCH}")


if(NOT EXISTS "${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules" )
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule init WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE _res_init OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE _res_update OUTPUT_QUIET ERROR_QUIET)
    if( ${_res_init} GREATER 0 OR ${_res_update} GREATER 0 )
        message(FATAL_ERROR "ECRC CMake modules were not found.\n"
                            "We tried: 'git submodule init && git submodule update' and resulted in error" )
    endif()
endif()


## ECRC INITIALIZATION
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules")
set(ECRC_CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules )
include(EcrcInit)


# Set the RPATH config
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# the RPATH to be used when installing
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Misc options
# ------------
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)


###############################################################################
# Look for dependencies #
#########################
set(HCORE_DEP "")
# HCORE depends on CBLAS
#---------------------------
find_package(CBLAS COMPONENTS BLASEXT)
if(BLAS_FOUND)
    if (BLAS_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${BLAS_LIBRARY_DIRS}")
    endif()
    if(BLAS_LINKER_FLAGS)
        list(APPEND CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
    endif()
else()
    message(FATAL_ERROR "BLAS library has not been found")
endif()
if(CBLAS_FOUND)
    include_directories(${CBLAS_INCLUDE_DIRS})
    if(CBLAS_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${CBLAS_LIBRARY_DIRS}")
    endif()
    if (CBLAS_LIBRARIES)
        if (CBLAS_LIBRARIES_DEP)
            list(INSERT HCORE_DEP 0 ${CBLAS_LIBRARIES_DEP})
        else()
            list(INSERT HCORE_DEP 0 ${CBLAS_LIBRARIES})
        endif()
    endif()
else()
    if(ECRC_VERBOSE_FIND_PACKAGE)
        if(CBLAS_STANDALONE OR NOT CBLAS_WORKS)
            if (NOT CBLAS_cblas.h_DIRS)
                Print_Find_Header_Status(cblas cblas.h)
            endif ()
            if (NOT CBLAS_cblas_LIBRARY)
                Print_Find_Library_Status(cblas libcblas)
            endif ()
        endif()
    else()
        message(WARNING "CBLAS library has not been found and ECRC_VERBOSE_FIND_PACKAGE is set to OFF."
        " Try to activate ECRC_VERBOSE_FIND_PACKAGE option (-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints for the detection")
    endif()
    message(FATAL_ERROR "A CBLAS library is required but has not been found")
endif()

list(REMOVE_DUPLICATES CMAKE_EXE_LINKER_FLAGS)
string(REPLACE ";" " " CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# HCORE depends on LAPACKE
#-----------------------------
# standalone version of lapacke seems useless for now
# let the comment in case we meet some problems of non existing lapacke
# functions in lapack library such as mkl, acml, ...
#set(LAPACKE_STANDALONE TRUE)
find_package(LAPACKE COMPONENTS LAPACKEXT)
if(LAPACK_FOUND AND LAPACK_LIBRARY_DIRS)
    # the RPATH to be used when installing
    list(APPEND CMAKE_INSTALL_RPATH "${LAPACK_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "A LAPACK library is required but has not been found")
endif()
if(LAPACKE_FOUND)
    include_directories(${LAPACKE_INCLUDE_DIRS})
    if(LAPACKE_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${LAPACKE_LIBRARY_DIRS}")
    endif()
    if(LAPACKE_LINKER_FLAGS)
        list(APPEND CMAKE_EXE_LINKER_FLAGS "${LAPACKE_LINKER_FLAGS}")
    endif()
    if (LAPACKE_LIBRARIES)
        if (LAPACKE_LIBRARIES_DEP)
            list(INSERT HCORE_DEP 0 ${LAPACKE_LIBRARIES_DEP})
        else()
            list(INSERT HCORE_DEP 0 ${LAPACKE_LIBRARIES})
        endif()
    endif()
else()
    if(ECRC_VERBOSE_FIND_PACKAGE)
        if (LAPACKE_STANDALONE OR NOT LAPACKE_WORKS)
            if (NOT LAPACKE_lapacke.h_DIRS)
                Print_Find_Header_Status(lapacke lapacke.h)
            endif ()
            if (NOT LAPACKE_lapacke_LIBRARY)
                Print_Find_Library_Status(lapacke liblapacke)
            endif ()
        endif()
    else()
        message(WARNING "LAPACKE library has not been found and ECRC_VERBOSE_FIND_PACKAGE is set to OFF."
        " Try to activate ECRC_VERBOSE_FIND_PACKAGE option (-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints for the detection")
    endif()
    message(FATAL_ERROR "A LAPACKE library is required but has not been found")
endif()

if(BLAS_VENDOR_FOUND MATCHES "Intel*")
    add_definitions(-DMKL)
endif()


###############################
## HCORE
###############################
add_subdirectory( src )

# Tests
option( HCORE_TESTING "Enable testing code" ON )
if( HCORE_TESTING )
    message(STATUS "HCORE testing is enabled")
    enable_testing()
    include(CTest)
    add_subdirectory( testing )
endif()


## Generate pkg-config file
include(GenPkgConfig)
generate_pkgconfig_file()


###############################################################################
# Release source #
##################
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "hcore")
set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HCORE library ")
set(CPACK_PACKAGE_VERSION "${HCORE_VERSION_MAJOR}.${HCORE_VERSION_MINOR}.${HCORE_VERSION_PATCH}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "KAUST")
set(CPACK_PACKAGE_CONTACT "eduardo.gonzalezfisher @ kaust.edu.sa")
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_SOURCE_IGNORE_FILES "plasma-conversion;build;.cproject;.settings;.dir-locals.el;.project;.pydevproject;.svn;.git;.gitmodules;.gitlab-ci.yml")
include(CPack)
