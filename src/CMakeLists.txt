
set( HCORE_CONTROL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/control/hcore_init.c
    )
SET( HCORE_COMPUTE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dpotrf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dtrsm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dsyrk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dgemm_fast.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dgemm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_sgemm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zgemm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dgemmcd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_dgemmbdcd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_duncompress.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zuncompress.c
  )

set( HCORE_MISC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/misc/compute/dnormest.c
    ${CMAKE_CURRENT_SOURCE_DIR}/misc/descutil/descprint.c
    ${CMAKE_CURRENT_SOURCE_DIR}/misc/descutil/descsame.c
    ${CMAKE_CURRENT_SOURCE_DIR}/misc/descutil/flop_counts.c
  )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/misc/include )
add_library( hcore ${HCORE_CONTROL_SRC} ${HCORE_COMPUTE_SRC} ${HCORE_MISC_SRC})
set_target_properties( hcore PROPERTIES LINKER_LANGUAGE C )


# Installation step
install( TARGETS hcore
         DESTINATION lib )
Install( FILES include/hcore_z.h include/hcore_d.h include/hcore_s.h include/hcore.h include/hcore_init.h 
         misc/include/flop_counts.h
         misc/include/flop_util.h
         misc/include/flop_util_structs.h
         DESTINATION include )
install( FILES misc/include/auxcompute_z.h misc/include/auxcompute_d.h
         DESTINATION include/misc )
install( FILES misc/include/auxdescutil.h
         DESTINATION include/misc )

