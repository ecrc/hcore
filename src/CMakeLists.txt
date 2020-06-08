
set( HCORE_CONTROL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/control/hcore_init.c
    )
SET( HCORE_COMPUTE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zsyrk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zgemm_fast.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zgemm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zgemmbdcd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/hcore_zuncompress.c
  )

set( HCORE_MISC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/misc/compute/znormest.c
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
install( FILES include/hcore_z.h include/hcore.h include/hcore_init.h 
         misc/include/flop_counts.h
         misc/include/flop_util.h
         misc/include/flop_util_structs.h
         DESTINATION include )
install( FILES misc/include/auxcompute_z.h
         DESTINATION include/misc )
install( FILES misc/include/auxdescutil.h
         DESTINATION include/misc )

