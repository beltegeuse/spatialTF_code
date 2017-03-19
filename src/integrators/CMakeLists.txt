# Integrators
if (NOT MTS_VERSION)
  message(FATAL_ERROR "Use the top level configuration file")
endif()

macro(add_integrator)
  add_mts_plugin(${ARGN} TYPE integrator)
endmacro()

macro(add_bidir)
  add_integrator(${ARGN} MTS_BIDIR)
endmacro()

add_integrator(ao       direct/ao.cpp)
add_integrator(direct   direct/direct.cpp)
add_integrator(path     path/path.cpp)
add_integrator(volpath  path/volpath.cpp)
add_integrator(volpath_simple path/volpath_simple.cpp)
add_integrator(ptracer  ptracer/ptracer.cpp
                        ptracer/ptracer_proc.h ptracer/ptracer_proc.cpp)

# Photon mapping-based techniques
add_integrator(photonmapper photonmapper/photonmapper.cpp photonmapper/bre.cpp)
add_integrator(ppm          photonmapper/ppm.cpp)
add_integrator(sppm         photonmapper/sppm.cpp)
#add_integrator(apa        photonmapper/apa/apa.cpp
#						  photonmapper/apa/apa_splatting_proc.cpp
#						  photonmapper/apa/splatting.cpp)

add_integrator(avg        avg.cpp)

# Miscellaneous
add_integrator(vpl          vpl/vpl.cpp MTS_HW)
add_integrator(adaptive     misc/adaptive.cpp)
add_integrator(irrcache     misc/irrcache.cpp
                            misc/irrcache_proc.h misc/irrcache_proc.cpp)
add_integrator(multichannel misc/multichannel.cpp)
add_integrator(field        misc/field.cpp)

# Bidirectional techniques
add_bidir(bdpt          bdpt/bdpt.h      bdpt/bdpt.cpp
                        bdpt/bdpt_proc.h bdpt/bdpt_proc.cpp
                        bdpt/bdpt_wr.h   bdpt/bdpt_wr.cpp)

add_bidir(pssmlt        pssmlt/pssmlt.h         pssmlt/pssmlt.cpp
                        pssmlt/pssmlt_proc.h    pssmlt/pssmlt_proc.cpp
                        pssmlt/pssmlt_sampler.h pssmlt/pssmlt_sampler.cpp)

add_bidir(mlt           mlt/mlt.h      mlt/mlt.cpp
                        mlt/mlt_proc.h mlt/mlt_proc.cpp)

add_bidir(erpt          erpt/erpt.h      erpt/erpt.cpp
                        erpt/erpt_proc.h erpt/erpt_proc.cpp)

add_bidir(msppm 
	photonmapper/msppm/msppm.cpp
	photonmapper/msppm/splatting.h
    photonmapper/msppm/splatting.cpp
    photonmapper/msppm/msppm_proc.cpp
    photonmapper/msppm/msppm_sampler.cpp
    photonmapper/msppm/msppm_path.cpp
    photonmapper/msppm/imp/isppm/isppm_scalarFunction_proc.cpp
    photonmapper/msppm/imp/importanceFunc.cpp
)

add_bidir(sppm_splat 
	photonmapper/sppm_splat/sppm_splat.cpp
	photonmapper/sppm_splat/splatting.h
    photonmapper/sppm_splat/splatting.cpp
    photonmapper/sppm_splat/sppm_wr.cpp
    photonmapper/sppm_splat/sppm_proc.cpp
    photonmapper/sppm_splat/sppm_splat.cpp
)

add_integrator(path_bounce     path/path_bounce.cpp)
