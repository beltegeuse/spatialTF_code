#ifndef SPLATTING_H
#define SPLATTING_H

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/gatherproc.h>

// === Internal includes
#include "gpstruct.h"
#include "../gpaccel.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Search/Update strategy
//////////////////////////////////////////////////////////////////////////

/// Definition of the operator of the hitpoint
struct RawGatherUpdateQuery {
  RawGatherUpdateQuery(const Intersection& its, const Spectrum& power,
                       int depth, int maxDepth, int idThread, const PhotonSplattingList & path,
					   int vertexIndex, int usedTechniques, int emittedPhotons )
      : itsPhoton(its),
        powerPhoton(power),
        depthPhoton(depth),
        maxDepth(maxDepth),
        idThread(idThread),
		path(path),
		vertexIndex(vertexIndex),
		usedTechniques(usedTechniques),
		emittedPhotons(emittedPhotons) {
    updateHitpointCount = 0;
  }

#if USE_NODE_FUNC
  inline void operator()(const GPAccel<GatherPoint>::Type::SearchResult& gp);
#else
  inline void operator()(const GatherPoint* gp);
#endif

  const Intersection &itsPhoton;
  const Spectrum& powerPhoton;
  int depthPhoton;
  int maxDepth;
  const int idThread;

  const PhotonSplattingList & path;
  int vertexIndex;

  int usedTechniques;
  int emittedPhotons;

  int updateHitpointCount;
};

//////////////////////////////////////////////////////////////////////////
// Gather points Kd-tree
//////////////////////////////////////////////////////////////////////////
class GatherPointMap : public SerializableObject {
 protected:
  //////////////////////////////////////////////////////////////////////////
  // Attributes
  //////////////////////////////////////////////////////////////////////////
  Float m_maxDepth;
  GPAccel<GatherPoint>::Type m_accel;
  GatherBlocks * m_gbs;
  Scene * m_scene;
  std::vector<SerializableObject *> * m_samplers;

  int m_usedTechniques;
  int m_emittedPhotons;
  int m_photons;

 public:
  GatherPointMap(GatherBlocks & gbs, Scene * scene, std::vector<SerializableObject *> * samplers, Float maxDepth, int usedTechniques, int emittedPhotons)
      : m_maxDepth(maxDepth),
        m_accel(gbs),
		m_gbs(&gbs),
		m_scene(scene),
        m_samplers(samplers),
		m_usedTechniques(usedTechniques),
		m_emittedPhotons(emittedPhotons),
		m_photons(0){
  }
  GatherPointMap(Stream *stream, InstanceManager *manager)
      : SerializableObject(stream, manager),
        m_maxDepth(-1),
        m_accel(),
		m_gbs(NULL),
		m_scene(NULL),
		m_samplers(NULL),
		m_photons(0)
  {
    Log(EError, "No support of serialization ...");
  }
  virtual ~GatherPointMap() {
    Log(EDebug, "Delete the gather points Kd-tree");
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Log(EError, "No support of serialization ...");
  }

  /// Use NNsearch with max radius in the Kd-tree
  bool accumulatePhotons(const PhotonSplattingList & lightPath, int idThread);

  MTS_DECLARE_CLASS()

};

MTS_NAMESPACE_END

#endif // SPLATTING_H
