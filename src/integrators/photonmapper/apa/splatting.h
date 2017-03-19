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
 RawGatherUpdateQuery(const Intersection& its, const Spectrum& power, int depth, int maxDepth, int idThread, int minDepth,
                      bool addPower)
 : itsPhoton(its), powerPhoton(power), depthPhoton(depth),
   maxDepth(maxDepth), idThread(idThread),
   minDepth(minDepth), addPower(addPower){
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
 int minDepth;
 bool addPower;

 int updateHitpointCount;
};

//////////////////////////////////////////////////////////////////////////
// Gather points Kd-tree
//////////////////////////////////////////////////////////////////////////
class GatherPointMap: public SerializableObject {

 protected:
  //////////////////////////////////////////////////////////////////////////
  // Attributes
  //////////////////////////////////////////////////////////////////////////
  Float m_maxDepth;
  GPAccel<GatherPoint>::Type m_accel;

 public:
  GatherPointMap(std::vector<std::vector<GatherPoint*> >& gps, Float maxDepth)
 : m_maxDepth(maxDepth),
   m_accel(gps) {
  }

  GatherPointMap(Stream *stream, InstanceManager *manager)
    : SerializableObject(stream, manager),
      m_maxDepth(-1),
      m_accel() {
    Log(EError, "No support of serialization ...");
  }

  ~GatherPointMap() {
    Log(EDebug, "Delete the gather points Kd-tree");
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Log(EError, "No support of serialization ...");
  }

  /// Use NNsearch with max radius in the Kd-tree
  bool accumulatePhoton(const Intersection& its, const Spectrum& power, int depth, int idThread, bool added = false);

  MTS_DECLARE_CLASS()

};

MTS_NAMESPACE_END


#endif // SPLATTING_H
