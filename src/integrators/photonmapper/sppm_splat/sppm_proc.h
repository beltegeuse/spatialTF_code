#pragma once

#include <fstream>

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/render/gatherproc.h>

#include "splatting.h"
#include "sppm_splat.h"
#include "sppm_wr.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Worker
//////////////////////////////////////////////////////////////////////////

class SplattingPhotonWorker : public ParticleTracer {
 public:
  SplattingPhotonWorker(size_t granularity, int maxDepth, int rrDepth,
                        int idWorker);
  SplattingPhotonWorker(Stream *stream, InstanceManager *manager);

  ref<WorkProcessor> clone() const;
  virtual void prepare();
  void serialize(Stream *stream, InstanceManager *manager) const;
  ref<WorkResult> createWorkResult() const;

  void process(const WorkUnit *workUnit, WorkResult *workResult,
               const bool &stop);
  void handleMediumInteraction(int depth, int nullInteractions, bool delta,
                               const MediumSamplingRecord &mRec,
                               const Medium *medium, const Vector &wi,
                               const Spectrum &weighto) {
    // === No Volume support
    Log(EError, "No support of volume rendering");
  }
  void handleFinishParticule();
  void handleNewPath();
  void handleNewParticle(const Ray ray,const Emitter &emitter);
  void handleSurfaceInteraction(int depth_, int nullInteractions, bool delta,
                                const Intersection &its, const Medium *medium,
                                const Spectrum &weight, const Vector & w);

  MTS_DECLARE_CLASS()

 protected:
  /// Virtual destructor
  virtual ~SplattingPhotonWorker() {
  }
 protected:
  GatherPhotonResult* m_workResult;
  GatherPointMap* m_gathermap;
  size_t m_granularity;
  int m_idWorker;

  PhotonSplattingList m_path;

  // For MIS
  Float m_lastVertexForwardInversePdfSolidAngle;
  Float m_lastCosine;
  Point m_lastVertexPos;

};

//////////////////////////////////////////////////////////////////////////
// Parrall process
//////////////////////////////////////////////////////////////////////////
class SplattingPhotonProcess : public ParticleProcess {
 protected:
  //////////////////////////////////////////////////////////////////////////
  // Attributes
  //////////////////////////////////////////////////////////////////////////
  size_t m_photonCount;
  int m_maxDepth;
  int m_rrDepth;

  size_t m_numEmittedPath;

  mutable int m_idWorker;

 public:
  SplattingPhotonProcess(size_t photonCount, size_t granularity, int maxDepth,
                         int rrDepth, const void *progressReporterPayload);

  ref<WorkProcessor> createWorkProcessor() const;
  void processResult(const WorkResult *wr, bool cancelled);
  bool isLocal() const {
    return true;
  }
  int getNbEmittedPath() {
    return (int)m_numEmittedPath;
  }

  MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END

