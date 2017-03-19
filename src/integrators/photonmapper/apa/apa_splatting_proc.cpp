#include "apa_splatting_proc.h"

MTS_NAMESPACE_BEGIN

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Worker
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

SplattingPhotonWorker::SplattingPhotonWorker(size_t granularity, int maxDepth, int rrDepth, int idWorker) :
    ParticleTracer(maxDepth, rrDepth, false),
    m_workResult(NULL), m_gathermap(NULL),
    m_granularity(granularity), m_idWorker(idWorker) {
}

SplattingPhotonWorker:: SplattingPhotonWorker(Stream *stream, InstanceManager *manager)
    : ParticleTracer(stream, manager) {
        m_granularity = stream->readSize();
        Log(EError, "No Network support");
}

ref<WorkProcessor> SplattingPhotonWorker::clone() const {
    Log(EError, "No suport of worker cloning ... ");
    return 0;
}

void SplattingPhotonWorker::prepare() {
    m_gathermap = static_cast<GatherPointMap *>(getResource("gathermap"));
    ParticleTracer::prepare(); // === The sampler is already the metropolis sampler
}


void SplattingPhotonWorker::serialize(Stream *stream, InstanceManager *manager) const {
    ParticleTracer::serialize(stream, manager);
    stream->writeSize(m_granularity);
    Log(EError, "Not implemented");
}

ref<WorkResult> SplattingPhotonWorker::createWorkResult() const {
    return new GatherPhotonResult();
}

void SplattingPhotonWorker::process(const WorkUnit *workUnit, WorkResult *workResult,
    const bool &stop) {
        m_workResult = static_cast<GatherPhotonResult*>(workResult);
        m_workResult->clear();
        ParticleTracer::process(workUnit, workResult, stop);
#if VERBOSE
        stats->dump();
#endif
        m_workResult = 0;
}


void SplattingPhotonWorker::handleFinishParticule() {
}

void SplattingPhotonWorker::handleNewPath() {
	m_workResult->nextEmittedPath();
}

void SplattingPhotonWorker::handleSurfaceInteraction(int depth_, int nullInteractions, bool delta,
  const Intersection &its, const Medium *medium,
  const Spectrum &weight) {
    // === Don't test if it's a diract function
    int bsdfType = its.getBSDF()->getType(), depth = depth_ - nullInteractions;
    if (!(bsdfType & BSDF::EDiffuseReflection) && !(bsdfType & BSDF::EGlossyReflection))
        return;

    // === test impactHitpoints
    m_gathermap->accumulatePhoton(its, weight, depth, m_idWorker, true);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Process
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

SplattingPhotonProcess::SplattingPhotonProcess(size_t photonCount,
    size_t granularity, int maxDepth, int rrDepth,
    const void *progressReporterPayload):
    ParticleProcess(ParticleProcess::EGather, photonCount, granularity,
    "Photon shooting", progressReporterPayload),
    m_photonCount(photonCount), m_maxDepth(maxDepth),
    m_rrDepth(rrDepth), m_numEmittedPath(0)
{
    // === Count the number of worker for unique id assignement
    m_idWorker = 0;
}


ref<WorkProcessor> SplattingPhotonProcess::createWorkProcessor() const {
	return new SplattingPhotonWorker(m_granularity, m_maxDepth, m_rrDepth, m_idWorker++);
}

void SplattingPhotonProcess::processResult(const WorkResult *wr, bool cancelled) {
  if (cancelled)
      return;
  const GatherPhotonResult &vec = *static_cast<const GatherPhotonResult *>(wr);
  LockGuard lock(m_resultMutex);

  m_numEmittedPath += vec.getNbEmittedPath();

  // === Use gather photon stats
  increaseResultCount(vec.getNbEmittedPath());
}

MTS_IMPLEMENT_CLASS(GatherPhotonResult, false, WorkResult);
MTS_IMPLEMENT_CLASS(SplattingPhotonProcess, false, ParticleProcess);
MTS_IMPLEMENT_CLASS_S(SplattingPhotonWorker, false, ParticleTracer);

MTS_NAMESPACE_END
