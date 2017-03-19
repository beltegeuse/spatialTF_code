#include "isppm_scalarFunction_proc.h"

MTS_NAMESPACE_BEGIN

#define USE_UNIFORMSHOOT 1

////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Worker
////////////////////////////////////////////////////////////////////////////////

ScalarFunctionPhotonWorker::ScalarFunctionPhotonWorker(size_t granularity, int maxDepth, int rrDepth, int idWorker) :
    ParticleTracer(maxDepth, rrDepth, false),
    m_workResult(NULL), m_gathermap(NULL),
    m_granularity(granularity), m_idWorker(idWorker),
    m_path(PixelData<GatherPoint>::nbChains) {
}

ScalarFunctionPhotonWorker:: ScalarFunctionPhotonWorker(Stream *stream, InstanceManager *manager)
    : ParticleTracer(stream, manager),
      m_path(PixelData<GatherPoint>::nbChains) {
        m_granularity = stream->readSize();
        Log(EError, "No Network support");
}

ref<WorkProcessor> ScalarFunctionPhotonWorker::clone() const {
    Log(EError, "No suport of worker cloning ... ");
    return 0;
}

void ScalarFunctionPhotonWorker::prepare() {
    m_gathermap = static_cast<GatherPointMap *>(getResource("gathermap"));
    ParticleTracer::prepare(); // === The sampler is already the metropolis sampler
}


void ScalarFunctionPhotonWorker::serialize(Stream *stream, InstanceManager *manager) const {
    ParticleTracer::serialize(stream, manager);
    stream->writeSize(m_granularity);
    Log(EError, "Not implemented");
}

ref<WorkResult> ScalarFunctionPhotonWorker::createWorkResult() const {
    return new GatherScalarFunctionPhotonResult();
}

void ScalarFunctionPhotonWorker::process(const WorkUnit *workUnit, WorkResult *workResult,
    const bool &stop) {
        m_workResult = static_cast<GatherScalarFunctionPhotonResult*>(workResult);
        m_workResult->clear();
        ParticleTracer::process(workUnit, workResult, stop);
        m_workResult = 0;
}

void ScalarFunctionPhotonWorker::handleFinishParticule() {
	// === test impactHitpoints
    // FIXME Verify Importance ID
	if(m_gathermap->impactHitpointsScalarFunction(m_path, m_idWorker, 0)) {
		// Nothings ...
	}
}

void ScalarFunctionPhotonWorker::handleNewPath() {
	m_workResult->nextEmittedPath();
	m_path.clear();
}

void ScalarFunctionPhotonWorker::handleNewParticle(const Ray ray,const Emitter &emitter) {
	PhotonInfo p;
	m_path.add(p);
}

void ScalarFunctionPhotonWorker::handleSurfaceInteraction(int depth_, int nullInteractions, bool delta,
    const Intersection &its, const Medium *medium,
    const Spectrum &weight, const Vector & wo) {

	// === Don't test if it's a dirac function
	int bsdfType = its.getBSDF()->getType(), depth = depth_ - nullInteractions;
	bool deltaSurf = (!(bsdfType & BSDF::EDiffuseReflection)
		&& !(bsdfType & BSDF::EGlossyReflection));

	PhotonInfo p;
	p.its = its;
	p.power = weight;
	p.depth = depth;
	// Save only flags
	p.misInfo.m_flags = deltaSurf ? VertexMISInfo::DELTA : 0;

	m_path.add(p);
}


////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Process
////////////////////////////////////////////////////////////////////////////////

SplattingScalarFunctionPhotonProcess::SplattingScalarFunctionPhotonProcess(size_t photonCount,
    size_t granularity, int maxDepth, int rrDepth,
    const void *progressReporterPayload):
    ParticleProcess(ParticleProcess::ETrace, photonCount, granularity,
    "Photon shooting", progressReporterPayload),
    m_photonCount(photonCount), m_maxDepth(maxDepth),
    m_rrDepth(rrDepth), m_numEmittedPath(0)
{
    // === Count the number of worker for unique id assignement
    m_idWorker = 0;
}


ref<WorkProcessor> SplattingScalarFunctionPhotonProcess::createWorkProcessor() const {
	return new ScalarFunctionPhotonWorker(m_granularity, m_maxDepth, m_rrDepth, m_idWorker++);
}

void SplattingScalarFunctionPhotonProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const GatherScalarFunctionPhotonResult &vec = *static_cast<const GatherScalarFunctionPhotonResult *>(wr);
    LockGuard lock(m_resultMutex);

    m_numEmittedPath += vec.getNbEmittedPath();

    increaseResultCount(vec.getNbEmittedPath());


}

MTS_IMPLEMENT_CLASS(SplattingScalarFunctionPhotonProcess, false, ParticleProcess);
MTS_IMPLEMENT_CLASS_S(ScalarFunctionPhotonWorker, false, ParticleTracer);
MTS_IMPLEMENT_CLASS(GatherScalarFunctionPhotonResult, false, WorkResult);

MTS_NAMESPACE_END
