#include "sppm_proc.h"
#include "sppm_splat.h"

#include <mitsuba/render/range.h>

MTS_NAMESPACE_BEGIN

#define USE_UNIFORMSHOOT 1

////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Worker
////////////////////////////////////////////////////////////////////////////////

SplattingPhotonWorker::SplattingPhotonWorker(size_t granularity, int maxDepth,
                                             int rrDepth, int idWorker)
    : ParticleTracer(maxDepth, rrDepth, false),
      m_workResult(NULL),
      m_gathermap(NULL),
      m_granularity(granularity),
      m_idWorker(idWorker) {
}

SplattingPhotonWorker::SplattingPhotonWorker(Stream *stream,
                                             InstanceManager *manager)
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

  // Initialize:
  //  - scene
  //  - sensor
  //  - sampler
  ParticleTracer::prepare();
}

void SplattingPhotonWorker::serialize(Stream *stream,
                                      InstanceManager *manager) const {
  ParticleTracer::serialize(stream, manager);
  stream->writeSize(m_granularity);
  Log(EError, "Not implemented");
}

ref<WorkResult> SplattingPhotonWorker::createWorkResult() const {
  return new GatherPhotonResult();
}

void SplattingPhotonWorker::process(const WorkUnit *workUnit,
                                    WorkResult *workResult, const bool &stop) {
  m_workResult = static_cast<GatherPhotonResult*>(workResult);
  m_workResult->clear();

  // Convert the work unit
#if 0
  const RangeWorkUnit* wu = static_cast<const RangeWorkUnit*>(workUnit);
  Log(EInfo, "Work unit with range: %i [%i -> %i] (id: %i)", wu->getSize(),
      wu->getRangeStart(), wu->getRangeEnd(), m_idWorker);
#endif

  // Process by launching several paths
  ParticleTracer::process(workUnit, workResult, stop);
  m_workResult = 0;
}

void SplattingPhotonWorker::handleFinishParticule() {
	// Accumulate the photon to the gather point
	m_gathermap->accumulatePhotons(m_path, m_idWorker);
}

void SplattingPhotonWorker::handleNewPath() {
  // Count the number of emitted path
  m_workResult->nextEmittedPath();
  m_path.clear();
}
void SplattingPhotonWorker::handleNewParticle(const Ray ray,const Emitter &emitter) {
	VertexMISInfo info;
	info.m_flags = emitter.isEnvironmentEmitter() ? VertexMISInfo::DISTANT_LIGHT : 0;
	float emitPdf, directPdf, cosTheta;
	// EmitPdf is in solid angle measure, directPdf is in area measure
	emitter.pdfRay(ray, emitPdf, directPdf, cosTheta);
	float prob = m_scene->pdfEmitterDiscrete(&emitter);
	directPdf *= prob;
	emitPdf *= prob;
	info.m_vertexInverseForwardPdf = 1.f / directPdf;
	info.m_importance[0] = 1.f;
	info.m_importance[1] = 1.f;
	m_lastVertexForwardInversePdfSolidAngle = directPdf / emitPdf;
	m_lastCosine = cosTheta;
	m_lastVertexPos = ray.o;
	PhotonInfo p;
	p.misInfo = info;
	m_path.add(p);
	m_path.importance[0] = 1.f;
	m_path.importance[1] = 1.f;
}

void SplattingPhotonWorker::handleSurfaceInteraction(int depth_,
                                                     int nullInteractions,
                                                     bool delta,
                                                     const Intersection &its,
                                                     const Medium *medium,
                                                     const Spectrum &power, 
													 const Vector & wo) {
	// === Don't test if it's a dirac function
	int bsdfType = its.getBSDF()->getType(), depth = depth_ - nullInteractions;
	bool deltaSurf = (!(bsdfType & BSDF::EDiffuseReflection)
		&& !(bsdfType & BSDF::EGlossyReflection));

	// Get bsdf pdfs
	BSDFSamplingRecord bRecForward(its,its.wi,its.toLocal(wo),EImportance);
	Float forwardPdf = its.getBSDF()->pdf(bRecForward);
	BSDFSamplingRecord bRecReverse(its,its.toLocal(wo),its.wi,ERadiance);
	Float reversePdf = its.getBSDF()->pdf(bRecReverse);

	PhotonInfo p;
	p.its = its;
	p.power = power;
	p.depth = depth;

	// TODO multiply bsdfPdf and reversePdf by RR probability
	// First compute forward and backward distance and cosine for solid angle to area measure converting
	VertexMISInfo &lastInfo = m_path.photons.back().misInfo;
	Float distSqr = (its.p - m_lastVertexPos).lengthSquared();
	Float forwardAreaMeasure, backwardAreaMeasure;
	// Delta surface on current vertex influences backward distance and cosine
	if ( deltaSurf )
	{
		forwardPdf = reversePdf = 1.f;
		backwardAreaMeasure = 1.f;
		p.misInfo.m_flags = VertexMISInfo::DELTA;
	}
	else
	{
		backwardAreaMeasure = m_lastCosine / distSqr;
		p.misInfo.m_flags = 0;
	}
	// Delta surface on previous vertex influences forward distance and cosine
	if ( lastInfo.m_flags & VertexMISInfo::DELTA ) 
		forwardAreaMeasure = 1.f;
	else if ( lastInfo.m_flags & VertexMISInfo::DISTANT_LIGHT ) {
		forwardAreaMeasure = 1.f / std::abs(dot(its.toWorld(its.wi),its.geoFrame.n));
		backwardAreaMeasure = 1.f;
	}
	else
		forwardAreaMeasure = distSqr / std::abs(dot(its.toWorld(its.wi),its.geoFrame.n));
	// Update last vertex MIS info
	lastInfo.m_vertexReversePdfWithoutBSDF = backwardAreaMeasure;
	assert(lastInfo.m_vertexReversePdfWithoutBSDF);
	lastInfo.m_vertexReversePdf = lastInfo.m_vertexReversePdfWithoutBSDF * reversePdf;
	assert(lastInfo.m_vertexReversePdf || reversePdf == 0.f);
	// Update current vertex MIS info
	p.misInfo.m_vertexInverseForwardPdf = m_lastVertexForwardInversePdfSolidAngle * forwardAreaMeasure;
	assert(lastInfo.m_vertexInverseForwardPdf);
	p.misInfo.m_importance[0] = 1.f;
	p.misInfo.m_importance[1] = 1.f;
	m_lastVertexPos = its.p;
	m_lastVertexForwardInversePdfSolidAngle = 1.0f / forwardPdf;
	m_lastCosine = std::abs(dot(wo,its.geoFrame.n));

	m_path.add(p);
}

////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Process
////////////////////////////////////////////////////////////////////////////////

SplattingPhotonProcess::SplattingPhotonProcess(
    size_t photonCount, size_t granularity, int maxDepth, int rrDepth,
    const void *progressReporterPayload)
    : ParticleProcess(ParticleProcess::ETrace, photonCount, granularity,
                      "Photon shooting", progressReporterPayload),
      m_photonCount(photonCount),
      m_maxDepth(maxDepth),
      m_rrDepth(rrDepth),
      m_numEmittedPath(0) {
  // === Count the number of worker for unique id assignement
  m_idWorker = 0;
}

ref<WorkProcessor> SplattingPhotonProcess::createWorkProcessor() const {
  return new SplattingPhotonWorker(m_granularity, m_maxDepth, m_rrDepth,
                                   m_idWorker++);
}

void SplattingPhotonProcess::processResult(const WorkResult *wr,
                                           bool cancelled) {
  if (cancelled)
    return;
  const GatherPhotonResult &vec = *static_cast<const GatherPhotonResult *>(wr);
  LockGuard lock(m_resultMutex);

  m_numEmittedPath += vec.getNbEmittedPath();

  // === Use gather photon stats
  increaseResultCount(vec.getNbEmittedPath());
}

MTS_IMPLEMENT_CLASS(SplattingPhotonProcess, false, ParticleProcess);
MTS_IMPLEMENT_CLASS_S(SplattingPhotonWorker, false, ParticleTracer);

MTS_NAMESPACE_END
