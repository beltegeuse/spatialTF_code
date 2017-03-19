#include "splatting.h"
#include "../misWeights.h"

MTS_NAMESPACE_BEGIN

#if USE_TRACKING_PERF
StatsCounter avgGatherpointsTested("sppmSplat",
    "Avg Gatherpoints Tested", EAverage);
StatsCounter nbGathermapRequest("sppmSplat",
    "Nb Gathermap query", ENumberValue);
StatsCounter percentGPHit("sppmSplat",
    "Percentage valid GP for range", EPercentage);
#endif

//#undef MTS_NOSHADINGNORMAL

#if USE_NODE_FUNC
void RawGatherUpdateQuery::operator()(const GPAccel<GatherPoint>::Type::SearchResult& gpSrc) {
  const GatherPoint& gp = (*gpSrc.data);
#else
void RawGatherUpdateQuery::operator()(const GatherPoint* gpPtr) {
  const GatherPoint& gp = (*gpPtr);
#endif

  Vector wi = itsPhoton.toWorld(itsPhoton.wi);
  Normal photonNormal = itsPhoton.geoFrame.n;
#ifndef MTS_NOSHADINGNORMAL
  Float wiDotGeoN = absDot(photonNormal, wi);
#endif

  if (gp.depth == -1) {
    SLog(EError, "Pushed an invalid GP?");
    return;
  }

  // Test the radius
  Float lengthSqr = (gp.its.p - itsPhoton.p).lengthSquared();
  if ((gp.points->radius * gp.points->radius - lengthSqr) < 0)
    return;

  GatherPoint& gatherPoint = const_cast<GatherPoint&>(gp);
  if (dot(photonNormal, gp.its.shFrame.n) < 1e-1f
#ifndef MTS_NOSHADINGNORMAL
      || wiDotGeoN < 1e-2f
#endif
      )
    return;

  // Test the depth of the photon
  if (maxDepth > 0 && (gp.depth + depthPhoton) > (maxDepth))
    return;

  GatherPointsList * gplist = gatherPoint.points;
  // Increase the count of number of photons
  gplist->tempM[idThread] += 1;  //< Hachisuka done this !

  // Accumulate the contribution of the photon
  // on the impacted gather point
  updateHitpointCount += 1;
  BSDFSamplingRecord bRec(gp.its, gp.its.toLocal(wi), gp.its.wi, EImportance);
  const BSDF * bsdf = gp.its.shape->getBSDF();
  BSDFSamplingRecord bRecForward(gp.its, gp.its.wi, gp.its.toLocal(wi),  ERadiance);
  bRec.component = gp.sampledComponent;
  bRecForward.component = gp.sampledComponent;
#ifdef MTS_NOSHADINGNORMAL
  Spectrum value = powerPhoton * (bsdf->eval(bRec)) / std::abs(Frame::cosTheta(bRec.wo));
#else
  Spectrum value = powerPhoton * bsdf->eval(bRec);  //Frame::cosTheta(bRec.wo)
#endif


  if (value.isZero())
    return;

  // Account for non-symmetry due to shading normals
#ifndef MTS_NOSHADINGNORMAL
  value *= std::abs(
      Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
#endif

  // Compute MIS
  float normalized, unnormalized;
  MISHelper::computeMergeWeights(gp, path, vertexIndex, bsdf->pdf(bRecForward) * gp.pdfComponent, bsdf->pdf(bRec) * gp.pdfComponent, usedTechniques, emittedPhotons, 0, normalized, unnormalized);
  float mis = 1.f / (normalized + unnormalized); // No normalization needed
  if (!(usedTechniques & MISHelper::SPPM_ONLY))
	value *= gp.weight / (gp.points->radius * gp.points->radius * M_PI * emittedPhotons);
  value *= mis;

  // Update flux associated value
  gplist->addFlux(value,idThread,0); // Add to the first buffer
}

bool GatherPointMap::accumulatePhotons(const PhotonSplattingList & lightPath, int idThread) {
#if USE_TRACKING_PERF
  ++nbGathermapRequest;
  avgGatherpointsTested.incrementBase();
#endif

  int totalUpdate = 0;
  // Ignores first vertex
  for ( int i = 1; i < (int)lightPath.photons.size(); ++i )
  {
	  const PhotonInfo & p = lightPath.photons[i];
	  if (p.misInfo.m_flags & VertexMISInfo::DELTA)
		  continue;
	  if (!p.power.isValid()) {
		SLog(EError, "Power invalid !");
	  }
	  if ( m_usedTechniques & MISHelper::MERGE ) {
		  // Create the query and execute the splatting operation
		  RawGatherUpdateQuery query(p.its, p.power, p.depth, (int)m_maxDepth,
									 idThread, lightPath, i, m_usedTechniques, m_emittedPhotons);
		  m_accel.executeQuery(p.its.p, query);
		  totalUpdate += query.updateHitpointCount;
	  }
  }
  if ( m_usedTechniques & MISHelper::CONNECT ) {
	  Vector2i size = m_scene->getFilm()->getCropSize();
	  int blockSize = m_scene->getBlockSize();
	  Sampler * sampler = dynamic_cast<Sampler *>((*m_samplers)[idThread]);
	  Point2 rnd = sampler->next2D();
	  int x = (int)(rnd.x * size.x);
	  int y = (int)(rnd.y * size.y);
	  int blockX = x / blockSize;
	  int blockY = y / blockSize;
	  int blocksInRow = (size.x + blockSize - 1) / blockSize;
	  int blockId = blockY * blocksInRow + blockX;
	  int pixelsInRow = std::min( blockSize, size.x - blockSize * blockX);
	  int listId = (y % blockSize) * pixelsInRow + x % blockSize;
	  MISHelper::computeConnections((*m_gbs)[blockId][listId],lightPath,m_usedTechniques,m_emittedPhotons,idThread, (int)m_maxDepth,1.f, m_scene, 0);
  }

  // If more than one GP is impacted, return true
  return totalUpdate != 0;
}

MTS_IMPLEMENT_CLASS_S(GatherPointMap, false, SerializableObject);
MTS_NAMESPACE_END
