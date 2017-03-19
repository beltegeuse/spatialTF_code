#include "splatting.h"

MTS_NAMESPACE_BEGIN

#if USE_TRACKING_PERF
StatsCounter avgGatherpointsTested("multicontext4",
    "Avg Gatherpoints Tested", EAverage);
StatsCounter nbGathermapRequest("multicontext4",
    "Nb Gathermap query", ENumberValue);
StatsCounter percentGPHit("multicontext4",
    "Percentage valid GP for range", EPercentage);
#endif

#if USE_NODE_FUNC
void RawGatherUpdateQuery::operator()(const GPAccel<GatherPoint>::Type::SearchResult& gpSrc) {
  const GatherPoint& gp = (*gpSrc.data);
#else
void RawGatherUpdateQuery::operator()(const GatherPoint* gpPtr) {
  const GatherPoint& gp = (*gpPtr);
#endif

  Vector wi = its.toWorld(its.wi);
  Normal photonNormal = its.geoFrame.n;
  Float wiDotGeoN = absDot(photonNormal, wi);

  // Test the radius
  Float lengthSqr = (gp.its.p - its.p).lengthSquared();
  if(gp.depth == -1 || (gp.radius*gp.radius - lengthSqr) < 0)
    return;

  GatherPoint& gatherPoint = const_cast<GatherPoint&>(gp);
  if(addPower) {
    gatherPoint.tempN[idThread] += 1;
  }

  if (dot(photonNormal, gp.its.shFrame.n) < 1e-1f // photon.getDepth() > maxDepth || dot(gatherNormal, its.shFrame.n) < 1e-1f
      || wiDotGeoN < 1e-2f)
    return;

  // Test the depth of the photon
  if(maxDepth > 0 && (gp.depth + depth) > (maxDepth))
    return;

  if(depth < minDepth)
    return;

  updateHitpointCount += 1;
  if(addPower) {
    BSDFSamplingRecord bRec(gp.its,
                            gp.its.toLocal(wi), gp.its.wi, EImportance);
    const BSDF * bsdf = gp.its.shape->getBSDF();

    Spectrum value = power * bsdf->eval(bRec);//Frame::cosTheta(bRec.wo)
    if (value.isZero())
      return;

    /* Account for non-symmetry due to shading normals */
    value *= std::abs(Frame::cosTheta(bRec.wi) /
                      (wiDotGeoN * Frame::cosTheta(bRec.wo)));

    // === Update all thread value
    gatherPoint.tempFlux[idThread] += value;
  }
}

bool GatherPointMap::impactHitpoints(const Intersection& its,
                                     const Spectrum& power, int depth,
                                     int idThread, bool added) {
#if USE_CHECK_DEBUG
  if(!power.isValid()) {
    SLog(EError, "Power invalid !");
  }
#endif
#if USE_TRACKING_PERF
  ++nbGathermapRequest;
  avgGatherpointsTested.incrementBase();
#endif
  RawGatherUpdateQuery query(its, power, depth, m_maxDepth, idThread, 0, added);
  m_accel.executeQuery(its.p, query); // XXX Get count to see the efficiently
  return query.updateHitpointCount != 0;
}


MTS_IMPLEMENT_CLASS_S(GatherPointMap, false, SerializableObject);
MTS_NAMESPACE_END
