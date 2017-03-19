#include "splatting.h"

MTS_NAMESPACE_BEGIN

#if USE_NODE_FUNC
void RawGatherScalarFunctionUpdateQuery::operator()(const GPAccel<GatherPoint>::Type::SearchResult& gpSrc) {
  const GatherPoint& gp = (*gpSrc.data);
#else
void RawGatherScalarFunctionUpdateQuery::operator()(const GatherPoint* gpPtr) {
  const GatherPoint& gp = (*gpPtr);
#endif

  Normal photonNormal = its.geoFrame.n;
#ifndef MTS_NOSHADINGNORMAL
  Vector wi = its.toWorld(its.wi);
  Float wiDotGeoN = absDot(photonNormal, wi);
#endif

  // Test the radius
  Float lengthSqr = (gp.its.p - its.p).lengthSquared();
  if(gp.depth == -1 || (gp.points->radius*gp.points->radius - lengthSqr) < 0)
     return;

  //GatherPoint& gatherPoint = const_cast<GatherPoint&>(gp);
  int tempId = gp.points->maxThread * gp.tempIndex + idThread;
  gp.points->tempM[tempId] += 1;

  if (dot(photonNormal, gp.its.shFrame.n) < 1e-1f // photon.getDepth() > maxDepth || dot(gatherNormal, its.shFrame.n) < 1e-1f
#ifndef MTS_NOSHADINGNORMAL
      || wiDotGeoN < 1e-2f
#endif
     )
     return;

  // Test the depth of the photon
  if(config.maxDepth > 0 && (gp.depth + depth) > (config.maxDepth))
     return;

  updateHitpointCount += 1;
  gp.points->addFlux(Spectrum(1.f), idThread, idImportance);

}

#if USE_NODE_FUNC
void RawGatherImportanceQuery::operator()(const GPAccel<GatherPoint>::Type::SearchResult& gpSrc) {
  const GatherPoint& gp = (*gpSrc.data);
#else
void RawGatherImportanceQuery::operator()(const GatherPoint* gpPtr)  {
  const GatherPoint& gp = (*gpPtr);
#endif
  // Get the information of the photon
  // Like incident directions ... etc.
  Normal photonNormal = its.geoFrame.n;
  Vector wi = its.toWorld(its.wi);
#ifndef MTS_NOSHADINGNORMAL
  Float wiDotGeoN = absDot(photonNormal, wi);
#endif

  // Error checking
  if(gp.depth == -1) {
    SLog(EError, "Depth -1 gp pushed");
    return;
  }

  // Non contribution test: Not inside the radius
  Float lengthSqr = (gp.its.p - its.p).lengthSquared();
  if((gp.points->radius*gp.points->radius - lengthSqr) < 0)
    return;

  // Non contribution test: Not on the same surface
  if (dot(photonNormal, gp.its.geoFrame.n) < 1e-1f
      // photon.getDepth() > maxDepth || dot(gatherNormal, its.shFrame.n) < 1e-1f
#ifndef MTS_NOSHADINGNORMAL
      || wiDotGeoN < 1e-2f
#endif
      )
    return;

  // Non contribution test: Depth of the entire path too long
  if(config.maxDepth > 0 && (gp.depth + depth) > (config.maxDepth))
    return;

#if 1
  // Compute the maximum importance
  // of the current photon surface interaction
  for(size_t idChain = 0; idChain < maxImportances.size(); idChain++) {
    maxImportances[idChain] = std::max(maxImportances[idChain], gp.importance[idChain]);
  }

#else
  if(minDepthGather > gp.radius*gp.radius) {
    maxImportance = gp.importance;
    minDepthGather = gp.radius*gp.radius;
  }
#endif

  // =======================
  // In case of adding statistic
  // in the gather point
  // =======================
  if(addPower) {
	int tempId = gp.points->maxThread * gp.tempIndex + idThread;
    // Error checking to be robust
    if(power.isZero() || weightPath == 0 || path.getImp(idImportance) == 0) {
      SLog(EWarn,"Try to added an non valid path: %s | %f | %f", power.toString().c_str(), weight, path.getImp(idImportance));
    }

    Float * tempM = gp.points->tempM + tempId;
    // Update photon count statistic
    if(config.numberStrat == ENbDifferentContrib) {
      *tempM += 1;
    } else if(config.numberStrat == ENbNormal) {
      *tempM += weightPath;
    } else if(config.numberStrat == ENbMetropolis) {
      *tempM += weightPath / path.getImp(idImportance);
    } else {
      SLog(EError, "Unknow Nb Strat to push into gp statistic");
    }

    // TODO: See this statistic
    if(PixelData<GatherPoint>::usePhiStatistics) {
      gp.points->addPhi(weight, uniqueWeight, idThread, idImportance, gp.tempIndex, isOdd);
    }

    // Evaluate the BSDF
    // To compute the total throughput of the path
    BSDFSamplingRecord bRec(gp.its,
      gp.its.toLocal(wi), gp.its.wi, EImportance);
	BSDFSamplingRecord bRecForward(gp.its, gp.its.wi, gp.its.toLocal(wi), ERadiance);
    const BSDF * bsdf = gp.its.shape->getBSDF();
    bRec.component = gp.sampledComponent;
	bRecForward.component = gp.sampledComponent;
#ifdef MTS_NOSHADINGNORMAL
    Spectrum value = power * bsdf->eval(bRec) / std::abs(Frame::cosTheta(bRec.wo));
#else
    Spectrum value = power * bsdf->eval(bRec);
#endif
    if (!value.isValid()) {
      //TODO: Add GP information
      SLog(EWarn, "Eval issue: (%f) %s\n %s", Frame::cosTheta(bRec.wo),
                 value.toString().c_str(),
                 bsdf->toString().c_str());
      return;
    }
    
    if (value.isZero())
      return;
	
	// Compute MIS
	float normalized, unnormalized;
	MISHelper::computeMergeWeights(gp,path,vertexIndex,bsdf->pdf(bRecForward) * gp.pdfComponent,bsdf->pdf(bRec) * gp.pdfComponent,usedTechniques, emittedPhotons, idImportance, normalized, unnormalized);
	float mis = 1.f / (normalized + unnormalized * normalization); // Normalization needed!
	if (!(usedTechniques & MISHelper::SPPM_ONLY))
		value *= gp.weight / (gp.points->radius * gp.points->radius * M_PI * emittedPhotons);
	value *= mis;

    /* Account for non-symmetry due to shading normals */
#ifndef MTS_NOSHADINGNORMAL
    value *= std::abs(Frame::cosTheta(bRec.wi) /
        (wiDotGeoN * Frame::cosTheta(bRec.wo)));
#endif

    // Update collected flux in the gather points
    gp.points->addFlux(value, idThread, idImportance);
  }

}


MTS_IMPLEMENT_CLASS_S(GatherPointMap, false, SerializableObject);
MTS_NAMESPACE_END
