#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>

#include "misWeights.h"

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

#define USE_BBOX_INITIALRADIUS 1

class RadiusInitializer {
 public:
  RadiusInitializer(const Properties& props)
      : scene(0),
        maxDepth(-1) {

    referenceMod = props.getBoolean("referenceMod", false);
    m_bounceRoughness = props.getFloat("bounceRoughness");

    if(m_bounceRoughness <= 0.0 || m_bounceRoughness > 1.0) {
      SLog(EError,"Bad roughtness constant: %f", m_bounceRoughness);
    }
  }

  virtual ~RadiusInitializer() {
  }

  virtual void init(Scene* scene, int maxDepth,
                    GatherBlocks & gatherPoints,
                    const std::vector<Point2i>& offset,
					int techniques) {
    SLog(EInfo, "Initialize GP generation object");

    // Copy the object
    // These will be used to generate
    // the gather points through the scene
    this->scene = scene;
    this->maxDepth = maxDepth;
    this->gatherBlocks = &gatherPoints;
    this->offset = &offset;
	this->techniques = techniques;

    // And then initialise the samplers
    // without shift in the random
    // sequence
    generateSamplers(0);
  }

  void generateSamplers(int shiftRandomNumber) {
    // Initialize samplers for generate GP
    // Each pixel blocks will have is own
    // sampler to generate the same set
    // of gather points

    Properties props("independent");
    if (referenceMod) {
      props.setBoolean("randInit", true);
    }

    // --- Sampler to generate seed of other
    ref<Sampler> samplerIndependent =
        static_cast<Sampler *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(Sampler), props));

    if (shiftRandomNumber != 0) {
      SLog(EInfo, "Make an shift of random number generator: %i (%i)",
           shiftRandomNumber, offset->size());
      // Make the shift by advancing in the
      // sequence by calling the
      for (size_t i = 0; i < (offset->size() * shiftRandomNumber); ++i) {
        samplerIndependent->next2D();
      }
    }

    // --- Create all samplers
    samplers.resize(offset->size());  //< Create sampler
    for (size_t i = 0; i < offset->size(); ++i) {
      ref<Sampler> clonedIndepSampler = samplerIndependent->clone();
      samplers[i] = clonedIndepSampler;
    }
  }

  virtual Float getRadiusRayDifferential(RayDifferential& ray,
                                         Float totalDist) {

    if (ray.hasDifferentials) {  // nbComponentExtra == 0 &&
      ray.scaleDifferential(1.f);  // Scale to one (TODO: Is it necessary ??? )
      Point posProj = ray.o + ray.d * totalDist;
      Point rX = ray.rxOrigin + ray.rxDirection * totalDist;
      Point rY = ray.ryOrigin + ray.ryDirection * totalDist;
      Float dX = (rX - posProj).length();
      Float dY = (rY - posProj).length();

      Float r = std::max(dX, dY);
      if (r > 100) {
        SLog(EError, "Infinite radius %f", r);
      }
      return r;
    } else {
      SLog(EError, "No ray differential");
      return 0.f;
    }
  }

  void regeneratePositionAndRadius() {
    // Get some data ...
    ref<Sensor> sensor = scene->getSensor();
    bool needsApertureSample = sensor->needsApertureSample();
    bool needsTimeSample = sensor->needsTimeSample();
    ref<Film> film = sensor->getFilm();
    Vector2i cropSize = film->getCropSize();
    Point2i cropOffset = film->getCropOffset();
    int blockSize = scene->getBlockSize();

#if defined(MTS_OPENMP)
    ref<Scheduler> sched = Scheduler::getInstance();
    size_t nCores = sched->getCoreCount();
    Thread::initializeOpenMP(nCores);
#endif

#if defined(MTS_OPENMP)
    //  schedule(dynamic) removed
#pragma omp parallel for
#endif
    for (int i = 0; i < (int) gatherBlocks->size(); ++i) {  // For all gather points
      // Get the sampler associated
      // to the block
      ref<Sampler> sampler = samplers[i];

      // For all the gather points int the block
      // be carefull of the offset of the image plane
      GatherBlock & gb = (*gatherBlocks)[i];
      int xofs = (*offset)[i].x, yofs = (*offset)[i].y;
      int index = 0;
      for (int yofsInt = 0; yofsInt < blockSize; ++yofsInt) {
        if (yofsInt + yofs - cropOffset.y >= cropSize.y)
          continue;
        for (int xofsInt = 0; xofsInt < blockSize; ++xofsInt) {
          if (xofsInt + xofs - cropOffset.x >= cropSize.x)
            continue;

          // Get the gather point and clear it
          // (temp data erased) + prepare data
          // to sample the gp position
          Point2 apertureSample, sample;
          Float timeSample = 0.0f;  // TODO
          GatherPointsList &gatherPointList = (gb[index++]);
		  gatherPointList.clear();
		  gatherPointList.push_back();
		  // Select first gather point
		  GatherPoint * gatherPoint = &(gatherPointList[0]);
		  gatherPoint->pointIndex = 0;
		  gatherPoint->points = &gatherPointList;
          gatherPointList.resetTemp();  //< Reset temp associated values

          // Initialize the GP plane position
          gatherPointList.pos = Point2i(xofs + xofsInt, yofs + yofsInt);

          // (sample): Randomly select the pixel position
          // Special case to the first pass where the gp
          // are sent in the middle of the pixel
          // to compute the associated radii
          // TODO: Fix that as random number problem to other values
          if (needsApertureSample)
            apertureSample = Point2(0.5f);
          if (needsTimeSample)
            timeSample = 0.5f;

          sample = sampler->next2D();
          sample += Vector2((Float) gatherPointList.pos.x,
                            (Float) gatherPointList.pos.y);

          // Sample the primary ray from the camera
          RayDifferential rayCamera;
          sensor->sampleRayDifferential(rayCamera, sample, apertureSample,
                                        timeSample);
          RayDifferential ray = rayCamera;

          // Initialize data to bounce
          // through the scene
          Spectrum weight(1.0f);
          int depth = 1;
		  gatherPoint->directIllumBSDF = Spectrum(0.f);
		  gatherPoint->directIllumLightSample = Spectrum(0.f);
		  gatherPoint->completelyDelta = true;
		  gatherPointList.radius = 1;
		  gatherPointList.emission = Spectrum(0.f);
          Float traveledDistance = 0.f;  //< The total distance traveled
		  bool createdGP = false;
		  float radius = 0.f;

		  // Performs first intersection
		  scene->rayIntersect(ray, gatherPoint->its);

		  // Emitter is hit
		  if (gatherPoint->its.isValid()) {
			  if (gatherPoint->its.isEmitter()) 
				  gatherPointList.emission = gatherPoint->its.Le(-ray.d);
		  } else 
			  gatherPointList.emission = scene->evalEnvironment(ray);

		  // Invalid - for now
		  gatherPoint->depth = -1;

		  Float lastVertexForwardInversePdfSolidAngle = 1.0f;
		  Float lastCosine = 1.f;
		  Point lastVertexPos = ray.o;

		  // Temp index - increased for every merging gather point
		  int tempIndex = 0;

          // Bounce GP in the scene
          while (true) {

			if (!gatherPoint->its.isValid()) {
				// === If there is not intersection
				/* Generate an invalid sample */
				break;
			}
			
			// Compute radius using the total distance.
            traveledDistance += gatherPoint->its.t;
            

            // If we reach maximum depth,
            // put the gather point as invalid
            if (depth >= maxDepth && maxDepth != -1) {
				break;
            }

			// Get bsdf at intersection
			const BSDF *bsdf = gatherPoint->its.getBSDF();

			DirectSamplingRecord dRec(gatherPoint->its);

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */
			if (bsdf->getType() & BSDF::ESmooth && (techniques & MISHelper::DIRLIGHT)) {
				Spectrum value = scene->sampleEmitterDirect(dRec, sampler->next2D());
				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					/* Allocate a record for querying the BSDF */
					Vector wo = gatherPoint->its.toLocal(dRec.d);
					BSDFSamplingRecord bRec(gatherPoint->its, wo, ERadiance);

					/* Evaluate BSDF * cos(theta) */
					const Spectrum bsdfVal = bsdf->eval(bRec);

					/* Prevent light leaks due to the use of shading normals */
					if (!bsdfVal.isZero() && dot(gatherPoint->its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0) {

						/* Calculate prob. of having generated that direction
							using BSDF sampling */
						Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
							? bsdf->pdf(bRec) : 0;

						BSDFSamplingRecord bRecReverse(gatherPoint->its,wo,gatherPoint->its.wi, EImportance);
						Float reversePdf = bsdf->pdf(bRecReverse);

						// Store MIS info and value
						gatherPoint->directIllumLightSample = value * bsdfVal * weight;
						gatherPoint->lightSampleMIS.delta = false;
						gatherPoint->lightSampleMIS.m_lightWeight = 1.f / miWeight(dRec.pdf, bsdfPdf);
						Float distSqr = (gatherPoint->its.p - lastVertexPos).lengthSquared();
						gatherPoint->lightSampleMIS.m_lastButOneVertexReversePdf = lastCosine * reversePdf / distSqr;
						float emitPdf, directPdf, cosTheta;
						Ray ray(dRec.p,-dRec.d,dRec.time);
						ray.n = dRec.n;
						// Direct pdf is in area measure already, emitPdf is in solid angle measure
						emitter->pdfRay(ray, emitPdf, directPdf, cosTheta);
						float prob = scene->pdfEmitterDiscrete(emitter);
						directPdf *= prob;
						emitPdf *= prob;
						emitPdf *= std::abs(dot(dRec.d,gatherPoint->its.geoFrame.n));
						if ( !emitter->isEnvironmentEmitter() )
							emitPdf /= (gatherPoint->its.p - dRec.p).lengthSquared();
						gatherPoint->lightSampleMIS.m_lastVertexReversePdf = emitPdf / directPdf;
						assert(gatherPoint->lightSampleMIS.m_lastVertexReversePdf);
					}
				}
			}
            // Sample randomly one direction
            BSDFSamplingRecord bRec(gatherPoint->its, sampler);

			Float pdfComponent = 1.f;
			Point2 randomBSDFSample = sampler->next2D();
			int componentSelected = -1;
			bool roughtFunction = true;

			if ( techniques & MISHelper::SPPM_ONLY ) {
				// Make the bounce decision
				// This constant will be used for:
				//  - component selection
				//  - bounce decision

				// Select one component
				componentSelected = bsdf->sampleComponent(bRec, pdfComponent,
															randomBSDFSample,
                                                            m_bounceRoughness);
			

				// Check the PDF for sample an component
				// There is a problem when it's equal to 0
				// Arrives when ray leaking in RoughtPlastic
				if (pdfComponent == 0) {
				  //SLog(EWarn, "Component selection failed");
					// Debug
					if (componentSelected != -1) {
						SLog(
							EError,
							"Not the good component is returned for component selection");
					}
					break;
				}

				// Sanity check
				if (componentSelected == -1 && pdfComponent != 1.f) {
					SLog(EError, "All component is selected by the pdf is not 1");
				}

				// Query the roughness to know the bounce decision
				// if the componentSelected is all, the bounce decision
				// will be the same for all component, so take one of them
				// to query the roughness
				int componentQuery = (
					componentSelected == -1 ? 0 : componentSelected);
				roughtFunction = bsdf->getRoughness(gatherPoint->its,
														componentQuery)
                  >= m_bounceRoughness && (!(bRec.sampledType & BSDF::EDelta));

				// Sample with the selected component only
				bRec.component = componentSelected;
			}
			float bsdfPdf = 0.f;
			Spectrum bsdfValue = bsdf->sample(bRec, bsdfPdf, randomBSDFSample);
			bool deltaBounce = bRec.sampledType & BSDF::EDelta;
			roughtFunction &= !deltaBounce;
			bsdfPdf *= pdfComponent;
			Vector wo = gatherPoint->its.toWorld(bRec.wo),
				wi = gatherPoint->its.toWorld(gatherPoint->its.wi);

			BSDFSamplingRecord bRecReverse(gatherPoint->its,bRec.wo,gatherPoint->its.wi,EImportance);
			Float reversePdf = bsdf->pdf(bRecReverse);
			// TODO multiply bsdfPdf and reversePdf by RR probability
			// First compute forward and backward distance and cosine for solid angle to area measure converting
			Float distSqr = (gatherPoint->its.p - lastVertexPos).lengthSquared();
			Float distSqrForward, distSqrBackward, cosineForward, cosineBackward;
			VertexMISInfo &misInfo = gatherPoint->misInfo;
			// Delta surface on current vertex influences backward distance and cosine
			if ( deltaBounce ) {
				reversePdf = bsdfPdf = 1.f;
				distSqrBackward = cosineBackward = 1.f;
				misInfo.m_flags = VertexMISInfo::DELTA;
			}
			else {
				distSqrBackward = distSqr;
				cosineBackward = lastCosine;
				misInfo.m_flags = 0;
			}
			// Delta surface on previous vertex influences forward distance and cosine
			distSqrForward = distSqr;
			cosineForward = std::abs(dot(wi,gatherPoint->its.geoFrame.n));
			if (depth > 1)
			{
				VertexMISInfo &lastInfo = gatherPointList[depth-2].misInfo;
				if ( lastInfo.m_flags & VertexMISInfo::DELTA )
				{
					cosineForward = distSqrForward = 1.f;
				}
				lastInfo.m_vertexReversePdfWithoutBSDF = cosineBackward / distSqrBackward;
				lastInfo.m_vertexReversePdf = reversePdf * lastInfo.m_vertexReversePdfWithoutBSDF;
				assert(lastInfo.m_vertexReversePdf || bsdfValue.isZero());
			}
			misInfo.m_vertexInverseForwardPdf = lastVertexForwardInversePdfSolidAngle * distSqrForward / cosineForward; 
			assert(misInfo.m_vertexInverseForwardPdf);
			// May be changed in importance function update for more complicated importance functions
			misInfo.m_importance[0] = 1.f;
			misInfo.m_importance[1] = 1.f;
			lastCosine = std::abs(dot(wo,gatherPoint->its.geoFrame.n)); // Remember backward cosine
			lastVertexForwardInversePdfSolidAngle = 1.0f / bsdfPdf;
			lastVertexPos = gatherPoint->its.p;

			if (bsdfValue.isZero())
				break;

			// Trace rays and get emittance from a hit
			bool hitEmitter = false;
			Spectrum value(0.f);

			/* Trace a ray in this direction */
			// Update the bouncing ray
			// and update the depth of the GP
			ray = RayDifferential(gatherPoint->its.p,
				wo,
				ray.time);

			Intersection its;
			const Emitter * emitter;
			if (scene->rayIntersect(ray, its)) {
				/* Intersected something - check if it was a luminaire */
				if (its.isEmitter()) {
					value = its.Le(-ray.d);
					hitEmitter = true;
					emitter = its.shape->getEmitter();
				}
			} else {
				/* Intersected nothing -- perhaps there is an environment map? */
				const Emitter *env = scene->getEnvironmentEmitter();
				emitter = env;

				if (env) {
					value = env->evalEnvironment(ray);
					if (!env->fillDirectSamplingRecord(dRec, ray))
						break;
					hitEmitter = true;
				}
			}

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (hitEmitter && (techniques & MISHelper::DIRLIGHT) && !value.isZero()) {
				/* Compute the prob. of generating that direction using the
				   implemented direct illumination sampling technique */
				float emitPdf, directPdf, cosTheta;
				Ray rayE(its.p, -ray.d, ray.time);
				rayE.n = its.geoFrame.n;
				// Direct pdf is in area measure already, emitPdf is in solid angle measure
				emitter->pdfRay(rayE, emitPdf, directPdf, cosTheta);
				Float distSqr = (its.p - gatherPoint->its.p).lengthSquared();
				if ( emitter->isEnvironmentEmitter())
					distSqr = 1.f;
				directPdf *= distSqr / cosTheta; // Convert directPdf to solid angle measure
				float prob = scene->pdfEmitterDiscrete(emitter);
				directPdf *= prob;
				emitPdf *= prob;
				if (deltaBounce) {
					directPdf = 0.f;
					gatherPoint->bsdfMIS.m_lastVertexReversePdf = emitPdf * std::abs(dot(wo,gatherPoint->its.geoFrame.n)) / (bsdfPdf * distSqr);
				}
				else {
					gatherPoint->bsdfMIS.m_lastVertexReversePdf = emitPdf * std::abs(dot(wo,gatherPoint->its.geoFrame.n)) / (bsdfPdf * cosTheta);
				}
				// Store MIS info and value
				gatherPoint->directIllumBSDF = weight * (bsdfValue / pdfComponent) * value;
				gatherPoint->bsdfMIS.delta = deltaBounce;
				gatherPoint->bsdfMIS.m_lightWeight = 1.f / miWeight(bsdfPdf, directPdf);
				if (depth > 1)
					gatherPoint->bsdfMIS.m_lastButOneVertexReversePdf = gatherPointList[depth-2].misInfo.m_vertexReversePdf;
				else
					gatherPoint->bsdfMIS.m_lastButOneVertexReversePdf = 1.f;
				
				assert(gatherPoint->bsdfMIS.m_lastVertexReversePdf);
			}
			gatherPoint->weight = weight;
			gatherPoint->pdfComponent = pdfComponent;
			gatherPoint->sampledComponent = componentSelected;

			// Go inside only for rough functions if MERGING is enabled. Go inside only once for SPPM
            if ((techniques & MISHelper::MERGE) && (!createdGP || !(techniques & MISHelper::SPPM_ONLY)) &&
				(roughtFunction || (depth + 1 > maxDepth && maxDepth != -1)) &&
				(!PixelData<GatherPoint>::limitMaxGatherPoints || tempIndex < PixelData<GatherPoint>::maxGatherPoints)) {
				// If the gather reach an sufficient smooth BSDF
				// deposit it!
				gatherPoint->depth = depth;
				if(gatherPointList.scale == 0.f) {
					SLog(EError, "Zero scale on valid gather point");
				}
				// Compute the radius using the ray differential if this is the first gather point
				// Note that this radius is rescale by the scale factor of the gatherpoint
				if (!createdGP) {
					radius = getRadiusRayDifferential(rayCamera, traveledDistance);
					gatherPointList.radius = radius * gatherPointList.scale;
				}
				createdGP = true;
				gatherPoint->setTempIndex(tempIndex);
				++tempIndex;
            } 
			// Quits after first GP was created if we use pure SPPM
			if (createdGP && (techniques & MISHelper::SPPM_ONLY) && !(techniques & MISHelper::DIRLIGHT))
				break;
			
			// Prepare new gather point
			gatherPointList.push_back();
			gatherPoint = &(gatherPointList[depth-1]);
			GatherPoint * newGP = &(gatherPointList[depth]);
			newGP->points = &gatherPointList;
			newGP->pointIndex = depth;
			newGP->completelyDelta = gatherPoint->completelyDelta & deltaBounce;
			gatherPoint = newGP;

			// Store last intersection
			gatherPoint->its = its;
			gatherPoint->directIllumBSDF = Spectrum(0.f);
			gatherPoint->directIllumLightSample = Spectrum(0.f);
			gatherPoint->depth = -1;

			// Update the weight associated
			// to the GP
			weight *= bsdfValue / pdfComponent;

			++depth;

			// Russian roulette decision
			if (depth > 10) {
				Float q = std::min(weight.max(), (Float) 0.95f);
				if (sampler->next1D() >= q) {
					// RR decided to stop here
					// make invalid GP
					break;
				}
				// Scale to take into account RR
				weight /= q;
			}

		  } // End of while(true)

          sampler->advance();
        }
      }
    }  // End of the sampling

    //////////////////////////////////
    // Display extra informations
    // And in case not initialized GP
    // gets the maximum radii
    //////////////////////////////////
    // === Find max Size
    Float radiusMax = 0;
	for (size_t j = 0; j < gatherBlocks->size(); j++) {
		GatherBlock & gb = (*gatherBlocks)[j];
		for (size_t i = 0; i < gb.size(); ++i) {
			GatherPointsList & list = gb[i];
			radiusMax = std::max(radiusMax, list.radius);
		}
	}
    SLog(EInfo, "Finding max radius: %f", radiusMax);
  }

  void rescaleFlux() {
	if ( !(techniques & MISHelper::SPPM_ONLY) )
		return;
    for (size_t j = 0; j < gatherBlocks->size(); j++) {
      GatherBlock & gb = (*gatherBlocks)[j];
      for (size_t i = 0; i < gb.size(); ++i) {
        GatherPointsList & list = gb[i];
		for (GatherPointsList::iterator it = list.begin(); it != list.end(); ++it) {
			if (it->depth != -1 && it->its.isValid()) {
			  // Valid GP
			  if (list.radius == 0) {
				// No radius => Error because we will loose the flux
				SLog(EError, "Valid GP with null radius");
			  } else {
				list.flux *= list.radius * list.radius * M_PI;
			  }
			  break;
			} else {

			}
		}
      }
    }
  }

 protected:

  void resetInitialRadius(Float initialRadius) {
    SLog(EInfo, "Reset Initial radius to: %f", initialRadius);
    for (size_t j = 0; j < gatherBlocks->size(); j++) {
      GatherBlock & gb = (*gatherBlocks)[j];

      // === Create & Initialize all gather points in the block
      for (size_t i = 0; i < gb.size(); ++i) {
		  GatherPointsList & list = gb[i];
		  list.radius = initialRadius;
      }
    }
  }

  inline Float miWeight(Float pdfA, Float pdfB) const {
	  pdfA *= pdfA;
	  pdfB *= pdfB;
	  return pdfA / (pdfA + pdfB);
  }

 protected:
  // === Config attributs
  Scene* scene;
  int maxDepth;

  // === Gather blocks attributes
  GatherBlocks * gatherBlocks;
  const std::vector<Point2i>* offset;

  // === Sampler attributes
  ref_vector<Sampler> samplers;

  // In the reference mode, the sampler are initialized
  // randomly to be sure to not have the same
  // sequence of gather points generated
  bool referenceMod;

  // Used techniques
  int techniques;

  // Bounce constant decision
  Float m_bounceRoughness;

};

MTS_NAMESPACE_END
