#pragma once

#include "../importanceFunc.h"
#include "isppm_scalarFunction_proc.h"

MTS_NAMESPACE_BEGIN

#define NUM_DEFAULT_PASS 4

class ISPPM : public ImportanceFunction {
 public:
  ISPPM(const Properties &props)
      : ImportanceFunction(props, "isppm", false, false) {
    // === Debug
    // --- Dump false color images
    m_useFakeColor = false;
    // --- Dump debug image to see what produce ISPPM
    m_debugImages = props.getBoolean("debugImages", true);
    // --- Number of rendering pass to compute the importance
    m_nbPassImportance = props.getInteger("nbPassImportance", NUM_DEFAULT_PASS);

    // --- Some parameters from the paper
    // there are use to filter the importance map
    m_beta = props.getFloat("beta", 10.f);
    m_sigma = props.getFloat("sigma", 4.f);
    m_vLowRemove = props.getFloat("vLowRemove", 0.05);
    m_vHighRemove = props.getFloat("vHighRemove", 0.6);

    m_totalEmittedPath = 0;

    // --- Dump all values
    SLog(EInfo, " --- ISPPM:");
    SLog(EInfo, "   Nb Pass Importance          : %i", m_nbPassImportance);
    SLog(EInfo, "   Beta (Scale range)          : %f", m_beta);
    SLog(EInfo, "   Sigma (Extra weight)        : %f", m_sigma);
    SLog(EInfo, "   Quantity remove Low         : %f", m_vLowRemove);
    SLog(EInfo, "   Quantity remove High        : %f", m_vHighRemove);
    SLog(EInfo, "   Low pass Filter Size        : %f", m_lowPassFilterSize);

    // --- Error checking
    if (m_vLowRemove < 0 || m_vLowRemove >= 1.f) {
      SLog(EError, "vLow need to be a percentage");
    }
    if (m_vHighRemove < 0 || m_vHighRemove >= 1.f) {
      SLog(EError, "vHigh need to be a percentage");
    }
    if (m_vLowRemove + m_vHighRemove > 1.f) {
      SLog(EError, "vLow + vHigh != 1 (Not remove all the range)");
    }
  }
  virtual ~ISPPM() {
  }

  virtual void precompute(Scene* scene, RenderQueue *queue,
                          const RenderJob *job, int sceneResID,
                          int sensorResID) {
    precomputeInternal(scene, queue, job, sceneResID, sensorResID);
  }

  virtual void update(size_t idCurrentPass, size_t totalEmitted) {
#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
	  for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size(); ++blockIdx) {
		  GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		  for (size_t i = 0; i < gatherBlock.size(); ++i) {
			  GatherPointsList &gps = gatherBlock[i];
			  Float imp = gps.front().importance[m_idImportance];
			  for (GatherPointsList::iterator it = gps.rbegin(); it != gps.rend(); --it ) {
				  if (it->depth != -1 && it->its.isValid()) {
				      it->importance[m_idImportance] = imp;
					  it->misInfo.m_importance[m_idImportance] = imp;
				  }
				  else
					  it->misInfo.m_importance[m_idImportance] = 0.f;
			  }
		  }
	  }
  }

 protected:
  void precomputeInternal(Scene* scene, RenderQueue *queue,
                          const RenderJob *job, int sceneResID,
                          int sensorResID) {
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Timer> stepTimer = new Timer;
    ref<Sensor> sensor = scene->getSensor();
    ref<Film> film = sensor->getFilm();
    size_t nCores = sched->getCoreCount();

    SLog(EInfo, "ISPPM Precomputation procedure");

    // === Create bitmap
    m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
    m_bitmap->clear();

    // Create the independent sampler
    // used to shoot photons
    // make it completely random if needed
    Properties propsIndependent("independent");
    if(m_config.referenceMod) {
      propsIndependent.setBoolean("randInit", true);
    }
    ref<Sampler> samplerIndependent =
        static_cast<Sampler *>(PluginManager::getInstance()->createObject(
            MTS_CLASS(Sampler), propsIndependent));

    // To be sure that the generator will be not the same
    // by advancing in the sampling sequence
    for (size_t i = 0; i < sched->getCoreCount(); ++i) {
      samplerIndependent->next2D();
    }

    // Create the sampler
    std::vector<SerializableObject *> samplersIndependent(
        sched->getCoreCount());
    for (size_t i = 0; i < sched->getCoreCount(); ++i) {
      ref<Sampler> clonedIndepSampler = samplerIndependent->clone();
      clonedIndepSampler->incRef();
      samplersIndependent[i] = clonedIndepSampler.get();
    }
    int samplerIndependentResID = sched->registerMultiResource(
        samplersIndependent);

    ///////////////////////////////////
    // Step 1: Compute Scalar function
    ///////////////////////////////////
    int itCurrentPass = 0;

    // ==========================
    // Step 1.1: Collect Density
    // ==========================
    std::vector<bool> usedGatherPoint(m_bitmap->getPixelCount(), false);

#if defined(MTS_OPENMP)
    Thread::initializeOpenMP(nCores);
#endif

    // === Set ray differential to 3 in ISPPM
    // to be sure to collect enough photons
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
            ++blockIdx) {
          GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
          for (size_t i = 0; i < gatherBlock.size(); ++i) {
            GatherPointsList &gpl = (gatherBlock[i]);
			gpl.scale = 3.f;
			// radius will be regenerated in m_gpManager->regeneratePositionAndRadius();
			// below
          }
    }

    // To be sure that not producing the same sequence of GP
    // in the msppm core code, regenerate the sampler
    // and make an shift to don't have the same sequence
    m_gpManager->generateSamplers(1);
    m_running = true;
    while (m_running && (itCurrentPass < m_nbPassImportance)) {
      // === Distributes gather points in the scene
      m_gpManager->regeneratePositionAndRadius();
      //FIXME: No scale is needed here ?
      //m_gpManager->rescaleFlux();

      // === Create gather map
      SLog(EInfo, "Build the gather map... pass: %i", itCurrentPass + 1);
	  int tech = m_config.usedTechniques;
	  m_config.usedTechniques = MISHelper::SPPM_ONLY;
      ref<GatherPointMap> gatherMap = new GatherPointMap(*m_gatherBlocks,
														 scene,
														 NULL, // Only used for BPT/VCM
														 m_config,
														 PixelData<GatherPoint>::nbChains);
	  m_config.usedTechniques = tech;
      int gatherMapID = sched->registerResource(gatherMap);

      photonMapPassScalarFunction(++itCurrentPass, queue, job, film, sceneResID,
                                  sensorResID, samplerIndependentResID,
                                  gatherMapID, usedGatherPoint, nCores);

      // Free memory + Update
      sched->unregisterResource(gatherMapID);
    }

    // Then remake the original setup
    m_gpManager->generateSamplers(0);

    // ==========================
    // Step 1.2: Filter Density & Compute Statistics
    // ==========================
    Float vMax = 0.f;
    Float vMin = 100000.f;
    m_bitmap->clear();
    // --- Search max vMax and min vMin
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
      GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
      Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
      for (size_t i = 0; i < gatherBlock.size(); ++i) {
		GatherPointsList &gpl = (gatherBlock[i]);
		if (usedGatherPoint[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x]) {
			//gp->flux = gp->flux / ((Float) m_totalEmittedPath * gp->radius*gp->radius * M_PI);
			vMax = std::max(vMax, gpl.fluxDirect.getLuminance());
			vMin = std::min(vMin, gpl.fluxDirect.getLuminance());
			target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = gpl.fluxDirect;
		} else {
			target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = gpl.fluxDirect;
		}
      }
    }
    SLog(EInfo, "Values found vMin and vMax: [%f, %f]", vMin, vMax);

    if (m_debugImages) {
      ref<Bitmap> cloneBitmap = m_bitmap->clone();
      if (m_useFakeColor) {
        convertFalseColor(cloneBitmap, 0.f, 1.f);
      }
      film->clear();
      film->setBitmap(cloneBitmap);
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_scalarUnNorm";
      std::string path = ss.str();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }

    // --- Normalize data
    Float vRange = vMax - vMin;
    if(vRange == 0.f) {
      SLog(EError, "No range detected in ISPPM importance computation: [%f, %f]", vMin, vMax);
    } else {
      SLog(EInfo, "Range found: %f [%f, %f]", vRange, vMin, vMax);
    }
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
		GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		for (size_t i = 0; i < gatherBlock.size(); ++i) {
			GatherPointsList &gpl = (gatherBlock[i]);
			if (!usedGatherPoint[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x]) {  // Case of direct gather points
				gpl.fluxDirect = Spectrum(1.f);
			} else {
				gpl.fluxDirect = ((gpl.fluxDirect - Spectrum(vMin)) / vRange);  // < Normalisation to 1
			}
			target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = gpl.fluxDirect;
		}
    }

    // --- Compute vLow and vHigh
    Float vLow = m_vLowRemove;
    Float vHigh = m_vHighRemove;

    SLog(EInfo, "Values found vLow and vHigh: %f -> %f", vLow, vHigh);

    if (m_debugImages) {
      ref<Bitmap> cloneBitmap = m_bitmap->clone();
      if (m_useFakeColor) {
        convertFalseColor(cloneBitmap, 0.f, 1.f);
      }
      film->clear();
      film->setBitmap(cloneBitmap);
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_scalarNorm";
      std::string path = ss.str();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }

    // --- Normalize values between vLow and vHigh
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
		GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		for (size_t i = 0; i < gatherBlock.size(); ++i) {
			GatherPointsList &gpl = (gatherBlock[i]);
			Float lum = gpl.fluxDirect.getLuminance();
			if (lum > vHigh) {
				gpl.fluxDirect *= (vHigh / lum);
			} else if (lum < vLow) {
				if (lum != 0) {
					gpl.fluxDirect *= (vLow / lum);
				} else {
					gpl.fluxDirect = Spectrum(vLow);
				}
			}
			target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = gpl.fluxDirect;
      }
    }

    if (m_debugImages) {
      ref<Bitmap> cloneBitmap = m_bitmap->clone();
      if (m_useFakeColor) {
        convertFalseColor(cloneBitmap, 0.f, 1.f);
      }
      film->clear();
      film->setBitmap(cloneBitmap);
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_scalarClamp";
      std::string path = ss.str();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }

    // --- Gaussian filter pass
    applyGaussianFilter(vLow, vHigh);

    if (m_debugImages) {  // FIXME: Verification range
      ref<Bitmap> cloneBitmap = m_bitmap->clone();
      if (m_useFakeColor) {
        convertFalseColor(cloneBitmap, 0.f, 1.f);
      }
      film->clear();
      film->setBitmap(cloneBitmap);
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_scalarBlur";
      std::string path = ss.str();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }

    // --- Reput the values
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
	for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
		++blockIdx) {
		GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		for (size_t i = 0; i < gatherBlock.size(); ++i) {
			GatherPointsList &gpl = (gatherBlock[i]);
			Spectrum imp = target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x];
			gpl.fluxDirect = imp;
		}
    }

    // ==========================
    // Step 1.3: Compute Scalar Contribution Function
    // ==========================

    // --- Compute vMid
    Float vMid = (vLow + vHigh) / 2;

    // --- Compute Scalar function
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
		GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		for (size_t i = 0; i < gatherBlock.size(); ++i) {
			GatherPointsList &gpl = (gatherBlock[i]);
			float importance;
			Float lum = gpl.fluxDirect.getLuminance();
			if (lum >= vMid) {
				importance = 1.f;
			} else {
				importance = 1.f + m_beta * (expf(1.f - (lum / vMid)) - 1.f);
			}
			target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = Spectrum(
				importance);
			// Now redistribute importance to all gather points
			for (GatherPointsList::iterator gp = gpl.begin(); gp != gpl.end(); ++gp) {
				gp->importance[m_idImportance] = importance;
			}
		}
    }

    if (m_debugImages) {
      ref<Bitmap> cloneBitmap = m_bitmap->clone();
      if (m_useFakeColor) {
        convertFalseColor(cloneBitmap, 1.f, m_beta);
      }
      film->clear();
      film->setBitmap(cloneBitmap);
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_importanceScalar";
      std::string path = ss.str();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
      film->setDestinationFile(scene->getDestinationFile(), 0);
    }

    SLog(EInfo, "Time for Step 1: %f", (stepTimer->getMilliseconds() / 1000.f));

    // === Reinit Stats for next turn
    usedGatherPoint.clear();

    //////////////////////////
    // Free memory
    //////////////////////////
    for (size_t i = 0; i < samplersIndependent.size(); ++i)
      samplersIndependent[i]->decRef();
    sched->unregisterResource(samplerIndependentResID);

    m_running = false;
  }

  void photonMapPassScalarFunction(int it, RenderQueue *queue,
                                   const RenderJob *job, Film *film,
                                   int sceneResID, int sensorResID,
                                   int samplerResID, int gatherMapID,
                                   std::vector<bool>& gatherPointsUsed,
                                   size_t nCores) {
    SLog(EInfo, "Performing a photon mapping pass %i", it);
    ref<Scheduler> sched = Scheduler::getInstance();

    int granularity = 0;
    if(!m_config.referenceMod) {
      granularity = (int)std::max(
        (size_t) 1,
        m_config.photonCount / (Scheduler::getInstance()->getWorkerCount()));
    }

    /* Generate the photon and compute their contribution on the gathering points */
    ref<SplattingScalarFunctionPhotonProcess> proc =
        new SplattingScalarFunctionPhotonProcess(
            m_config.photonCount, granularity,
            m_config.maxDepth == -1 ? -1 : m_config.maxDepth - 1,
            m_config.rrDepth, job);

    proc->bindResource("scene", sceneResID);
    proc->bindResource("sensor", sensorResID);
    proc->bindResource("sampler", samplerResID);
    proc->bindResource("gathermap", gatherMapID);

    sched->schedule(proc);
    sched->wait(proc);

    // === Get some statistics
    m_totalEmittedPath += proc->getNbEmittedPath();

    film->clear();
    // No OpenMP here
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
		GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
		Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		for (size_t i = 0; i < gatherBlock.size(); ++i) {
			GatherPointsList &gpl = (gatherBlock[i]);
			for (GatherPointsList::iterator gp = gpl.begin(); gp != gpl.end(); ++gp) {
				if ( !gp->its.isValid() || gp->depth == -1)
					continue;

				if (it == 1) {
					gatherPointsUsed[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = false;
				}

				// M: Number of photons collected to this pass
				// N: Number of photons collected to the previous pass (weighted alpha)
				Float M = 0;
				Spectrum contrib(0.f);

				// === Collect to all thread photon stats
				for (size_t idThread = 0; idThread < nCores; idThread++) {
					M += (Float) gpl.tempM[idThread];
				}
				Spectrum flux = gpl.getFlux(0);
				gatherPointsUsed[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = true;

				//FIXME
				gpl.fluxDirect = (gpl.fluxDirect + flux);  // * ratio; // Remove the weight : gp->weight *
				gpl.N += M;
				contrib = gpl.fluxDirect / ((Float) m_totalEmittedPath * M_PI);  //  gp->radius * gp->radius

				// No filter, we write directly in the bitmap
				target[gpl.pos.y * m_bitmap->getWidth() + gpl.pos.x] = contrib;
				// Use only single gather point
				break;
			}
		}
    }

    /* Update the bitmap  */
    film->setBitmap(m_bitmap);
    queue->signalRefresh(job);
  }

 protected:
  void applyGaussianFilter(Float vMin, Float vMax) {
    Properties gaussianProps("gaussian");
    gaussianProps.setFloat("stddev", 2.f);
    ref<ReconstructionFilter> gaussianFilterW =
        static_cast<ReconstructionFilter *>(PluginManager::getInstance()
            ->createObject(MTS_CLASS(ReconstructionFilter), gaussianProps));
    ref<ReconstructionFilter> gaussianFilterH =
        static_cast<ReconstructionFilter *>(PluginManager::getInstance()
            ->createObject(MTS_CLASS(ReconstructionFilter), gaussianProps));

    Resampler<Float> scaleDownW(gaussianFilterW, ReconstructionFilter::EMirror,
                                m_bitmap->getWidth(), m_bitmap->getWidth());
    Resampler<Float> scaleDownH(gaussianFilterH, ReconstructionFilter::EMirror,
                                m_bitmap->getHeight(), m_bitmap->getHeight());

    ref<Bitmap> blurBitmap = m_bitmap->clone();
    m_bitmap->clear();

#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
    for (int y = 0; y < blurBitmap->getHeight(); ++y) {
      const Float *srcPtr = (Float *) blurBitmap->getUInt8Data()
          + y * blurBitmap->getWidth() * 3;
      Float *trgPtr = (Float *) m_bitmap->getUInt8Data()
          + y * m_bitmap->getWidth() * 3;

      scaleDownW.resampleAndClamp(srcPtr, 1, trgPtr, 1, 3, vMin, vMax);
    }

    blurBitmap = m_bitmap->clone();
    m_bitmap->clear();

#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
    for (int x = 0; x < blurBitmap->getWidth(); ++x) {
      const Float *srcPtr = (Float *) blurBitmap->getUInt8Data() + x * 3;
      Float *trgPtr = (Float *) m_bitmap->getUInt8Data() + x * 3;

      scaleDownH.resampleAndClamp(srcPtr, m_bitmap->getWidth(), trgPtr,
                                  m_bitmap->getWidth(), 3, vMin, vMax);
    }
  }

 protected:
  ref<Bitmap> m_bitmap;
  size_t m_totalEmittedPath;
  bool m_debugImages;
  bool m_useFakeColor;
  int m_nbPassImportance;

  // ISPPM related
  Float m_beta;
  Float m_sigma;
  Float m_vLowRemove;
  Float m_vHighRemove;
  Float m_lowPassFilterSize;  //< Percentage total image

};

MTS_NAMESPACE_END
