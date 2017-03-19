#include "msppm_proc.h"

#include <mitsuba/core/statistics.h>
#include "imp/localImp/localImp.h"

MTS_NAMESPACE_BEGIN

////////////////////////////////////////////////////////////////////////////////
/// Some statistics
////////////////////////////////////////////////////////////////////////////////
StatsCounter largeStepRatio("Primary sample space MLT", "Accepted large steps",
		EPercentage);
StatsCounter smallStepRatio("Primary sample space MLT", "Accepted small steps",
		EPercentage);
StatsCounter acceptanceRate("Primary sample space MLT",
		"Overall acceptance rate", EPercentage);

////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Worker
////////////////////////////////////////////////////////////////////////////////

MSPPMPhotonWorker::MSPPMPhotonWorker(size_t granularity,
		const MSPPMConfiguration& config, int idWorker) :
		ParticleTracer(config.maxDepth == -1 ? -1 : config.maxDepth - 1,
				config.rrDepth, false, false), m_workResult(NULL), m_granularity(
				granularity), m_idWorker(idWorker), m_gathermap(NULL),
				m_mcData(0), m_config(config) {
	m_checkinitial = false;
}

MSPPMPhotonWorker::MSPPMPhotonWorker(Stream *stream, InstanceManager *manager) :
		ParticleTracer(stream, manager) {
	m_granularity = stream->readSize();
	Log(EError, "No Network support");
}

ref<WorkProcessor> MSPPMPhotonWorker::clone() const {
	Log(EError, "No suport of worker cloning ... ");
	return 0;
}

void MSPPMPhotonWorker::prepare() {
	ParticleTracer::prepare(); // === The sampler is already the metropolis sampler

	// === Get all passed structures
	m_gathermap = static_cast<GatherPointMap *>(getResource("gathermap"));
	m_mcData = static_cast<MCData *>(getResource("mcdata"));
	m_mcData->checkPaths();

	m_impFunc = static_cast<ImportanceFunction*>(getResource("impFunc"));
	m_sampler = m_mcData->getSampler(0); // Initialize with the first sampler
}

void MSPPMPhotonWorker::serialize(Stream *stream,
		InstanceManager *manager) const {
	ParticleTracer::serialize(stream, manager);
	stream->writeSize(m_granularity);
	Log(EError, "Not implemented");
}

ref<WorkResult> MSPPMPhotonWorker::createWorkResult() const {
	return new GatherMSPPMPhotonResult(m_config.numberChains);
}

Float MSPPMPhotonWorker::misWeight(int impA, const PhotonSplattingList & path, bool isMISPsi) const {
  double sum = 0.f;
  std::vector<double> P(PixelData<GatherPoint>::nbChains, 0.f);
  for(int i = 0; i < PixelData<GatherPoint>::nbChains; i++) {
    P[i] = path.getImp(i) / m_gathermap->getNormalization(i);
    if(isMISPsi) {
      P[i] *= m_sWeightPsi[i];
      P[i] *= P[i]; // Powered MIS
    } else {
      P[i] *= m_sWeight[i];
      if(m_config.usePowerHeuristic) {
        P[i] *= P[i]; // Powered MIS
      }
    }

    sum += P[i];
  }
  return P[impA] / sum;
}

void MSPPMPhotonWorker::process(const WorkUnit *workUnit,
		WorkResult *workResult, const bool &stop) {
	// === Get the result object
	// and prepare it to accumulate some information
	m_workResult = static_cast<GatherMSPPMPhotonResult*>(workResult);
	m_workResult->clear();
	m_mcData->setWeightZero();

	// === Condition test to be sure that the
	// markov chain is valid. To check that, we check that
	// the current state impact almost one gather point.
	if (!m_checkinitial) {
		// Don't redo it for next call
		// (Object is destroyed at each iteration).
		m_checkinitial = true;
		m_mcData->checkImportance(m_gathermap);
	}

	initialization();

	// === Call the internal process procedure
	// This is cast serveral new path in the scene
	// For each path:
	//   - Call handleNewPath() && handleNewParticle()
	//   - Call handleSurfaceInteraction() for each surface interaction
	//   - Call handleFinishParticule() at the end to make the metropolis decision
	ParticleTracer::process(workUnit, workResult, stop);

	//for(int idImportance = 0; idImportance < 2; idImportance++){

	// === Last splatting before exit
	for(int idChain = 0; idChain < m_config.numberChains; idChain++) {
    if (m_mcData->getWeight(idChain) != 0) {
        splatPath(idChain, m_mcData->getPath(idChain)->current, m_mcData->getWeight(idChain));
    }
  }

	m_workResult = 0;
}


void MSPPMPhotonWorker::splatPath(int idImportance, PhotonSplattingList* path, Float w) {
  Float weightMIS = 1.f;
  Float weightMISPsi = 1.f;
  if(m_config.useMISLevel) {
      weightMIS = misWeight(idImportance,*path);
  }
  if(m_config.phiStatisticStrategy == 2) {
    weightMISPsi = misWeight(idImportance,*path, true);
  }

  // Cancel MIS
  if( m_config.showUpperLevels) {
    weightMIS = 1.f;
  }

  Float powerMult = w / path->getImp(idImportance);
  m_gathermap->updateGPImpacted(*path, powerMult,
      m_idWorker, w, idImportance, weightMIS, weightMISPsi, m_sampleConfig.isUniform,
      m_mcData->getSampler(idImportance)->getRandom()->nextFloat() > 0.5);
}

void MSPPMPhotonWorker::handleNewPath() {
  // Change the sampling ID
  m_sampleConfig = newSample();
  m_sampler = m_sampleConfig.sampler;
  m_mcData->getPath(m_sampleConfig.idImp)->proposed->clear();
}

void MSPPMPhotonWorker::handleNewParticle(const Ray ray,const Emitter &emitter) {
	VertexMISInfo info;
	info.m_flags = emitter.isEnvironmentEmitter() ? VertexMISInfo::DISTANT_LIGHT : 0;
	float emitPdf, directPdf, cosTheta;
	emitter.pdfRay(ray, emitPdf, directPdf, cosTheta);
	float prob = m_scene->pdfEmitterDiscrete(&emitter);
	directPdf *= prob;
	emitPdf *= prob;
	info.m_vertexInverseForwardPdf = 1.f / directPdf;
	info.m_importance[0] = 0.f;
	info.m_importance[1] = 0.f;
	info.m_importance[2] = 0.f;
	m_lastVertexForwardInversePdfSolidAngle = directPdf / emitPdf;
	m_lastCosine = cosTheta;
	m_lastVertexPos = ray.o;
	PhotonInfo p;
	p.misInfo = info;
	m_mcData->getPath(m_sampleConfig.idImp)->proposed->add(p);
}

void MSPPMPhotonWorker::handleSurfaceInteraction(int depth_,
		int nullInteractions, bool delta, const Intersection &its,
		const Medium *medium, const Spectrum &weight, const Vector & wo) {

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
	p.power = weight;
	p.depth = depth;

	// TODO multiply bsdfPdf and reversePdf by RR probability
	// First compute forward and backward distance and cosine for solid angle to area measure converting
	VertexMISInfo &lastInfo = m_mcData->getPath(m_sampleConfig.idImp)->proposed->photons.back().misInfo;
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
	// Update current vertex MIS info
	p.misInfo.m_vertexInverseForwardPdf = m_lastVertexForwardInversePdfSolidAngle * forwardAreaMeasure;
	assert(lastInfo.m_vertexInverseForwardPdf);

	for(int idChain = 0; idChain < m_config.numberChains; idChain++) {
      p.misInfo.m_importance[idChain] = m_mcData->getPath(m_sampleConfig.idImp)->proposed->getImp(idChain);
  }

	m_lastVertexPos = its.p;
	m_lastVertexForwardInversePdfSolidAngle = 1.0f / forwardPdf;
	m_lastCosine = std::abs(dot(wo,its.geoFrame.n));

	m_mcData->getPath(m_sampleConfig.idImp)->proposed->add(p);

	// TODO: See this !
	// (Because it can be an issue for MIS, no ?)
	if (deltaSurf)
		return;

	// Test if the current interaction impact an gather point
	// If there is an interaction (imp != 0), store the interaction point
	// and update the importance of the path.
	ImportanceRes imp = m_gathermap->queryGPImpactedImportance(its, depth, m_sampleConfig.idImp);
	for(int idChain = 0; idChain < m_config.numberChains; idChain++) {
    m_mcData->getPath(m_sampleConfig.idImp)->proposed->setImp(idChain,
                                                std::max(imp.importances[idChain],
                                                m_mcData->getPath(m_sampleConfig.idImp)->proposed->getImp(idChain)));
    m_mcData->getPath(m_sampleConfig.idImp)->proposed->photons.back().misInfo.m_importance[idChain] = m_mcData->getPath(m_sampleConfig.idImp)->proposed->getImp(idChain);
  }
}

void MSPPMPhotonWorker::handleFinishParticule() {
  beforeMH();
	EMCMCOperationStatus res = doMH(m_sampleConfig.idImp);
	afterMH(res);

	/////////////////////////
	// Finally update some statistiques
	/////////////////////////
	acceptanceRate.incrementBase();
	if (m_sampleConfig.isUniform) {
		largeStepRatio.incrementBase();
	} else {
		smallStepRatio.incrementBase();
	}

	if (res == EAccepted) {
		acceptanceRate += 1;
		if (m_sampleConfig.isUniform) {
			largeStepRatio += 1;
		} else {
			smallStepRatio += 1;
		}
	}
}

MSPPMPhotonWorker::EMCMCOperationStatus MSPPMPhotonWorker::doMH(int idImp) {
  // Count statistic
  if(m_mcData->getPath(idImp)->current->getImp(idImp) != 0.f && takeContribution()) {
    m_workResult->nextEmittedPath(idImp);
  }


  if (m_mcData->getPath(idImp)->proposed->getImp(idImp) == 0.f) {

    ///////////////////////////////////////////////////////////////
    // If the path is empty (imp == 0)
    // we reject it directly
    ///////////////////////////////////////////////////////////////

    // Computation of the weight
    // of the current path (not the proposed)
    if(takeContribution()) {
      m_mcData->increaseWeight(idImp, 1);
    }
    m_mcData->getSampler(idImp)->reject();
    if (m_mcData->getPath(idImp)->current->getImp(idImp) == 0.f) {
      return EInvalid; // Said that is invalid.
    } else {
      return ERejected; // Ok just rejected
    }
  } else {

    ///////////////////////////////////////////////////////////////
    // If the path is not empty (imp != 0)
    // Make some Metropolis related computation
    ///////////////////////////////////////////////////////////////

    // =============
    // Special case:
    // If the current (not proposed) path have no importance
    // This is mean that there is an problem in the initialisation
    // So we take the proposed path as the new competly skip
    // the rest of the computation
    // =============
    if (m_mcData->getPath(idImp)->current->getImp(idImp) == 0.f) { // Handling no initialisation
      Log(EWarn, "Find new valid path (%i): %f (w: %f)", idImp,
          m_mcData->getPath(idImp)->proposed->getImp(idImp),
          m_mcData->getWeight(idImp));
      m_mcData->setWeight(idImp,1);
      m_mcData->getSampler(idImp)->accept();
      m_mcData->getPath(idImp)->swap();
      // FIXME: Need this for the second chain !
      return EInvalid;
    }

    // Compute the Metropolis ratio
    Float a = std::min(1.f,
                       m_mcData->getPath(idImp)->proposed->getImp(idImp) /
                       m_mcData->getPath(idImp)->current->getImp(idImp));
    // Update the weight associated to the current state
    if (m_config.useExpectedValue) {
      m_mcData->increaseWeight(idImp, 1 - a);
    }

    if(a > m_mcData->getSampler(idImp)->getRandom()->nextFloat()) {
      newValidContribution(idImp);
      // =============
      // The proposed path (state) is accepted
      // =============

      // Accept the move
      m_mcData->getSampler(idImp)->accept();

      // If the current path weight is non zero
      // Splat it's contribution
      if (m_mcData->getWeight(idImp) != 0 && takeContribution()) {
        splatPath(idImp, m_mcData->getPath(idImp)->current, m_mcData->getWeight(idImp));
      }

      // Compute the new weight of the current path
      if(takeContribution()) {
        if (m_config.useExpectedValue) {
            m_mcData->setWeight(idImp,a);
        } else {
            m_mcData->setWeight(idImp,1);
        }
      }
      // Swap the paths (proposed -> current)
      m_mcData->getPath(idImp)->swap();
      return EAccepted;
    } else {
      // =============
      // The proposed path (state) is rejected
      // =============
      m_mcData->getSampler(idImp)->reject();

      // If we use expected value
      // Splat the contribution from the proposed path
      if(takeContribution()) {
        if (m_config.useExpectedValue) {
          if (a != 0) {
            splatPath(idImp, m_mcData->getPath(idImp)->proposed, a);
          }
        } else { //< No excepted value
          m_mcData->increaseWeight(idImp,1);
        }
      }

      return ERejected;
    }
  }
}

MSPPMPhotonWorker::EMCMCOperationStatus MSPPMPhotonWorker::doRE(int idImp0, int idImp1 ) {
  if((m_mcData->getPath(idImp0)->current->getImp(idImp0) == 0 ||
     m_mcData->getPath(idImp1)->current->getImp(idImp1) == 0) &&
     takeContribution()) {
    return EInvalid;
  } else {
    // R: Replica exchange prob
    Float upFactor =   m_mcData->getPath(idImp0)->current->getImp(idImp1)*m_mcData->getPath(idImp1)->current->getImp(idImp0);
    Float downFactor = m_mcData->getPath(idImp0)->current->getImp(idImp0)*m_mcData->getPath(idImp1)->current->getImp(idImp1);
    Float R =  std::min(1.f, upFactor/downFactor);

    if(R > m_mcData->getSampler(m_sampleConfig.idImp)->getRandom()->nextFloat()) {
       // Splat the two chains
       if (m_mcData->getWeight(idImp0) != 0) {
         splatPath(idImp0, m_mcData->getPath(idImp0)->current, m_mcData->getWeight(idImp0));
         m_mcData->setWeight(idImp0,0);
       }
       if (m_mcData->getWeight(idImp1) != 0) {
        splatPath(idImp1, m_mcData->getPath(idImp1)->current, m_mcData->getWeight(idImp1));
        m_mcData->setWeight(idImp1,0);
       }

       // Swap the chains
       m_mcData->swapChains(idImp0,idImp1);

       return EAccepted;
    } else {
      return ERejected;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Splatting Photon Process
////////////////////////////////////////////////////////////////////////////////

SplattingMSPPMPhotonProcess::SplattingMSPPMPhotonProcess(size_t photonCount,
		size_t granularity, const MSPPMConfiguration& config,
		const void *progressReporterPayload) :
		ParticleProcess(ParticleProcess::ETrace, photonCount, granularity,
				"Photon shooting", progressReporterPayload), m_photonCount(
				photonCount), m_config(config), m_results(config.numberChains) {
	// === Count the number of worker for unique id assignement
	m_idWorker = 0;
}

ref<WorkProcessor> SplattingMSPPMPhotonProcess::createWorkProcessor() const {
  if(m_config.numberChains == 1) {
    return new MSPPMPhotonWorkerOneChain(m_granularity, m_config, m_idWorker++);
  } else if(m_config.numberChains == 2) {
    return new MSPPMPhotonWorkerTwoChain(m_granularity, m_config, m_idWorker++);
  } else if(m_config.numberChains == 3) {
    return new MSPPMPhotonWorkerThreeChain(m_granularity, m_config, m_idWorker++);
  } else {
    SLog(EError, "Impossible WORKER");
    return 0;
  }
}

void SplattingMSPPMPhotonProcess::processResult(const WorkResult *wr,
		bool cancelled) {
	if (cancelled)
		return;
	const GatherMSPPMPhotonResult &vec =
			*static_cast<const GatherMSPPMPhotonResult *>(wr);
	LockGuard lock(m_resultMutex);

	// Update stats
	m_results.add(vec);

	// Internal Mitsuba update for the rendering procedure
	increaseResultCount(vec.getTotalEmittedPaths());
}

MTS_IMPLEMENT_CLASS(SplattingMSPPMPhotonProcess, false, ParticleProcess);
MTS_IMPLEMENT_CLASS_S(MSPPMPhotonWorkerOneChain, false, ParticleTracer);
MTS_IMPLEMENT_CLASS_S(MSPPMPhotonWorkerTwoChain, false, ParticleTracer);
MTS_IMPLEMENT_CLASS_S(MSPPMPhotonWorkerThreeChain, false, ParticleTracer);
MTS_IMPLEMENT_CLASS(GatherMSPPMPhotonResult, false, WorkResult);

MTS_NAMESPACE_END
