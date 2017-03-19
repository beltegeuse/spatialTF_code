#pragma once

#include <fstream>

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/render/gatherproc.h>

#include "splatting.h"

#include "msppm_sampler.h"
#include "msppm_path.h"
#include "imp/importanceFunc.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Work Result
//////////////////////////////////////////////////////////////////////////
class GatherMSPPMPhotonResult: public WorkResult {
public:
	GatherMSPPMPhotonResult(int nbChains): WorkResult(),
	  m_nbEmittedPath(nbChains, 0.f),
	  m_nbNormalisationPath(nbChains, 0.f),
	  m_accumulatedImp(nbChains, 0.f)
	{
		m_contributingUniform = 0;
	}

  //////////////////////////////////////////////////////////////////////////
  // Pure virtual impl methods
  //////////////////////////////////////////////////////////////////////////
  void load(Stream* stream) {
      Log(EError,"No serialization implemented ... ");
  }
  void save(Stream* stream) const {
      Log(EError,"No serialization implemented ... ");
  }
  std::string toString() const {
      return "GatherISPPMPhotonResult[NULL]";
  }

  //////////////////////////////////////////////////////////////////////////
  // Use full methods
  //////////////////////////////////////////////////////////////////////////
  void add(const GatherMSPPMPhotonResult& res) {
    for(size_t i = 0; i < m_nbEmittedPath.size(); i++) {
      m_nbEmittedPath[i] += res.m_nbEmittedPath[i];
      m_nbNormalisationPath[i] += res.m_nbNormalisationPath[i];
      m_accumulatedImp[i] += res.m_accumulatedImp[i];
    }

    m_contributingUniform += res.m_contributingUniform;
  }

  /// Clear all results data
  void clear() {
    for(size_t i = 0; i < m_nbEmittedPath.size(); i++) {
      m_nbEmittedPath[i] = 0;
      m_nbNormalisationPath[i] = 0;
      m_accumulatedImp[i] = 0.f;
    }

    m_contributingUniform = 0;
  }

  void nextEmittedPath(int idImportance) {
    m_nbEmittedPath[idImportance]++;
  }

  Float accumulatedImp(int idImportance) const {
    return m_accumulatedImp[idImportance];
  }

  void addImportance(Float importance, int idImportance) {
    m_accumulatedImp[idImportance] += importance;
  }

  size_t getNbEmittedPath(int idImportance) const {
    return m_nbEmittedPath[idImportance];
  }

  size_t getUniformEmittedPath(int idImportance) const {
    return m_nbNormalisationPath[idImportance];
  }

  void nextUniformEmittedPath(int idImportance) {
    m_nbNormalisationPath[idImportance]++;
  }

  void nextContributingUniform() {
    ++m_contributingUniform;
  }

  size_t getContributingUniformPaths() const {
    return m_contributingUniform;
  }


  //////////////////////////////////////////////////////////////////////////
  // Special getter
  //////////////////////////////////////////////////////////////////////////
  Float getNormalisation(int idImportance, int nbChains) const {
    if(nbChains <= idImportance) {
      // Impossible, just throw an exception
      Log(EError,"Not possible to query normalisation factor of the %i chain level (levels: %i)",
          idImportance, nbChains);
    }

    return m_accumulatedImp[idImportance] / m_nbNormalisationPath[idImportance];
  }

  size_t getTotalEmittedPaths() const {
    size_t total = 0;
    for(size_t i = 0; i < m_nbEmittedPath.size(); i++) {
      total += m_nbEmittedPath[i];
    }
    return total;
  }

  MTS_DECLARE_CLASS()

protected:
  std::vector<size_t> m_nbEmittedPath; // Nb emitted path for the two MC
  std::vector<size_t> m_nbNormalisationPath; // Nb emitted path to compute the normalisation factor
  std::vector<Float> m_accumulatedImp; // Accumulate importance to compute the normalisation factor
  size_t m_contributingUniform; // Contributing uniform paths count
};


//////////////////////////////////////////////////////////////////////////
// Worker
//////////////////////////////////////////////////////////////////////////

class MSPPMPhotonWorker : public ParticleTracer {
 public:
  enum EMCMCOperationStatus {
    EAccepted,
    ERejected,
    EInvalid
  };

 public:

	MSPPMPhotonWorker(size_t granularity, const MSPPMConfiguration& config, int idWorker);
	MSPPMPhotonWorker(Stream *stream, InstanceManager *manager);
  ref<WorkProcessor> clone() const;
  void serialize(Stream *stream, InstanceManager *manager) const;
  ref<WorkResult> createWorkResult() const;

  virtual void prepare();
  void process(const WorkUnit *workUnit, WorkResult *workResult,
      const bool &stop);

  void handleMediumInteraction(int depth, int nullInteractions, bool delta,
      const MediumSamplingRecord &mRec, const Medium *medium,
      const Vector &wi, const Spectrum &weight) {
      // === No Volume support
      Log(EError, "No support of volume rendering");
  }
  void handleFinishParticule();
  void handleNewPath();
  void handleNewParticle(const Ray ray,const Emitter &emitter);
  void handleSurfaceInteraction(int depth_, int nullInteractions, bool delta,
      const Intersection &its, const Medium *medium,
      const Spectrum &weight, const Vector & wo);

 protected:

  /**
   * Functions to change sampling
   * behavior
   */
  struct MCMCSampleConfig {
    int idImp;
    bool isUniform;
    Sampler* sampler;
  };

  virtual MCMCSampleConfig newSample() = 0;
  virtual void beforeMH() = 0;
  virtual void afterMH(EMCMCOperationStatus res) = 0;
  virtual void initialization() {}

  void newValidContribution(int idImp) {
    if(m_needCancelPhotons) {
      m_cancelPhotons[idImp] = std::max(0, m_cancelPhotons[idImp]-1);
      int i = 0;
      while(i < m_cancelPhotons.size() && m_cancelPhotons[i] == 0) { i++; }
      m_needCancelPhotons = (i != m_cancelPhotons.size());
    }
  }
  bool takeContribution() const {
    return !m_needCancelPhotons;
  }
 protected:

  /// Virtual destructor
  virtual ~MSPPMPhotonWorker() { }

  /// Helpers
  void splatPath(int idImportance, PhotonSplattingList* path, Float w);
  EMCMCOperationStatus doRE(int idImp1, int idImp2 );
  EMCMCOperationStatus doMH(int idImp);
  Float misWeight(int impA, const PhotonSplattingList & path, bool isMISPsi = false) const;

protected:
  // === Worker basic information
  GatherMSPPMPhotonResult* m_workResult;
  size_t m_granularity;
  int m_idWorker;

  bool m_checkinitial; //< If the chain is initialized or not?
  MCMCSampleConfig m_sampleConfig;


  GatherPointMap* m_gathermap;
  MCData* m_mcData;

  // For MIS (VCM)
  Float m_lastVertexForwardInversePdfSolidAngle;
  Float m_lastCosine;
  Point m_lastVertexPos;

  MSPPMConfiguration m_config;
  ImportanceFunction* m_impFunc;

  // --- MIS weights for photons contributions
  std::vector<Float> m_sWeight;
  // --- MIS weights to compute psi statistics
  std::vector<Float> m_sWeightPsi;
  // --- The number of photons to cancel to avoid startup bias
  std::vector<int> m_cancelPhotons;
  bool m_needCancelPhotons;
};

/**
 * This class refer to 2 chains in the paper
 * Uniform and a custom TF.
 * This implementation is use for:
 *  - Ours with only 2 chains
 *  - Hachisuka and Jensen 2011
 *  - Chen et al. 2011
 */
class MSPPMPhotonWorkerOneChain : public MSPPMPhotonWorker {
 public:
  MSPPMPhotonWorkerOneChain(size_t granularity, const MSPPMConfiguration& config, int idWorker):
    MSPPMPhotonWorker(granularity, config, idWorker) {
    m_lastAccepted = false;
    m_idSampling = 1;
    m_needCancelPhotons = false;
  }

  MSPPMPhotonWorkerOneChain(Stream *stream, InstanceManager *manager):
    MSPPMPhotonWorker(stream, manager) {
  }

  MTS_DECLARE_CLASS()

 protected:
  MCMCSampleConfig newSample() {
    // Choose between uniform and MCMC
    if(m_idSampling == 1) {
      // MCMC before
      m_idSampling = 0; // go to uniform
    } else {
      // Uniform before
      if(!m_lastAccepted) {
        m_idSampling = 1; // change only if not accepted
      }
    }

    // Change sampler behavior if uniform is choose or not
    MCMCSampleConfig s;
    s.sampler = m_mcData->getSampler(0);
    s.idImp = 0;
    s.isUniform = (m_idSampling == 0);
    m_mcData->getSampler(0)->setLargeStep(m_idSampling == 0);

    return s;
  }

  virtual void beforeMH() {
    if (m_idSampling == 0) {
      // This statistic is to count the number of uniform path
      // This is used to estimate the current normalisation factor
      m_workResult->nextUniformEmittedPath(0);
      m_workResult->addImportance(m_mcData->getPath(0)->proposed->getImp(0), 0);
    }
  }

  virtual void afterMH(EMCMCOperationStatus res) {
    m_lastAccepted = (res == EAccepted);

    // Statistic count
    if ( m_lastAccepted && m_idSampling == 0) {
      m_workResult->nextContributingUniform();
    }
  }


 private:
  bool m_lastAccepted;
  int m_idSampling;
};

/**
 * This class refer to 3 chains in the paper.
 * Uniform with 2 custom TF.
 */
class MSPPMPhotonWorkerTwoChain : public MSPPMPhotonWorker {
 public:
  MSPPMPhotonWorkerTwoChain(size_t granularity, const MSPPMConfiguration& config, int idWorker):
    MSPPMPhotonWorker(granularity, config, idWorker) {
    m_idSampling = 2;
    m_sWeight.push_back(2.f);
    m_sWeight.push_back(1.f);
    m_sWeightPsi.push_back(2.f);
    m_sWeightPsi.push_back(1.f);

    m_cancelPhotons.push_back(config.cancelPhotons);
    m_cancelPhotons.push_back(config.cancelPhotons);

    m_needCancelPhotons = config.cancelPhotons != 0;
  }

  MSPPMPhotonWorkerTwoChain(Stream *stream, InstanceManager *manager):
    MSPPMPhotonWorker(stream, manager) {
  }

  MTS_DECLARE_CLASS()

 protected:

  MCMCSampleConfig newSample() {
    // Just change for everthings
    m_idSampling += 1;
    if(m_idSampling > 2) {
      m_idSampling = 0; // Return to uniform
    }

    // Fille MCMCSample Config
    MCMCSampleConfig s;
    if(m_idSampling == 0 || m_idSampling == 1) {
      s.idImp = 0;
      s.isUniform = (m_idSampling == 0);
      s.sampler = m_mcData->getSampler(0);
      m_mcData->getSampler(0)->setLargeStep(m_idSampling == 0);
     } else {
       s.idImp = 1;
       s.isUniform = false;
       s.sampler = m_mcData->getSampler(1);
       m_mcData->getSampler(1)->setLargeStep(false);
     }

    return s;
  }

  virtual void beforeMH() {
     if (m_idSampling == 0) {
       // This statistic is to count the number of uniform path
       // This is used to estimate the current normalisation factor
       m_workResult->nextUniformEmittedPath(0);
       m_workResult->addImportance(m_mcData->getPath(0)->proposed->getImp(0), 0);
     }
   }

   virtual void afterMH(EMCMCOperationStatus res) {
     // Make RE
     if(m_idSampling == 2) {
       doRE(0, 1);
     }

     // Use upper level to compute normalisation factor
     if((m_idSampling == 0 && res == EAccepted) || m_idSampling == 1) {
         if(m_mcData->getPath(0)->current->getImp(0) != 0) {
              m_workResult->nextUniformEmittedPath(1);
              m_workResult->addImportance(m_mcData->getPath(0)->current->getImp(1) / m_mcData->getPath(0)->current->getImp(0), 1);
           } else {
               //SLog(EInfo, "Zero importance!");

               // This case is due to not well initialized chain.
               // We arrived in this condition if the current technique (idSampling == 1), completly failed.
               // We need to found other way to handle correctly this condition.
           }
     }

     // Statistic count
     if (m_idSampling == 0 && res == EAccepted) {
       m_workResult->nextContributingUniform();
     }
   }

 private:
  int m_idSampling;
};

/**
 * This class refer to 4 chains in the paper.
 * Uniform with 3 custom TF.
 */
class MSPPMPhotonWorkerThreeChain : public MSPPMPhotonWorker {
 public:
  MSPPMPhotonWorkerThreeChain(size_t granularity, const MSPPMConfiguration& config, int idWorker):
    MSPPMPhotonWorker(granularity, config, idWorker) {
    m_idSampling = 2;
    m_useVisibility = false;

    m_sWeight.push_back(1.5f);
    m_sWeight.push_back(0.5f);
    m_sWeight.push_back(1.f);

    m_sWeightPsi.push_back(1.f);
    m_sWeightPsi.push_back(1.f);
    m_sWeightPsi.push_back(0.f);

    m_cancelPhotons.push_back(config.cancelPhotons/2);
    m_cancelPhotons.push_back(config.cancelPhotons/2);
    m_cancelPhotons.push_back(config.cancelPhotons);

    m_needCancelPhotons = config.cancelPhotons != 0;
  }

  MSPPMPhotonWorkerThreeChain(Stream *stream, InstanceManager *manager):
    MSPPMPhotonWorker(stream, manager) {
  }

  MTS_DECLARE_CLASS()

 protected:
  MCMCSampleConfig newSample() {
   // Just change for everthings
   m_idSampling += 1;
   if(m_idSampling > 2) {
     m_idSampling = 0; // Return to uniform
   }

   // Fille MCMCSample Config
   MCMCSampleConfig s;
   if(m_idSampling == 0) {
     s.idImp = 0;
     s.isUniform = true;
     s.sampler = m_mcData->getSampler(0);
     m_mcData->getSampler(0)->setLargeStep(true);
   } else if(m_idSampling == 1) {
     m_useVisibility = !m_useVisibility;
     if(m_useVisibility) {
       s.idImp = 0;
       s.sampler = m_mcData->getSampler(0);
     } else {
       s.idImp = 1;
       s.sampler = m_mcData->getSampler(1);
     }

     s.isUniform = false;
     m_mcData->getSampler(s.idImp)->setLargeStep(false);
   } else {
      s.idImp = 2;
      s.isUniform = false;
      s.sampler = m_mcData->getSampler(2);
      m_mcData->getSampler(2)->setLargeStep(false);
    }

   return s;
 }

  virtual void initialization() {
    if(m_config.correctMIS) {
      // Based on the previous normalisation for the visibility TF
      // We can compute the right MIS weight
      // This weight will be used for uniform and visibility samples.
      Float bVis = m_gathermap->getNormalization(0);
      SLog(EInfo, "Bvis: %f", bVis);
      m_sWeight[0] = 0.5 + bVis*1.f;
      m_sWeightPsi[0] = 1.f + bVis;
    }
  }

  virtual void beforeMH() {
    // Estimate normalisation factor for the two intermediate level
    if (m_idSampling == 0) {
      // This statistic is to count the number of uniform path
      // This is used to estimate the current normalisation factor
      m_workResult->nextUniformEmittedPath(0);
      m_workResult->addImportance(m_mcData->getPath(0)->proposed->getImp(0), 0);
      if(!m_config.strongNormalisation) {
        m_workResult->nextUniformEmittedPath(1);
        m_workResult->addImportance(m_mcData->getPath(0)->proposed->getImp(1), 1);
      }
    }
  }

  virtual void afterMH(EMCMCOperationStatus res) {
    // Make RE (with the upper level and invRadii)
    if(m_idSampling == 2) {
      doRE(1, 2);
    } else if(m_idSampling == 1) {
      if(m_config.REAllTime) {
        doRE(0, 1);
      } else {
        if(!m_useVisibility) {
          doRE(0, 1);
        }
      }
    }

    if(m_config.strongNormalisation) {

      if(m_idSampling == 1 && m_useVisibility) {
        if(m_mcData->getPath(0)->current->getImp(0) != 0) {
          m_workResult->nextUniformEmittedPath(1);
          m_workResult->addImportance(m_mcData->getPath(0)->current->getImp(1) / m_mcData->getPath(0)->current->getImp(0), 1);
        }
      }

      if(m_idSampling == 1 && !m_useVisibility) {
        if(m_mcData->getPath(1)->current->getImp(1) != 0) {
          m_workResult->nextUniformEmittedPath(2);
          m_workResult->addImportance(m_mcData->getPath(1)->current->getImp(2) / m_mcData->getPath(1)->current->getImp(1), 2);
        }
      }

    } else {

      // Use upper level to compute normalisation factor
      // (m_idSampling == 0 && res == EAccepted) ||
      if(m_idSampling == 1 && !m_useVisibility) {
         if(m_mcData->getPath(1)->current->getImp(1) != 0) {
              m_workResult->nextUniformEmittedPath(2);
              m_workResult->addImportance(m_mcData->getPath(1)->current->getImp(2) / m_mcData->getPath(1)->current->getImp(1), 2);
           } else {
               //SLog(EInfo, "Zero importance!");

               // This case is due to not well initialized chain.
               // We arrived in this condition if the current technique (idSampling == 1), completly failed.
               // We need to found other way to handle correctly this condition.
           }
      }
    }
    // Statistic count
    if (m_idSampling == 0 && res == EAccepted) {
     m_workResult->nextContributingUniform();
    }
  }

 private:
  int m_idSampling;
  bool m_useVisibility;
};

//////////////////////////////////////////////////////////////////////////
// Parrall process
//////////////////////////////////////////////////////////////////////////
class SplattingMSPPMPhotonProcess: public ParticleProcess {
protected:
  //////////////////////////////////////////////////////////////////////////
  // Attributes
  //////////////////////////////////////////////////////////////////////////
  size_t m_photonCount;
  mutable int m_idWorker;

  const MSPPMConfiguration& m_config;
  GatherMSPPMPhotonResult m_results;
public:
  SplattingMSPPMPhotonProcess(size_t photonCount,
      size_t granularity, const MSPPMConfiguration& config,
      const void *progressReporterPayload);

  ref<WorkProcessor> createWorkProcessor() const;

  void processResult(const WorkResult *wr, bool cancelled);

  bool isLocal() const {
      return true;
  }

  Float getNormalisation(int idImportance) const {
    return m_results.getNormalisation(idImportance, m_config.numberChains);
  }

  int getNbEmittedPath(int idImportance) const {
    return (int)m_results.getNbEmittedPath(idImportance);
  }

  size_t getNbUniformEmitted(int idImportance) const {
    return m_results.getUniformEmittedPath(idImportance);
  }

	size_t getContributingUniform() const {
		return m_results.getContributingUniformPaths();
	}

  MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END

