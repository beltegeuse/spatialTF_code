#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>

#include "splatting.h"
#include "msppm_sampler.h"

MTS_NAMESPACE_BEGIN

class PhotonPaths: public SerializableObject {
public:
	PhotonPaths(int nbChains) {
		current = new PhotonSplattingList(nbChains);
		proposed = new PhotonSplattingList(nbChains);
	}

	PhotonSplattingList* current, *proposed;

	bool isNotInitialised(int idImportance) {
    return current->getImp(idImportance) == 0;
	}

	void swap() {
		std::swap(current, proposed);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Log(EError, "Not implemented");
	}

	~PhotonPaths()
	{
		delete current;
		delete proposed;
		current = proposed = NULL;
	}
	MTS_DECLARE_CLASS()
};

/**
 * This struct will be responsible to store the
 * current data for one Markov Chain
 * This is include:
 *  - Proposed and current light path
 *  - Sampler used ... etc.
 */
class MCData: public SerializableObject {
 private:
    std::vector<PhotonPaths*> m_paths;
    std::vector<MSPPMSampler*> m_samplers;
    std::vector<Float> m_weights; //< Current weight of the current state
 public:
    MCData(std::vector<MSPPMSampler*> samplers):
      m_samplers(samplers),
      m_weights(samplers.size(), 0.f)
   {
      m_paths.reserve(samplers.size());
      for(size_t i = 0; i < samplers.size(); i++) {
        m_paths.push_back(NULL);
      }
    }

    void reinitAMCMCSamplers() {
      for(size_t i = 0; i < m_samplers.size(); i++) {
          SLog(EInfo, "Reinit AMCMC %i: %f", (int)i, m_samplers[i]->getAccRate());
          m_samplers[i]->reinitAMCMC();
       }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Log(EError, "Not implemented");
    }

    MSPPMSampler* getSampler(int idImportance) {
      if(idImportance >= (int)m_samplers.size()) {
        SLog(EError, "Out of bound getSampler");
      }
      return m_samplers[idImportance];
    }

    PhotonPaths* getPath(int idImportance) {
      if(idImportance >= (int)m_samplers.size()) {
        SLog(EError, "Out of bound getPath");
      }
      return m_paths[idImportance];
    }

    void setPath(int idImportance, PhotonPaths* path) {
      if(idImportance >= (int)m_samplers.size()) {
        SLog(EError, "Out of bound getPath");
      }
      if(m_paths[idImportance] != NULL) {
        SLog(EError, "Memory leaks!");
      }
      m_paths[idImportance] = path;
    }

    bool checkPaths() {
      for(size_t i = 0; i < m_paths.size(); i++) {
        if (m_paths[i]->isNotInitialised(i)) {
          SLog(EWarn, "Not initialised Path for Level %i !!!", i);
          return false;
        }
      }
      return true;
    }

    void setWeightZero() {
      for(size_t i = 0; i < m_weights.size(); i++) {
        m_weights[i] = 0;
      }
    }

    Float getWeight(int idImportance) {
      return m_weights[idImportance];
    }

    void setWeight(int idImportance, Float v) {
      m_weights[idImportance] = v;
    }
    void increaseWeight(int idImportance, Float v) {
      m_weights[idImportance] += v;
    }

    void swapChains(int id1, int id2) {
     // Do the replica exchange
     // - For paths
     std::swap(getPath(id1)->current, getPath(id2)->current);
     // - And ramdom numbers
     std::swap(m_samplers[id1], m_samplers[id2]);
     // Undo swap for the AMCMC
     std::swap(getSampler(id1)->amcmc, getSampler(id2)->amcmc);
    }

    // This function check the current importance of the stored path
    // And update it
    void checkImportance(GatherPointMap* gathermap) {
      int nbChains = m_weights.size();
      // For each chain, we compute all the values
      for(int idImportance = 0; idImportance < nbChains; idImportance++){
        // Compute the importance
        // of the current markov chain
        std::vector<Float> currentImportance(nbChains, 0.f);

        for (size_t p = 0; p < m_paths[idImportance]->current->photons.size(); ++p) {
          ImportanceRes impPhoton = gathermap->queryGPImpactedImportance(
                  m_paths[idImportance]->current->photons[p].its,
                  m_paths[idImportance]->current->photons[p].depth,
                  idImportance);
          for(int idChain = 0; idChain < nbChains; idChain++) {
            currentImportance[idChain] = std::max(impPhoton.importances[idChain], currentImportance[idChain]);
          }
        }

        // If the importance is not same, just scale it !
        if (currentImportance[idImportance] != m_paths[idImportance]->current->getImp(idImportance)) {
          // FIXME: Show other importance
          Log(EInfo, "Found different importance: %f (old: %f, nbPhoton: %i)",
                  currentImportance[idImportance], m_paths[idImportance]->current->getImp(idImportance),
                  m_paths[idImportance]->current->photons.size());

          for(int idChain = 0; idChain < nbChains; idChain++) {
            m_paths[idImportance]->current->setImp(idChain,currentImportance[idChain]);
          }

          // No importance means zero contribution
          // of the current markov state.
          // Need to found one before the rendering.
          if (currentImportance[idImportance] == 0) {
            Log(EWarn, "Empty old path detected !: %i", idImportance);
          }
        }
      }
    }

    MTS_DECLARE_CLASS()
};

struct SeedNormalisationPhoton {
	Float importance;
	size_t sampleIndex;
};

class PhotonPathBuilder: public Object {
public:
	PhotonPathBuilder(Scene *scene, int maxDepth, int rrDepth,
			Sampler* sampler, int idWorker,
			GatherPointMap* gatherMap);
	Float generateSeeds(size_t sampleCount, size_t seedCount,
			std::vector<SeedNormalisationPhoton> &seeds, int idImportance);
	void samplePaths(PhotonSplattingList& list, int idImportance);
	void setSampler(Sampler* sampler) {
		m_sampler = sampler;
	}
	MTS_DECLARE_CLASS()

private:
	ref<Scene> m_scene;
	int m_maxDepth;
	int m_rrDepth;
	Sampler* m_sampler;
	int m_idWorker;
	GatherPointMap* m_gatherMap;
};

MTS_NAMESPACE_END
