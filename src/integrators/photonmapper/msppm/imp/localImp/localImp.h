#pragma once

#include "../importanceFunc.h"
#include "nodeImpl.h"
#include <mitsuba/core/fstream.h>

MTS_NAMESPACE_BEGIN

class LocalImp : public ImportanceFunction {
 private:
  bool m_use3D;
  bool m_precomputed;
  Node* m_nodes;
  ref<Mutex> m_updateGPMutex;
  SplittingStrat* m_splittingStrat;  //< Don't delete it, it will be in the destructor of the node
 public:
  LocalImp(const Properties &props)
      : ImportanceFunction(props, "localImp", true, true) {
    m_updateGPMutex = new Mutex;

    // Version of the node representation
    m_use3D = props.getBoolean("use3D");
    Float speedFactor = props.getFloat("speedFactor", 1.f);
    m_precomputed = false;

    setIs3D(m_use3D);


    // Subdivisions strategy
    std::string nameSplittingStrat = props.getString("splittingStrat");
    if (nameSplittingStrat == "constant") {
        int projectedLevels = 256 / speedFactor;
        int levels = pow(2, ceil(log(projectedLevels)/log(2)));

        SLog(EInfo, "Use %i constant levels", levels);
        m_splittingStrat = new SplittingStratConstant(levels, 0);

    } else if (nameSplittingStrat == "iteration") {
        SLog(EError, "This options is deprecied: %s", nameSplittingStrat.c_str());
    } else if (nameSplittingStrat == "iterationvalid") {
        if (m_use3D) {
          m_splittingStrat = new SplittingStratIterationValid(1, true, 2, 0, speedFactor);
        } else {
          m_splittingStrat = new SplittingStratIterationValid(1, false, 1, 0, speedFactor);
        }
    } else if (nameSplittingStrat == "samples") {
        SLog(EError, "This options is deprecied: %s", nameSplittingStrat.c_str());
    } else if(nameSplittingStrat == "variance") {
        Float varianceThres = props.getFloat("varianceThres", 0.05);
        size_t nbSamplesThres = props.getInteger("nbSamplesThres", 10000);

        m_splittingStrat = new SplittingStratVariance(varianceThres,0,nbSamplesThres);
    } else {
      SLog(EError, "No Splitting strat given %s", nameSplittingStrat.c_str());
    }
  }

  virtual ~LocalImp() {
    delete m_nodes;
  }

  virtual void precompute(Scene* scene, RenderQueue *queue,
                          const RenderJob *job, int sceneResID,
                          int sensorResID) {
    // Nothing to do
  }

  void buildHierachy(Scene* scene, size_t nCores,
                     LocalImpTree & accelStruct) {

    if(m_precomputed) {
      // Skip building nodes
      return;
    }

    if (m_use3D) {
      // Copy the tree information
      int depthTree = 15;  // 2^12 cells expected (in reality less)
      // TODO: Improve max node copy
      std::vector<GPNodeCopy *> nodes;
      accelStruct.copyKDTree(nodes, depthTree);

      //Remark: don't delete nodes elements,
      // there will be deleted by node3D
      m_nodes = new Node3D(m_splittingStrat, nodes, 0);  // 0: root
      ((Node3D*) m_nodes)->buildHierachy(nodes, (int)nCores);
      m_nodes->activate();

      SLog(EInfo, "LocalImp3D:");
      SLog(EInfo, "  * Node copied: %i", nodes.size());
      SLog(EInfo, "  * Nb Leaf (max active clusters): %i",
           ((Node3D* ) m_nodes)->nbLeaf());
    } else {
      // For the 2D version of the clustering strategy
      // the cluster need to be initialized here
      // for the 3D version, the initialisation will be
      // on the get tree procedure.
      ref<Sensor> sensor = scene->getSensor();
      ref<Film> film = sensor->getFilm();
      Point2i maxPoint = Point2i(film->getSize().x, film->getSize().y);
      m_nodes = new Node2D(m_splittingStrat, Point2i(0, 0), maxPoint);
      ((Node2D*) m_nodes)->buildHierachy();
      m_nodes->activate();  //< Active the cut to be the top of the tree
    }
  }

  virtual void update(size_t idCurrentPass, size_t totalEmitted) {
    // === Update the splitting cut
    m_nodes->splittingUpdate(idCurrentPass);
    SLog(EInfo, "Number of activated cluster: %d", m_nodes->nbNodeActivated());

    // === Recompute the Radii
    m_nodes->reinitAssociation();
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
      GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

      for (size_t i = 0; i < gatherBlock.size(); ++i) {
        GatherPointsList &gpl = gatherBlock[i];
        for (GatherPointsList::iterator it = gpl.begin(); it != gpl.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            m_nodes->associate(*it);
          }
        }
      }
    }

    if (idCurrentPass == 0 && !m_precomputed) {
      setInvSurfaceImportanceGP(m_idImportance);
      return;
    }

    // === Assign the importance
    //  - find the max kappa
    Float maxDensity = 0.f;
    Float minDensity = 100000.f;
    Float maxSurf = 0.f;
    Float minSurf = 100000.f;
    Float minKappa = 100000.f;
    Float maxKappa = 0.f;
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
      GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

      for (size_t i = 0; i < gatherBlock.size(); ++i) {
        GatherPointsList &gpl = gatherBlock[i];
        for (GatherPointsList::iterator it = gpl.begin(); it != gpl.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            Float kappaGP = m_nodes->getKappa(*it);
            Float surfGP = (it->points->radius * it->points->radius * M_PI);
            Float densityGP = kappaGP / surfGP;

            maxKappa = std::max(maxKappa, kappaGP);
            maxDensity = std::max(maxDensity, densityGP);

            maxSurf = std::max(maxSurf, surfGP);
            minSurf = std::min(minSurf, surfGP);

            if (kappaGP > 0.f) {  // Don't take into account bad kappa statistics
              // like:
              //  - empty cluster without any contributiohn
              //  - not initialized clusters
              minKappa = std::min(minKappa, kappaGP);
              minDensity = std::min(minDensity, densityGP);
            }
          }
        }
      }
    }

    // Show some info
    SLog(EInfo, "Importance function info: ");
    SLog(EInfo, " - Kappa  : [%f, %f] (scale: %f)", minKappa, maxKappa,
         maxKappa / minKappa);
    SLog(EInfo, " - Density: [%f, %f] (scale: %f)", minDensity, maxDensity,
         maxDensity / minDensity);
    SLog(EInfo, " - Surf   : [%f, %f] (scale: %f)", minSurf, maxSurf,
         maxSurf / minSurf);

    // Simulation of importance range
    SLog(EInfo, "Importance range: ");
    SLog(EInfo, " - Surf : [%f, %f]", 1.f / (1.f + m_config.epsilonInvSurf),
         1.f / ((minSurf / maxSurf) + m_config.epsilonInvSurf));
    SLog(EInfo, " - Kappa: [%f, %f]", 1.f / (1.f + m_config.epsilonLocalImp),
         1.f / ((minKappa / maxKappa) + m_config.epsilonLocalImp));

    //  - recompute the importance associated to the new GP
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
        ++blockIdx) {
      GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

      for (size_t i = 0; i < gatherBlock.size(); ++i) {
        GatherPointsList &gpl = gatherBlock[i];
        for (GatherPointsList::iterator it = gpl.begin(); it != gpl.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            Float kappaGP = m_nodes->getKappa(*it);
            Float surfGP = (it->points->radius * it->points->radius * M_PI);
            Float densityGP = kappaGP / surfGP;

            if (kappaGP < 0.f) {
              densityGP = maxDensity;
              kappaGP = densityGP * surfGP;
            }

            it->importance[m_idImportance] = 1.f
                / ((surfGP / maxSurf) + m_config.epsilonInvSurf);
            it->importance[m_idImportance] *= 1.f
                / ((densityGP / maxDensity) + m_config.epsilonLocalImp);

            if (it->importance[m_idImportance] <= 0.f) {
              SLog(
                  EError,
                  "Bad importance function value: %f (kappaGP: %f, surfGP: %f)",
                  it->importance[m_idImportance], kappaGP, surfGP);
            }

            it->kappa = kappaGP;
            it->invSurf = surfGP;
            it->density = densityGP;
          } else {
            // Invalid GP, set statistics to 0
            it->kappa = 0;
            it->invSurf = 0;
            it->density = 0;
          }
        }
      }
    }

    if (!m_precomputed) {
      // === Rescale kappa stats
      m_nodes->rescaleKappa();
    }
  }

  /////////////////////////////////
  // Update cluster procedure
  /////////////////////////////////
  void updateCluster(const GatherPoint& gp, Float Mi, Float PhiOdd, Float PhiEven, Float nbUniqueSamples) {
    if(m_precomputed)
        return; // Don't update cluster if precomputed

    if (gp.depth != -1 && gp.its.isValid()) {
      LockGuard lock(m_updateGPMutex);
      //SLog(EInfo, "add: Mi: %f Phi: %f", Mi, Phi);
      m_nodes->updatePhi(Mi, PhiOdd, PhiEven, gp, nbUniqueSamples);  //< Update the GP statistics
    }
  }

  void printVariance() {
    for(int i = 0; i < 4; i++) {
      m_nodes->printVariance(i,0);
    }
  }

  void compactClusterStatistic(int nbCores) {
    if(m_precomputed)
      return; // Don't update clusters

    // Nothing in the classical case
    SLog(EInfo, "Update GP");
    m_nodes->updateKappa(nbCores);  //< Compact all Kappa
  }

  Spectrum getDebugInfo(const GatherPoint& gp) {
    return m_nodes->debugInfo(gp);
  }

  int getDepth(const GatherPoint& gp) {
    return m_nodes->getDepth(gp);
  }

  int getClusterID(const GatherPoint& gp) {
    return m_nodes->getClusterID(gp);
  }

  virtual bool updateOncePerPixel() const {
    return false;
  }

  Float getDensity(const GatherPoint& gp) {
      Float kappaGP = m_nodes->getKappa(gp);
      Float surfGP = (gp.points->radius*gp.points->radius*M_PI);
      Float densityGP = kappaGP / surfGP;
      return densityGP;
  }

  void saveData(const std::string& file) {
    ref<FileStream> outFile = new FileStream(file, FileStream::ETruncWrite);
    outFile->writeString("#IMP");
    if (m_use3D) {
      outFile->writeInt(3);
    } else {
      outFile->writeInt(2);
    }

    m_nodes->save(outFile.get());

  }

  void loadData(const std::string& file) {
    m_precomputed = true;
    ref<FileStream> inFile = new FileStream(file, FileStream::EReadOnly);
    std::string flag = inFile->readString();
    if(flag != "#IMP") {
      SLog(EError, "Wrong file !");
      return;
    }

    int dim = inFile->readInt();
    if((m_use3D && dim == 3) || (!m_use3D && dim == 2)) {
      if(m_use3D) {
        m_nodes = Node3D::load(inFile.get());
      } else {
        m_nodes = Node2D::load(inFile.get());
      }
    } else {
      SLog(EError, "Wrong dimension !");
    }

  }
};

MTS_NAMESPACE_END
