#pragma once

#include "../importanceFunc.h"
#include "splittingStrats.h"
#include <mitsuba/core/fstream.h>

MTS_NAMESPACE_BEGIN

// This structure will be used to
// add multiple information inside the node
// This structure will hold information
// which are computed using multi core.

struct NodeInfo {
  int nbUniformHit;

  NodeInfo():
    nbUniformHit(0)
  {}

  void clear() {
    nbUniformHit = 0;
  }

  void add(const NodeInfo& other) {
    nbUniformHit += other.nbUniformHit;
  }
};

class Node {
 public:
  // FIXME: Change acces
  std::vector<Node*> m_child; //< To store the hieracgy

 protected:
  // Node hierachy and vital info
  bool m_isActived; //< To know if the node is active or not
  SplittingStrat* m_strat; //< To know when splitting or not


  // Statistic of the cluster
  Float m_surf;  //< The total surface of one cluster

  // The phi statistic (how many uniform hits the cluster during the iteration)
  Float m_phiOdd;
  Float m_phiEven;

  //< The kappa statistic (see the paper)
  Float m_kappaOdd;
  Float m_kappaEven;
  double m_uniqueNbSamples;

  int m_nbIterationWithGP;  //< Number of iteration with GP to better normalise

  // Debug statistics
  int m_nbGP;  //< Number of gatherpoint in the cluster

 public:
  Node(SplittingStrat* s) // , int nbCores
      : m_isActived(false),
        m_surf(0.f),
        m_phiOdd(0.f),
        m_phiEven(0.f),
        m_kappaOdd(0.f),
        m_kappaEven(0.f),
        m_nbIterationWithGP(0),
        m_nbGP(0) {
    m_strat = s;

    //m_coresInfo = new NodeInfo[nbCores];
    //m_densityUniform = 0;
    m_uniqueNbSamples = 0;
  }

  virtual ~Node() {
    delete m_strat;
    for (size_t i = 0; i < m_child.size(); i++) {
      delete m_child[i];
    }
    //delete[] m_coresInfo;
  }

  virtual void save(FileStream* f) = 0;
  void saveData(FileStream* f) {
    f->writeBool(m_isActived);
    f->writeFloat(m_surf);

    f->writeFloat(m_phiOdd);
    f->writeFloat(m_phiEven);
    f->writeFloat(m_kappaOdd);
    f->writeFloat(m_kappaEven);

    f->writeDouble(m_uniqueNbSamples);
    f->writeInt(m_nbIterationWithGP);
    f->writeInt(m_nbGP);
  }

  void loadData(FileStream* f) {
    m_isActived = f->readBool();
    m_surf = f->readFloat();

    m_phiOdd = f->readFloat();
    m_phiEven = f->readFloat();
    m_kappaOdd = f->readFloat();
    m_kappaEven = f->readFloat();

    m_uniqueNbSamples = f->readDouble();
    m_nbIterationWithGP = f->readInt();
    m_nbGP = f->readInt();
  }

  /*
   * Method to know which child we need to selected
   * for other levels (criteria)
   */
  virtual size_t nodeID(const GatherPoint& p) const = 0;

  ///////////////////////////
  // Radius methods
  ///////////////////////////

  /*
   * Reinitialize the radii to 0
   * and do it for the child
   */
  void reinitAssociation() {
    // Reset the surface area and the number of GP
    m_surf = 0.f;
    m_nbGP = 0;

    for (size_t i = 0; i < m_child.size(); i++) {
      m_child[i]->reinitAssociation();
    }
  }

  void associate(const GatherPoint& gp) {

    // === Update the radius of the cluster
    m_surf += gp.points->radius * gp.points->radius * M_PI;
    if (m_child.size() != 0) {
      size_t id = nodeID(gp);
      m_child[id]->associate(gp);
    }

    // === Update the count of GP liying in the cluster
    m_nbGP += 1;

    return;
  }

  ///////////////////////////
  // Phi methods
  ///////////////////////////

  void updatePhi(const Float Mi,  //< The Mi statistic
      const Float PhiOdd, const Float PhiEven,  //< The Phi statistic
      const GatherPoint& gp,
      Float uniqueNbSamples) {

    m_phiOdd += PhiOdd;
    m_phiEven += PhiEven;
    m_uniqueNbSamples += uniqueNbSamples;

    m_strat->updateMi(Mi);

    if (m_child.size() != 0) {
      size_t id = nodeID(gp);
      m_child[id]->updatePhi(Mi, PhiOdd, PhiEven, gp, uniqueNbSamples);
    }
    return;
  }

  ///////////////////////////
  // Kappa methods
  ///////////////////////////

  void rescaleKappa() {
    m_kappaOdd = m_kappaOdd * m_surf;
    m_kappaEven = m_kappaEven * m_surf;

    //m_densityUniform = m_densityUniform * m_surf;
    for (size_t i = 0; i < m_child.size(); i++) {
      m_child[i]->rescaleKappa();
    }
  }

  void updateKappa(int nbCores) {
    if (m_surf <= 0) {
      if (m_phiEven + m_phiOdd > 0) {
        SLog(EError,
             "Phi is not zero with current radii equal 0 (R: %f, P: %f)",
             m_surf, m_phiEven + m_phiOdd);
      }

      if (m_nbGP > 0) {
        SLog(EError, "No surface but some GP are inside the cluster: %i",
             m_nbGP);
      }

      return;
    } else {
      // === Update kappa stats
      m_kappaOdd = (m_kappaOdd + m_phiOdd) / (m_surf);
      m_kappaEven = (m_kappaEven + m_phiEven) / (m_surf);

      // reset phi accumulate stats
      m_phiOdd = 0;
      m_phiEven = 0;

      // === Update Multi core info
      /*int nbUniformHit = 0;
      for(int i = 0; i < nbCores; i++) {
        nbUniformHit += m_coresInfo[i].nbUniformHit;
        m_coresInfo[i].clear();
      }*/
      //m_densityUniform = (m_densityUniform + nbUniformHit) / m_surf;

      // Update number of time we have kappa
      m_nbIterationWithGP += 1;
      m_strat->increaseNbIterationGP();
      //SLog(EInfo, "kappa: %f, iter: %i, nbGP: %i", m_kappa, m_nbIterationWithGP, m_nbGP);

      // === Update children only when sum radii is not zero
      // because if the current node have 0 for radii,
      // it's means that all the childs have 0 radii ... etc.
      for (size_t i = 0; i < m_child.size(); i++) {
        m_child[i]->updateKappa(nbCores);
      }
    }
  }

  Float variance() {
      if(m_nbIterationWithGP > 0) {
          return std::abs(m_kappaEven - m_kappaOdd) / (0.5*(m_kappaEven + m_kappaOdd) + Epsilon);
      } else {
          return 10000.f;
      }
  }

  void printVariance(int level, int currentLevel) {
      if(level == currentLevel) {
        SLog(EInfo, "Variance: %f (%f, %f, %f)", variance(), m_kappaOdd, m_kappaEven, m_uniqueNbSamples);
      } else if(level == currentLevel + 1) {
        SLog(EInfo, "==== Level: %i (variance parent: %f)", level, variance());
      }

      if(level > currentLevel) {
          for (size_t i = 0; i < m_child.size(); i++) {
              m_child[i]->printVariance(level, currentLevel+1);
          }
      }
  }

  Float getKappa(const GatherPoint& gp) const {
    if (m_isActived) {
      Float kappa = m_kappaOdd + m_kappaEven;

      if (m_nbIterationWithGP > 0 && kappa > 0) {
        return (kappa * gp.points->radius * gp.points->radius * M_PI) / (m_nbIterationWithGP);
      } else {
        return -1.f;  // Invalid kappa value
      }
    } else {
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update importance");
        return 0.f;
      } else {
        size_t id = nodeID(gp);
        return m_child[id]->getKappa(gp);
      }
    }
  }

  /**
   * Return the max density
   * for all active cluster which have
   * more than one gatherpoint.
   */
   Float getDensityMax() {
    if (m_isActived) {
      if (m_nbGP > 0) {
        return m_kappaOdd + m_kappaEven; //< Just return the density
      } else {
        return 0.f;
      }
    } else {
      Float maxKappa = 0.f;
      // Go to see the children
      for (size_t i = 0; i < m_child.size(); i++) {
        maxKappa = std::max(maxKappa, m_child[i]->getDensityMax());
      }
      return maxKappa;
    }
  }

 Float getDensityMin() {
    if (m_isActived) {
      if (m_nbGP > 0) {
        return m_kappaOdd + m_kappaEven; //< Just return the density
      } else {
        return 0.f;
      }
    } else {
      Float minKappa = 0.f;
      // Go to see the children
      for (size_t i = 0; i < m_child.size(); i++) {
  Float childMinDensity = m_child[i]->getDensityMin();
  if(childMinDensity > 0.f) {
    minKappa = std::min(minKappa, childMinDensity);
  }
      }
      return minKappa;
    }
  }

  ///////////////////////////
  // Splitting update methods
  ///////////////////////////

  void splittingUpdate(size_t idIteration) {
    if (m_isActived) {  // this node is in the current split
      if (m_child.size() != 0) {  // it's not a leaf
        Float maxChildVariance = m_child[0]->variance();
        double minUniqueSamples = m_child[0]->m_uniqueNbSamples;
        for(size_t j = 1; j < m_child.size(); j++) {
            maxChildVariance = std::max(maxChildVariance,m_child[j]->variance());
            minUniqueSamples = std::min(minUniqueSamples, m_child[j]->m_uniqueNbSamples);
        }

        if (m_strat->needSplit(idIteration, maxChildVariance, minUniqueSamples)) {  // we need to split
          m_isActived = false;
          //SLog(EInfo, "Splitting detected");
          m_strat->diplayInfo();
          for (size_t i = 0; i < m_child.size(); i++) {
            m_child[i]->activate();
            m_child[i]->splittingUpdate(idIteration);  // In order to continue to split
          }
        }
      }
    } else {  // current node not in the current split
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update splitting method");
      } else {
        for (size_t i = 0; i < m_child.size(); i++) {
          m_child[i]->splittingUpdate(idIteration);  // In order to continue to split
        }
      }
    }
  }

  void activate() {
    m_isActived = true;
  }

  ///////////////////////////
  // Update the importance associated to GP
  ///////////////////////////

  /*
   * Count the number of cluster in the current cut
   */
  int nbNodeActivated() {
    if (m_isActived)
      return 1;
    else {
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update splitting method");
        return 0;
      } else {
        int count = 0;
        for (size_t i = 0; i < m_child.size(); i++) {
          count += m_child[i]->nbNodeActivated();
        }
        return count;
      }
    }
  }

  //////////////////////////////
  // Debug info for the clusters
  //////////////////////////////
  Spectrum debugInfo(const GatherPoint& gp) {
    if (m_isActived) {
      Spectrum spec;
      spec.fromLinearRGB(m_kappaOdd + m_kappaEven, (Float)m_nbGP, (Float)m_nbIterationWithGP);
      return spec;
    } else {
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update importance");
        return Spectrum(0.f);
      } else {
        size_t id = nodeID(gp);
        return m_child[id]->debugInfo(gp);
      }
    }
  }

  int getDepth(const GatherPoint& gp) {
    if (m_isActived) {
      return 1;
    } else {
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update importance");
        return 0;
      } else {
        size_t id = nodeID(gp);
        return 1 + m_child[id]->getDepth(gp);
      }
    }
  }

  int getClusterID(const GatherPoint& gp) {
    if (m_isActived) {
      return 0;
    } else {
      if (m_child.size() == 0) {
        // Leaf of the hierachie and doesn't hit
        // the split, this is an error
        SLog(EError,
             "No childs meets before current split in update importance");
        return 0;
      } else {
        size_t id = nodeID(gp);
        return (int)(m_child.size() * (id + m_child[id]->getDepth(gp)));
      }
    }
  }
};

MTS_NAMESPACE_END
