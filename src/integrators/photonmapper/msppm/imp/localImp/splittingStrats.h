#pragma once

#include "../importanceFunc.h"

MTS_NAMESPACE_BEGIN

class SplittingStrat {
 protected:
  Float m_Mi;
  int m_level;
  size_t m_nbIterationGP;
 public:
  SplittingStrat(int level)
      : m_Mi(0.f),
        m_level(level),
        m_nbIterationGP(0) {
  }

  virtual ~SplittingStrat() {
  }

  void updateMi(Float Mi) {
    m_Mi += Mi;
  }

  void increaseNbIterationGP() {
      m_nbIterationGP += 1;
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) = 0;
  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) = 0;
  void diplayInfo() {
    SLog(EInfo, "Splitting on cluster[ Level:%i, Mi:%f ] (%s)", m_level, m_Mi, state().c_str());
  }

  virtual std::string state() const {
       return "[NO INFO]";
   }
};

/**
 * Empty Splitting strategy:
 * Don't change active cluster
 * This policy need to be used only for precomputed data
 */
class SplittingStratEmpty : public SplittingStrat {
 public:
  SplittingStratEmpty() : SplittingStrat(0) {
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
     return new SplittingStratEmpty();
   }

  bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    return false; // Don't change split
  }
};

class SplittingStratConstant : public SplittingStrat {
 private:
  int m_nbChildsAllow;
 public:
  SplittingStratConstant(int nbChildsMax, int level)
      : SplittingStrat(level),
        m_nbChildsAllow(nbChildsMax) {
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
    return new SplittingStratConstant(m_nbChildsAllow / nbChilds, m_level + 1);
  }
  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    return m_nbChildsAllow > 1;
  }
};

class SplittingStratIteration : public SplittingStrat {
 private:
  Float m_nbIterationWait;  //< Threadhold to split

  // It's to have the same behavior between 2D and 3D approaches
  bool m_3D;
  int m_nbSplit;  //< Number of split in the same time
  Float m_factor;
 public:
  /**
   * factor: 1 is the default value (splitting in completly align to the iteration number).
   * value below one will split more raplidly the clusters.
   */
  SplittingStratIteration(Float iter, bool is3D, int nbSplit, int level, Float factor)
      : SplittingStrat(level),
        m_nbIterationWait(iter),
        m_3D(is3D),
        m_nbSplit(nbSplit),
        m_factor(factor) {
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
    //TODO: Assume the same amount between children
    if (m_3D) {
      if (m_nbSplit == 1) {
        // One split, change the threadhold
        return new SplittingStratIteration((m_nbIterationWait * 4 + 1)*m_factor, m_3D, 2,
                                           m_level + 1, m_factor);
      } else {
        // Two split, not change the threadhold
        return new SplittingStratIteration(m_nbIterationWait, m_3D, 1,
                                           m_level + 1, m_factor);
      }
    } else {
      // Case in 2D, just change the threadhold
      return new SplittingStratIteration((m_nbIterationWait * 4 + 1)*m_factor, m_3D, 1,
                                         m_level + 1, m_factor);
    }

  }

  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    return m_nbIterationWait <= ((Float) idIteration);
  }
};

class SplittingStratIterationValid : public SplittingStrat {
 private:
  Float m_nbIterationWait;  //< Threadhold to split

  // It's to have the same behavior between 2D and 3D approaches
  bool m_3D;
  int m_nbSplit;  //< Number of split in the same time
  Float m_factor;
 public:
  SplittingStratIterationValid(Float iter, bool is3D, int nbSplit, int level, Float factor)
      : SplittingStrat(level),
        m_nbIterationWait(iter),
        m_3D(is3D),
        m_nbSplit(nbSplit),
        m_factor(factor) {
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
    //TODO: Assume the same amount between children
    if (m_3D) {
      if (m_nbSplit == 1) {
        // One split, change the threadhold
        return new SplittingStratIterationValid((m_nbIterationWait * 4 + 1)*m_factor, m_3D, 2,
                                           m_level + 1, m_factor);
      } else {
        // Two split, not change the threadhold
        return new SplittingStratIterationValid(m_nbIterationWait, m_3D, 1,
                                           m_level + 1, m_factor);
      }
    } else {
      // Case in 2D, just change the threadhold
      return new SplittingStratIterationValid((m_nbIterationWait * 4 + 1)*m_factor, m_3D, 1,
                                         m_level + 1, m_factor);
    }

  }

  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    return m_nbIterationWait <= ((Float)m_nbIterationGP);
  }

  virtual std::string state() const {
      std::stringstream ss;
      ss << "[" << m_nbIterationWait << " <= " << m_nbIterationGP << "]";
      return ss.str();
  }
};

class SplittingStratSamples : public SplittingStrat {
 private:
  Float m_nbSamples;
 public:
  SplittingStratSamples(Float nbSamples, int level)
      : SplittingStrat(level),
        m_nbSamples(nbSamples) {
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
    //TODO: Assume the same amount between children
    return new SplittingStratSamples(m_nbSamples * 1.01f, m_level + 1);
  }
  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    return m_nbSamples <= m_Mi;
  }
};

class SplittingStratVariance : public SplittingStrat {
 private:
  Float m_threadhold;
  double m_minNbUniqueSamples;

  Float m_lastVariance;
  double m_lastUniqueSamples;
 public:
  SplittingStratVariance(Float varianceThreadhold, int level, double minNbUniqueSamples)
      : SplittingStrat(level),
        m_threadhold(varianceThreadhold),
        m_minNbUniqueSamples(minNbUniqueSamples) {
      m_lastVariance = 0.f;
      m_lastUniqueSamples = 0.f;
  }

  virtual SplittingStrat* getChildStrat(int nbChilds) {
    //TODO: Assume the same amount between children
    return new SplittingStratVariance(m_threadhold, m_level + 1, m_minNbUniqueSamples);
  }
  virtual bool needSplit(size_t idIteration, Float variance, double nbUniqueSamples) {
    m_lastUniqueSamples = nbUniqueSamples;
    m_lastVariance = variance;

    if(nbUniqueSamples < m_minNbUniqueSamples) {
      return false;
    } else {
      return m_threadhold > variance;
    }
  }

  virtual std::string state() const {
       std::stringstream ss;
       ss << "[" << m_lastVariance << ", " << m_lastUniqueSamples << "]";
       return ss.str();
  }
};


MTS_NAMESPACE_END
