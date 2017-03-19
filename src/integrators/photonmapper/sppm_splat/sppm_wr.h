#pragma once

#ifndef HACHISUKA_WR_H_
#define HACHISUKA_WR_H_

#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>

MTS_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Empty Workresults
//////////////////////////////////////////////////////////////////////////
class GatherPhotonResult : public WorkResult {
 public:
  GatherPhotonResult()
      : WorkResult() {
    m_nbEmittedPath = 0;
  }

  //////////////////////////////////////////////////////////////////////////
  // Pure virtual impl methods
  //////////////////////////////////////////////////////////////////////////
  void load(Stream* stream) {
    Log(EError, "No serialization implemented ... ");
  }
  void save(Stream* stream) const {
    Log(EError, "No serialization implemented ... ");
  }
  std::string toString() const {
    return "GatherPhotonResult[NULL]";
  }

  //////////////////////////////////////////////////////////////////////////
  // Use full methods
  //////////////////////////////////////////////////////////////////////////
  /// Clear all results data
  void clear() {
    m_nbEmittedPath = 0;
  }

  void nextEmittedPath() {
    m_nbEmittedPath++;
  }

  size_t getNbEmittedPath() const {
    return m_nbEmittedPath;
  }

  MTS_DECLARE_CLASS()

 protected:
  size_t m_nbEmittedPath;
};
MTS_NAMESPACE_END

#endif /* HACHISUKA_WR_H_ */
