#pragma once

#include "importanceFunc.h"

MTS_NAMESPACE_BEGIN

class VSPPM : public ImportanceFunction {
 public:
  VSPPM(const Properties &props): ImportanceFunction(props, "vsppm", false, false) {
  }
  virtual ~VSPPM() {
  }

  virtual void precompute(Scene* scene, RenderQueue *queue,
       const RenderJob *job, int sceneResID, int sensorResID) {
    setConstantImportanceGP(m_idImportance);
  }

  virtual void update(size_t idCurrentPass, size_t totalEmitted) {
	  setConstantImportanceGP(m_idImportance);
	  return;
  }
};

MTS_NAMESPACE_END
