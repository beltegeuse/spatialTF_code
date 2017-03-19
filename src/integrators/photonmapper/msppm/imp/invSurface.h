#pragma once

#include "importanceFunc.h"

MTS_NAMESPACE_BEGIN

class InverseSurface : public ImportanceFunction {
 public:
  InverseSurface(const Properties &props)
      : ImportanceFunction(props, "invSurf", true, false) {
  }
  virtual ~InverseSurface() {
  }

  virtual void precompute(Scene* scene, RenderQueue *queue,
                          const RenderJob *job, int sceneResID,
                          int sensorResID) {
  }

  virtual void update(size_t idCurrentPass, size_t totalEmitted) {
    setInvSurfaceImportanceGP(m_idImportance);
  }

  virtual bool updateOncePerPixel() const {
    return false;
  }
};

MTS_NAMESPACE_END

