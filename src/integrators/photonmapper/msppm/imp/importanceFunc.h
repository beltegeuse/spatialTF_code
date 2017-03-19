#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>

#include "../msppm.h"
#include "../gpstruct.h"
#include "../../initializeRadius.h"

MTS_NAMESPACE_BEGIN

class ImportanceFunction: public SerializableObject {
 public:
  ImportanceFunction(const Properties &props, const std::string& name, bool dynamic, bool phiStatistic) {
    m_running = false;
    m_dynamic = dynamic;
    m_needPhiStatistic = phiStatistic;
    m_name = name;
    m_idImportance = 0;
    m_is3D = false;
  }

  virtual ~ImportanceFunction() {
  }

  /*
   * Initialize the importance function
   * to be able to query more easly all the ressources
   * (gatherpoints, config, gatherpoint generator, ... etc.)
   */
  void initializeObject(MSPPMConfiguration& config, RadiusInitializer* gp,
                        GatherBlocks* gps,
                        std::vector<Point2i>* offset) {
    m_config = config;
    m_gpManager = gp;
    m_gatherBlocks = gps;
    m_offset = offset;
  }

  /*
   * Methods which needs to be redefined in
   * the other importance function implementation
   */

  // Precompute the importance function
  virtual void precompute(Scene* scene, RenderQueue *queue,
                          const RenderJob *job, int sceneResID,
                          int sensorResID) = 0;

  // Update the importance function (with the new GP)
  // and update the associated importance of the gatherpoints
  virtual void update(size_t idCurrentPass, size_t totalEmitted) = 0;

  /*
   * methods to know the importance function caracteristics
   */
  bool isDynamic() const {
    return m_dynamic;
  }
  bool needPhiStatistic() const {
    return m_needPhiStatistic;
  }
  void setRunning(bool run) {
    m_running = run;
  }
  const std::string& getName() const {
     return m_name;
  }
  const bool is3D() const {
     return m_is3D;
  }

  virtual bool updateOncePerPixel() const {
    return true;
  }

  // Mitsuba relative fonctions
  void serialize(Stream *stream, InstanceManager *manager) const {
    Log(EError, "Not implemented");
  }

  void setImportanceID(int v) {
      m_idImportance = v;
  }

  void setIs3D(bool v) {
      m_is3D = v;
  }

  MTS_DECLARE_CLASS()

  void setConstantImportanceGP(int importanceID, Float value = 1.f) {
      // Set constant importance
      // over the gather points
      for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
          ++blockIdx) {
        GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

        for (size_t i = 0; i < gatherBlock.size(); ++i) {
          GatherPointsList &gpl = gatherBlock[i];
          for ( GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it ) {
            it->importance[importanceID] = 1.f;
          }
        }
      }
    }

    /**
     * Set inverse surface over gp on importance ID channel
     * \return the scale of the importance function
     */
    Float setInvSurfaceImportanceGP(int importanceID) {
      // --- Get max radius
      Float maxSurf = 0.f;
      Float minSurf = 100000.f;
      for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
              ++blockIdx) {
        GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

        for (size_t i = 0; i < gatherBlock.size(); ++i) {
          GatherPointsList &gpl = gatherBlock[i];
          for ( GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it ) {
            if (it->depth != -1 && it->its.isValid()) {
              Float surf = gpl.radius*gpl.radius*M_PI;
              maxSurf = std::max(maxSurf, surf);
              minSurf = std::min(minSurf, surf);
            }
          }
        }
      }
      SLog(EInfo, "Surface range found: [%f, %f]", minSurf, maxSurf);

      for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
          ++blockIdx) {
        GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];

        for (size_t i = 0; i < gatherBlock.size(); ++i) {
          GatherPointsList &gpl = gatherBlock[i];
          for ( GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it ) {
            it->importance[importanceID] = 1.f / (((gpl.radius*gpl.radius*M_PI) / maxSurf) + m_config.epsilonInvSurf);
          }
        }
      }
      return maxSurf;
    }

 protected:
  bool m_running;
  bool m_dynamic;
  bool m_needPhiStatistic;
  int m_idImportance;
  std::string m_name;
  bool m_is3D;

  MSPPMConfiguration m_config;
  RadiusInitializer* m_gpManager;
  GatherBlocks* m_gatherBlocks;
  std::vector<Point2i>* m_offset;
};

MTS_NAMESPACE_END
