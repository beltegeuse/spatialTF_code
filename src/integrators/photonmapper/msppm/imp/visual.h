#pragma once

#include "importanceFunc.h"
#include "../../gpaccel.h"
MTS_NAMESPACE_BEGIN

/**
 * This importance function is from the paper:
 * "Visual importance-based adaptive photon tracing"
 * Quan Zheng, Chang-Wen Zheng
 * The Visual Computer, 2015
 */

class VisualIF : public ImportanceFunction {
 public:
    VisualIF(const Properties &props)
            : ImportanceFunction(props, "visual", true /*dynamic*/, false) {
    }
    virtual ~VisualIF() {
    }

    virtual void precompute(Scene* scene, RenderQueue *queue,
                            const RenderJob *job, int sceneResID,
                            int sensorResID) {

        setToVisual();
    }

    virtual void update(size_t idCurrentPass, size_t totalEmitted) {
        setToVisual();
      }

    void setToVisual() {
        GPAccelPointKD<GatherPoint> pointKDTree(*m_gatherBlocks);

        const size_t maxEntry = 10; // How many GP is used to estimate the density of importance
        GPAccelPointKD<GatherPoint>::SearchResult *results =
                static_cast<GPAccelPointKD<GatherPoint>::SearchResult *>(alloca(
                        (maxEntry + 1)
                                * sizeof(GPAccelPointKD<GatherPoint>::SearchResult)));

        for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks->size();
                ++blockIdx) {
            GatherBlock &gatherBlock = (*m_gatherBlocks)[blockIdx];
            for (size_t i = 0; i < gatherBlock.size(); ++i) {
                PixelData<GatherPoint> &gpList = gatherBlock[i];
                typename PixelData<GatherPoint>::iterator gp = gpList.begin();

                // Collect for all GP attached to the pixel
                Float mi = 0.f;
                Float maxRadiusSqr = 0.f;
                for (; gp != gpList.end(); ++gp) {
                    if (!gp->its.isValid() || gp->depth == -1)
                        continue;
                    mi += gp->weight.getLuminance();

                    // Make a search
                    size_t nbRes = pointKDTree.nnSearch(gp->its.p, maxEntry,
                                                        results);
                    for (size_t idRes = 0; idRes < nbRes; idRes++) {
                        maxRadiusSqr = std::max(maxRadiusSqr,
                                                results[idRes].distSquared);
                        mi += pointKDTree[results[idRes].index]->data->weight
                                .getLuminance();
                    }  // Analyse all results
                }  // End all GP attached to the pixel

                if (maxRadiusSqr == 0.f) {
                    // Avoid bad TF weights
                    for (GatherPointsList::iterator it = gpList.begin();
                            it != gpList.end(); ++it) {
                        it->importance[m_idImportance] = 0.f;
                    }
                } else {
                    mi /= maxRadiusSqr;
                    for (GatherPointsList::iterator it = gpList.begin();
                            it != gpList.end(); ++it) {
                        it->importance[m_idImportance] = mi;
                    }
                }

            }

        }
    }

};

MTS_NAMESPACE_END
