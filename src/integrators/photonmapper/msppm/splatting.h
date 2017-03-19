#ifndef SPLATTING_H
#define SPLATTING_H

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/gatherproc.h>
#include "gpstruct.h"
#include "../gpaccel.h"
#include "../spatialTree.h"
#include "msppm.h"
#include "../misWeights.h"


MTS_NAMESPACE_BEGIN

/// Definition of the operator of the hitpoint
struct RawGatherScalarFunctionUpdateQuery {
	RawGatherScalarFunctionUpdateQuery(const Intersection& its, int depth, int idThread, const MSPPMConfiguration& c, const PhotonSplattingList & path,
		int vertexIndex, int usedTechniques, int emittedPhotons, Float normalization, int idImp)
        : its(its), depth(depth),
          idThread(idThread), config(c), path(path), vertexIndex(vertexIndex),
		  usedTechniques(usedTechniques), emittedPhotons(emittedPhotons), 
		  normalization(normalization), idImportance(idImp) {
            updateHitpointCount = 0;
    }

#if USE_NODE_FUNC
  void operator()(const GPAccel<GatherPoint>::Type::SearchResult& gp);
#else
  void operator()(const GatherPoint* gp);
#endif

  const Intersection &its;
  int depth;
  const int idThread;

  int updateHitpointCount;

  const MSPPMConfiguration& config;

  // For MIS
  const PhotonSplattingList & path;
  int vertexIndex;

  int usedTechniques;
  int emittedPhotons;

  // Importance normalization constant
  float normalization;
  int idImportance;
};

struct RawGatherImportanceQuery {
  RawGatherImportanceQuery(int nbImportance,
                           const Intersection& its,
                           int depth, int idThread,
                           const Spectrum& power, bool added, Float w,
                           Float wPath,
                           const MSPPMConfiguration& c,
                           const PhotonSplattingList & path,
                           int vertexIndex, int usedTechniques,
                           int emittedPhotons, Float normalization,
                           int idImp, bool isOddSample,
                           Float uW)
    : its(its),power(power),depth(depth),
      idThread(idThread),
      addPower(added), weight(w),
      weightPath(wPath), maxImportances(nbImportance, 0.f),
      config(c), path(path), vertexIndex(vertexIndex),
      usedTechniques(usedTechniques), emittedPhotons(emittedPhotons),
      normalization(normalization), idImportance(idImp), isOdd(isOddSample),
      uniqueWeight(uW){

    minDepthGather = 10000.0f;
  }

#if USE_NODE_FUNC
  void operator()(const GPAccel<GatherPoint>::Type::SearchResult& gp);
#else
  void operator()(const GatherPoint* gp);
#endif

  const Intersection &its;
  Spectrum power;
  int depth;
  const int idThread;

  bool addPower;
  Float weight;

  // Path values
  Float weightPath;

  // === Internal values
  Float minDepthGather;
  std::vector<Float> maxImportances; //< For the two importance function

  // The configuration !
  const MSPPMConfiguration& config;

  // For MIS
  const PhotonSplattingList & path;
  int vertexIndex;

  int usedTechniques;
  int emittedPhotons;

  // Importance normalization constant
  float normalization;
  int idImportance;
  bool isOdd;
  Float uniqueWeight;
};

struct ImportanceRes {
    std::vector<Float> importances;

    ImportanceRes(int nbImportance):
      importances(nbImportance, 0.f)
    {
    }
};

//////////////////////////////////////////////////////////////////////////
// Gather points Kd-tree
//////////////////////////////////////////////////////////////////////////
class GatherPointMap: public SerializableObject {
protected:

  Float m_maxDepth;
  GPAccel<GatherPoint>::Type m_accel;
  MSPPMConfiguration m_config;
  std::vector<Float> m_normalization;
  GatherBlocks * m_gbs;
  Scene * m_scene;
  std::vector<SerializableObject *> * m_samplers;


public:
  GatherPointMap(GatherBlocks& gps,
	    Scene * scene,
	    std::vector<SerializableObject *> * samplers,
      const MSPPMConfiguration& config,
      int nbChains):
            m_accel(gps), m_config(config),
            m_normalization(nbChains, 1.f),
            m_gbs(&gps), m_scene(scene),
            m_samplers(samplers)
  {
  }

  GatherPointMap(Stream *stream, InstanceManager *manager):
    SerializableObject(stream, manager),
		m_accel(),m_gbs(NULL),
		m_scene(NULL), m_samplers(NULL)
  {
    Log(EError, "No support of serialization ...");
  }

  virtual ~GatherPointMap() {
    Log(EDebug, "Delete the gather points Kd-tree");
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Log(EError, "No support of serialization ...");
  }

  void setNormalization(const std::vector<Float> & normalization) {
    m_normalization = normalization; // Copy values
  }

  double getNormalization(int importanceId) const {
    if(importanceId >= (int)m_normalization.size()) {
      SLog(EError, "Outbound normalisation");
    }
    return 1.0 / m_normalization[importanceId];
  }

  GPAccel<GatherPoint>::Type& getAccel() {
    return m_accel;
  }

  ImportanceRes updateGPImpacted(const PhotonSplattingList & lightPath, // The light path which needed to be splat
                                 Float powerMult, // The multiplication for the light path
                                 int idThread, // Other data
                                 Float wPath, int idImportance, Float misWeight,
                                 Float misWeightPhi,
                                 bool isUniform, bool isOdd) // If we add the contrib
  {
    ImportanceRes res(PixelData<GatherPoint>::nbChains);

    // Deduce the weight
    // This weight will be used to compute the uniform density
    // Depending the integrator configuration, some of techniques contribut or not to this statistic.
    Float weight = powerMult;
    if(m_config.phiStatisticStrategy == 2) {
        // In case of MIS, mult with MIS weight
        weight *= misWeightPhi;
    } else if(m_config.phiStatisticStrategy == 3) {
      if(!isUniform) {
        weight = 0;
      }
    }

    if(!m_config.useMISUniqueCount) {
      misWeightPhi = 1.f;
    }

		// Ignores first vertex
		for ( int i = 1; i < (int)lightPath.photons.size(); ++i )
		{
			const PhotonInfo & p = lightPath.photons[i];
			if (p.misInfo.m_flags & VertexMISInfo::DELTA)
				continue;
			
			if ( m_config.usedTechniques & MISHelper::MERGE ) {
				RawGatherImportanceQuery query(
				              PixelData<GatherPoint>::nbChains,
				              p.its,
											p.depth, idThread,
											p.power * powerMult * misWeight, true, weight,
											wPath,
											m_config, lightPath, i, m_config.usedTechniques,(int)m_config.photonCount, m_normalization[idImportance], // FIXME MARTIN (NORMALISATION FACTOR)
											idImportance, isOdd, misWeightPhi);
				m_accel.executeQuery(p.its.p, query);

				for(int idChains = 0; idChains < PixelData<GatherPoint>::nbChains; idChains++) {
          res.importances[idChains] = std::max(query.maxImportances[idChains], res.importances[idChains]);
        }
			}
		}
		if ( m_config.usedTechniques & MISHelper::CONNECT ) {
			Vector2i size = m_scene->getFilm()->getCropSize();
			int blockSize = m_scene->getBlockSize();
			Sampler * sampler = dynamic_cast<Sampler *>((*m_samplers)[idThread]);
			Point2 rnd = sampler->next2D();
			int x = (int)(rnd.x * size.x);
			int y = (int)(rnd.y * size.y);
			int blockX = x / blockSize;
			int blockY = y / blockSize;
			int blocksInRow = (size.x + blockSize - 1) / blockSize;
			int blockId = blockY * blocksInRow + blockX;
			int pixelsInRow = std::min( blockSize, size.x - blockSize * blockX);
			int listId = (y % blockSize) * pixelsInRow + x % blockSize;
			MISHelper::computeConnections((*m_gbs)[blockId][listId],lightPath,m_config.usedTechniques,(int)m_config.photonCount,idThread, (int)m_maxDepth,1.f, m_scene, idImportance);
		}
		return res;
  }

  ImportanceRes queryGPImpactedImportance(const Intersection& its,
                                  int depth, int idImportance) {
    PhotonSplattingList list(PixelData<GatherPoint>::nbChains);
    RawGatherImportanceQuery query(PixelData<GatherPoint>::nbChains,
                                   its,
                                   depth, -10000,
                                   Spectrum(0.f), false, 0.f,
                                   0.f,
                                   m_config, list, 0, m_config.usedTechniques, (int)m_config.photonCount,
                                   m_normalization[idImportance], idImportance, false, 0.f);
    m_accel.executeQuery(its.p, query);

    ImportanceRes res(PixelData<GatherPoint>::nbChains);
    for(int idChains = 0; idChains < PixelData<GatherPoint>::nbChains; idChains++) {
      res.importances[idChains] = query.maxImportances[idChains]; // Copy all
    }

    return res;
  }


  /// Use NNsearch with max radius in the Kd-tree
  bool impactHitpointsScalarFunction(const PhotonSplattingList & lightPath, int idThread, int idImportance) {
	  int totalUpdate = 0;
	  // Ignores first vertex
	  for ( int i = 1; i < (int)lightPath.photons.size(); ++i )
	  {
		const PhotonInfo & p = lightPath.photons[i];
		if (p.misInfo.m_flags & VertexMISInfo::DELTA)
			continue;
		//if (  m_config.usedTechniques & MISHelper::MERGE ) {
			RawGatherScalarFunctionUpdateQuery query(p.its, p.depth, idThread, m_config, lightPath, i, 
				m_config.usedTechniques, (int)m_config.photonCount, m_normalization[idImportance], idImportance); // FIXME MARTIN (NORMALISATION)
			m_accel.executeQuery(p.its.p, query);
			totalUpdate += query.updateHitpointCount;
		//}
	  }
	  // If more than one GP is impacted, return true
	  return totalUpdate != 0;
  }

   MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END

#endif // SPLATTING_H
