#pragma once

// MTS includes
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/octree.h>
#include "../pixeldata.h"

MTS_NAMESPACE_BEGIN

// Per vertex MIS info
struct VertexMISInfo {
	enum FLAGS {
		DISTANT_LIGHT = 1,
		DELTA = 2
	};
	// Vertex flags
	unsigned int m_flags;

	// Inverse of probability of sampling this vertex from the previous vertex with respect to area measure in standard bidirectional MC
	// (for gather point, previous vertex is gather point closer to the camera,
	// (for photon, previous vertex is a photon closer to a light source)
	float m_vertexInverseForwardPdf;

	// Probability of sampling this vertex from the next vertex with respect to area measure in standard bidirectional MC
	// (for gather point, next vertex is gather point farther from the camera,
	// (for photon, next vertex is a photon farther from a light source)
	float m_vertexReversePdf;

	// Same as m_vertexReversePdf but without bsdf reverse pdf
	float m_vertexReversePdfWithoutBSDF;

	// Importance that is associated with path up to this vertex. For light path is the maximum from the first
	// vertex on a light source up to this vertex, for eye path is the maximum from the last vertex (furthest from camera)
	// up to this vertex.
	float m_importance[3]; // FIMXE: 4CHAINS
};

// Stores MIS info for direct lighting at given GatherPoint
struct DirectLightMISInfo {
	// Is the last eye vertex delta?
	bool delta;

	// VertexMISInfo::m_vertexReversePdf for last eye vertex
	float m_lastVertexReversePdf;

	// VertexMISInfo::m_vertexReversePdf for last but one eye vertex
	float m_lastButOneVertexReversePdf;

	// Partial MIS weight (considering just bsdf sampling and light sampling)
	float m_lightWeight;
};

struct GatherPoint {
  // === Gather points informations
  Intersection its;
  Spectrum weight;
  Spectrum directIllumBSDF, directIllumLightSample;
  DirectLightMISInfo bsdfMIS, lightSampleMIS;

  Float importance[3]; // FIMXE: 4CHAINS
  Float kappa;
  Float invSurf;
  Float density;
  int depth;
  int sampledComponent;
  float pdfComponent;

  int tempIndex; // Index to tempPhi and tempM

  VertexMISInfo misInfo;

  // Pointer to list of gather points associated with the same pixel
  // as this gather point
  PixelData<GatherPoint> * points;
  // Index of this gather point in the points list
  int pointIndex;

  // Is this path completely delta?
  bool completelyDelta;

  // === Default constructor
  inline GatherPoint():
      weight(0.f),
      directIllumBSDF(0.0f),
      directIllumLightSample(0.0f),
      depth(-1),
      tempIndex(-1),
      points(NULL),
      pointIndex(-1),
      completelyDelta(true) {
    importance[0] = 0.f;
    importance[1] = 0.f;
    importance[2] = 0.f;

    kappa = 0;
    invSurf = 0;
    density = 0;
    sampledComponent =-1;
    pdfComponent = -1;
  }

  inline void setTempIndex(int index )
  {
	  tempIndex = index;
  }

};

typedef PixelData<GatherPoint> GatherPointsList;
typedef std::vector<GatherPointsList> GatherBlock;
typedef std::vector<GatherBlock> GatherBlocks;

//////////////////////////////////////////////////////////////////////////
// Photon Structure for splatting
//////////////////////////////////////////////////////////////////////////
struct PhotonInfo {
  Intersection its;
  Spectrum power;
  int depth;
  VertexMISInfo misInfo;
};

#define HEAVY_CHECKING 1

class PhotonSplattingList : public SerializableObject {
 private:
   std::vector<Float> m_importances;

 public:
  std::vector<PhotonInfo> photons;

  PhotonSplattingList(int nbChains):
    m_importances(nbChains, 0.f){
  }

  inline void clear() {
    photons.clear();
    for(size_t idImportance = 0; idImportance < m_importances.size(); idImportance++) {
      m_importances[idImportance] = 0.f;
    }
  }

  inline Float getImp(int idImportance) const {
    #if HEAVY_CHECKING
    if(idImportance >= (int)m_importances.size()) {
      SLog(EError, "Importance out bound");
    }
    #endif
    return m_importances[idImportance];
  }

  inline void setImp(int idImportance, Float value) {
    #if HEAVY_CHECKING
    if(idImportance >= (int)m_importances.size()) {
      SLog(EError, "Importance out bound");
    }
    #endif
    m_importances[idImportance] = value;
  }

  inline void add(const PhotonInfo& p) {
    photons.push_back(p);
  }

  inline bool isEmpty() const {
    return photons.size() == 0;
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Log(EError, "Not implemented");
  }

  MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END

