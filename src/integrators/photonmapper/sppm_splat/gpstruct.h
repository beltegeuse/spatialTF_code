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

	// Importance associated with this vertex - should be always one for sppm
	float m_importance[2];
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
  // Position informations
  Intersection its;  //< Intersection information
  int depth;

  // SPPM Statistics
  Spectrum weight;  //< Importance associate
  Spectrum directIllumBSDF, directIllumLightSample;
  DirectLightMISInfo bsdfMIS, lightSampleMIS;
  int sampledComponent;
  float pdfComponent;

  VertexMISInfo misInfo;

  // Pointer to list of gather points associated with the same pixel
  // as this gather point
  PixelData<GatherPoint> * points;
  // Index of this gather point in the points list
  int pointIndex;

  // Is this path completely delta?
  bool completelyDelta;


  // Default constructor
  inline GatherPoint():
	  depth(-1),
	  weight(0.f),
	  directIllumBSDF(0.0f),
	  directIllumLightSample(0.0f),
	  sampledComponent(-1),
	  points(NULL),
	  pointIndex(-1),
	  completelyDelta(true)
	{
	}

  inline void setTempIndex(int index )
  {
	  // NOTHING TO DO HERE
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

class PhotonSplattingList {
public:
	std::vector<PhotonInfo> photons;
	Float importance[2]; // FIXME: 4CHAINS

	PhotonSplattingList() {
	    importance[0] = 0.f;
	    importance[1] = 0.f;
	}

	inline void clear() {
		photons.clear();
		importance[0] = 0.f;
        importance[1] = 0.f;
	}

	inline void add(const PhotonInfo& p) {
		photons.push_back(p);
	}

	inline bool isEmpty() const {
		return photons.size() == 0;
	}

	inline Float getImp(int idImportance) const {
    // FIXME: Make more robust code
    return importance[idImportance];
  }
};

MTS_NAMESPACE_END

