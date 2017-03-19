#pragma once

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

#define IMP_MULT 0.1f

enum ENumberStrategy {
  ENbUnknow = -1, //< Handle unknow strategy
  ENbNormal = 0, //< Mi += 1 when there is a contribution
  ENbDifferentContrib = 1, //< Mi += 1 only when the markov chain state change
  ENbMetropolis = 2 //< Mi += 1/p_i*(S) like radiance computation
};

enum ETreeType {
	ETTMedian = 0,
	ETTSAH = 1
};
inline ENumberStrategy getNumberStrategy(const std::string& s) {
  if(s == "Normal") {
    return ENbNormal;
  } else if(s == "Different") {
    return ENbDifferentContrib;
  } else if(s == "Metro") {
    return ENbMetropolis;
  } else {
    SLog(EError, "No found number strategy: %s", s.c_str());
    return ENbUnknow;
  }
}

inline std::string getNumberStrategyName(ENumberStrategy s) {
  if(s == ENbUnknow) {
    return "unknow";
  } else if(s == ENbNormal) {
    return "normal";
  } else if(s == ENbDifferentContrib) {
    return "difference";
  } else if(s == ENbMetropolis) {
      return "metropolis";
  } else {
    SLog(EError, "No found compatible string for the strategy");
    return "";
  }
}

inline ETreeType getTreeType(const std::string& s) {
	if(s == "median") 
		return ETTMedian;
	else if (s == "sah")
		return ETTSAH;
	else
		SLog(EError, "Unknown tree type: %s", s.c_str());
	return ETTMedian;
}

inline std::string getTreeTypeName(ETreeType s) {
	if(s == ETTMedian) 
		return "median";
	else if (s == ETTSAH)
		return "sah";
	else
		SLog(EError, "Unknown tree type");
	return "unknown";
}

struct MSPPMConfiguration {
	// Photon trace related
	int maxDepth; //< Max path length
	int rrDepth; //< Russian roulette
	size_t photonCount; //< Number of photon shooted per passes
	int cancelPhotons;

	// PPM related
	Float alpha; //< Alpha for reduce rate the scale on gather point
	Float initialScale; //< Initial scale to determine the intial radius of the gather points

	// Algorithm parameters
	bool useExpectedValue; //< Waste reclycling as Veach introduced
	//bool useJaroMIS; //< Use MIS as Jaroslav propose
	ENumberStrategy numberStrat; //< Strategy to compute the MI statistic (see above)
	// By default it's ENbDifferentContrib which is used
	Float AStar;
	Float epsilonInvSurf;
	Float epsilonLocalImp;

	// Chains parameters
	int numberChains;
	bool useMISLevel;
	bool REAllTime;
	bool showUpperLevels;
	bool correctMIS;
	bool strongNormalisation;
	bool usePowerHeuristic;
	bool useMISUniqueCount;
	bool rescaleLastNorm;

	// The different way to compute Phi:
	// 0: Use first level
	// 1: Use second level
	// 2: Use combinaison of the two levels
	// 3: Use only uniform
	int phiStatisticStrategy;

	// Use reproductible strategy or not
	// depending of computation is an reference
	// or not
	bool referenceMod;

	// Which technique should be used: BPT, VCM, PATH-TRACER ETC..
	int usedTechniques;

	// Use multi-gather points (true for VCM, BPM..)
	bool multiGatherPoints;
	bool removeDeltaPaths;

	// Tree type for local importance 3D
	ETreeType treeType;

	// Dump function to know
	// the current configuration
	void dump() const {
		SLog(EInfo, "MSPPM configuration:");

		// Print informations about parameters related to Photon tracing
		SLog(EInfo, " --- Photon tracing:");
		SLog(EInfo, "   Maximum path depth          : %i", maxDepth);
		SLog(EInfo, "   Russian roulette depth      : %i", rrDepth);

		// Print informations about parameters related to SPPM
		SLog(EInfo, " --- Progressive:");
		SLog(EInfo, "   Alpha SPPM                  : %f", alpha);
		SLog(EInfo, "   Initial scale radius        : %f", initialScale);

		// Print informations about parameters related to our algorithm
		SLog(EInfo, " --- Metropolis:");
		SLog(EInfo, "   Use expected value          : %s", useExpectedValue ? "yes" : "no");
		//SLog(EInfo, "   Use MIS to merge diff. iter : %s", useJaroMIS ? "yes" : "no");
		SLog(EInfo, "   Mi computation strategy     : %s", getNumberStrategyName(numberStrat).c_str());
		SLog(EInfo, "   Is in reference mode        : %s", referenceMod ? "yes" : "no");
		SLog(EInfo, "   Acceptance rate targeted    : %f", AStar);
		SLog(EInfo, "   Epsilon value for InvSurf   : %f", epsilonInvSurf);
		SLog(EInfo, "   Epsilon value for LocalImp  : %f", epsilonLocalImp);
		SLog(EInfo, "   LocalImp3D tree type        : %s", getTreeTypeName(treeType).c_str());

		SLog(EInfo, " --- N Chain configuration:");
		SLog(EInfo, "   Number of chains used       : %i", numberChains);
		SLog(EInfo, "   Use MIS Level               : %s", useMISLevel ? "yes" : "no");

		SLog(EInfo, " --- Phi:");
		SLog(EInfo, "   Phi estimation strategy: %i", phiStatisticStrategy);

		SLog(EInfo, " --- Debug:");
		SLog(EInfo, "   RE all time                 : %s", REAllTime ? "yes" : "no");
		SLog(EInfo, "   Show only highest contrib   : %s", showUpperLevels ? "yes" : "no");
		SLog(EInfo, "   Have good MIS weights       : %s", correctMIS ? "yes" : "no");
		SLog(EInfo, "   Use regular normalisation   : %s", strongNormalisation ? "yes" : "no");
		SLog(EInfo, "   Use power heuristic         : %s", usePowerHeuristic ? "yes" : "no");
		SLog(EInfo, "   Use MIS unique count        : %s", useMISUniqueCount ? "yes" : "no");
	}

};

MTS_NAMESPACE_END
