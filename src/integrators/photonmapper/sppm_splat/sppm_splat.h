#ifndef HACHISUKA_H_
#define HACHISUKA_H_

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

struct SPPMSplatConfig {
  Float initialScale;
  Float alpha;

  int maxDepth;
  int minDepth;
  int rrDepth;

  int usedTechniques;

  bool multiGatherPoints;

  bool removeDeltaPaths;

  inline SPPMSplatConfig() {
  }

  void dump() const {
    SLog(EInfo, "SPPM configuration:");
    SLog(EInfo, "   Maximum path depth          : %i", maxDepth);
    SLog(EInfo, "   Minimum path depth          : %i", minDepth);
    SLog(EInfo, "   Russian roulette depth      : %i", rrDepth);
    SLog(EInfo, "   Alpha SPPM                  : %f", alpha);
    SLog(EInfo, "   Initial scale radius        : %f", initialScale);

  }
};

MTS_NAMESPACE_END

#endif /* HACHISUKA_H_ */
