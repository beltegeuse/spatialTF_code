#pragma once

#include "importanceFunc.h"
#include "isppm/isppm.h"
#include "vsppm.h"
#include "localImp/localImp.h"
#include "invSurface.h"
#include "visual.h"
MTS_NAMESPACE_BEGIN

ImportanceFunction* getImpFunc(const Properties& props) {
  std::string nameImpFunc = props.getString("impFunc");
  if (nameImpFunc == "ISPPM") {
    return new ISPPM(props);
  } else if (nameImpFunc == "VSPPM") {
    return new VSPPM(props);
  } else if(nameImpFunc == "Visual") {
      return new VisualIF(props);
  } else if (nameImpFunc == "InvSurf") {
    return new InverseSurface(props);
  } else if (nameImpFunc == "LocalImp") {
    return new LocalImp(props);
  } else {
    SLog(EError, "No Imp. Func given %s", nameImpFunc.c_str());
    return 0;
  }
}

MTS_NAMESPACE_END
