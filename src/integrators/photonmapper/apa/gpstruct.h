#pragma once

// MTS includes
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/octree.h>

MTS_NAMESPACE_BEGIN

/// Represents one individual PPM gather point including relevant statistics
struct GatherPoint {
  Intersection its;
  Spectrum weight;
  Spectrum flux;
  Spectrum fluxDirect;
  Spectrum emission;

  int depth;
  Point2i pos;
  Float radius;
  Float initRadius;

  // === Sync thread data
  Spectrum* tempFlux;
  int* tempM; // XXX Float ??
  int maxThread;
  // === Default constructor
  inline GatherPoint(int mT) : weight(0.f), flux(0.f), fluxDirect(0.f),
      emission(0.0f), depth(-1) {
      tempFlux = new Spectrum[mT];
      tempM = new int[mT];
      maxThread = mT;
  }

  inline ~GatherPoint() {
      delete[] tempFlux;
      delete[] tempM;
  }

  /// Reset the temp value associated to the gather point.
  inline void reset() {
      memset(tempFlux, 0, sizeof(Spectrum)*maxThread);
      memset(tempM, 0, sizeof(int)*maxThread);
  }
};

MTS_NAMESPACE_END


