#pragma once

#include <fstream>

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/render/gatherproc.h>

#include "../../splatting.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Work Result
//////////////////////////////////////////////////////////////////////////
class GatherScalarFunctionPhotonResult: public WorkResult {
public:
	GatherScalarFunctionPhotonResult(): WorkResult() {
		m_nbEmittedPath = 0;
	}

    //////////////////////////////////////////////////////////////////////////
    // Pure virtual impl methods
    //////////////////////////////////////////////////////////////////////////
    void load(Stream* stream) {
        Log(EError,"No serialization implemented ... ");
    }
    void save(Stream* stream) const {
        Log(EError,"No serialization implemented ... ");
    }
    std::string toString() const {
        return "GatherPhotonResult[NULL]";
    }

    //////////////////////////////////////////////////////////////////////////
    // Use full methods
    //////////////////////////////////////////////////////////////////////////
    /// Clear all results data
    void clear() {
        m_nbEmittedPath = 0;
    }

    void nextEmittedPath() {
    	m_nbEmittedPath++;
    }

    size_t getNbEmittedPath() const {
    	return m_nbEmittedPath;
    }

    MTS_DECLARE_CLASS()

protected:
    size_t m_nbEmittedPath;
};


//////////////////////////////////////////////////////////////////////////
// Worker
//////////////////////////////////////////////////////////////////////////

class ScalarFunctionPhotonWorker : public ParticleTracer {
public:
	ScalarFunctionPhotonWorker(size_t granularity, int maxDepth, int rrDepth, int idWorker);
	ScalarFunctionPhotonWorker(Stream *stream, InstanceManager *manager);

    ref<WorkProcessor> clone() const;

    virtual void prepare();

    void serialize(Stream *stream, InstanceManager *manager) const;

    ref<WorkResult> createWorkResult() const;

    void process(const WorkUnit *workUnit, WorkResult *workResult,
        const bool &stop);

    void handleMediumInteraction(int depth, int nullInteractions, bool delta,
        const MediumSamplingRecord &mRec, const Medium *medium,
        const Vector &wi, const Spectrum &weight) {
            // === No Volume support
            Log(EError, "No support of volume rendering");
    }


	void handleFinishParticule();
	void handleNewPath();
	void handleNewParticle(const Ray ray,const Emitter &emitter);
	void handleSurfaceInteraction(int depth_, int nullInteractions, bool delta,
		const Intersection &its, const Medium *medium,
		const Spectrum &weight, const Vector & w);

    MTS_DECLARE_CLASS()

protected:
    /// Virtual destructor
    virtual ~ScalarFunctionPhotonWorker() { }

protected:
    GatherScalarFunctionPhotonResult* m_workResult;
    GatherPointMap* m_gathermap;
    size_t m_granularity;
    int m_idWorker;

	PhotonSplattingList m_path;

	// For MIS
	Float m_lastVertexForwardInversePdfSolidAngle;
	Point m_lastVertexPos;
};

//////////////////////////////////////////////////////////////////////////
// Parrall process
//////////////////////////////////////////////////////////////////////////
class SplattingScalarFunctionPhotonProcess: public ParticleProcess {
protected:
    //////////////////////////////////////////////////////////////////////////
    // Attributes
    //////////////////////////////////////////////////////////////////////////
    size_t m_photonCount;
    int m_maxDepth;
    int m_rrDepth;

    size_t m_numEmittedPath;

    mutable int m_idWorker;

public:
    SplattingScalarFunctionPhotonProcess(size_t photonCount,
        size_t granularity, int maxDepth, int rrDepth,
        const void *progressReporterPayload);

    ref<WorkProcessor> createWorkProcessor() const;

    void processResult(const WorkResult *wr, bool cancelled);

    bool isLocal() const {
        return true;
    }

    int getNbEmittedPath() {
    	return (int)m_numEmittedPath;
    }

    MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END

