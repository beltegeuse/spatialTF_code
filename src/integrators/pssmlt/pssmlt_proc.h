/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__PSSMLT_PROC_H)
#define __PSSMLT_PROC_H

// === Include MTS
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/bitmap.h>
#include "pssmlt.h"

// === Include STL
#include <fstream>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                     Stats for pLarge adaptation                      */
/* ==================================================================== */
struct PSSMLTStats {
	size_t largeNoZero;
	size_t largeAccepted;
	size_t largeTotal;
	size_t smallAccepted;
	size_t smallTotal;

	Float luminanceLargeTotal;

	void clear() {
		largeNoZero = 0;
		largeAccepted = 0;
		largeTotal = 0;
		smallAccepted = 0;
		smallTotal = 0;
		luminanceLargeTotal = 0.f;
	}

	void add(const PSSMLTStats& other) {
		largeNoZero += other.largeNoZero;
		largeAccepted += other.largeAccepted;
		largeTotal += other.largeTotal;
		smallAccepted += other.smallAccepted;
		smallTotal += other.smallTotal;
		luminanceLargeTotal += other.luminanceLargeTotal;
	}

	Float getProbLarge() const {
		if(smallTotal == 0 || largeTotal == 0) {
			return 0.5f;
		}

		Float n0 = largeNoZero / (Float) largeTotal;
		Float nl = largeAccepted / (Float) largeTotal;
		Float ns = smallAccepted / (Float) smallTotal;

		if((nl/n0) < 0.1f)
			return 0.25f;
		return std::min(std::max(ns / (2.f*(ns - nl)), 0.25f),1.f);
	}

	std::string toString() const {
		Float n0 = largeNoZero / (Float) largeTotal;
		Float nl = largeAccepted / (Float) largeTotal;
		Float ns = smallAccepted / (Float) smallTotal;

		std::stringstream ss;
		ss << "PSSMLTStats["
		   << "   n0 = " << n0 << ",\n"
		   << "   nl = " << nl << ",\n"
		   << "   ns = " << ns << ",\n"
		   << "   plarge = " << getProbLarge() << "\n"
		   << "]";
		return ss.str();
	}
};

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

class PSSMLTProcess : public ParallelProcess {
public:
	PSSMLTProcess(const RenderJob *parent, RenderQueue *queue,
		const PSSMLTConfiguration &config, const Bitmap *directImage,
		const std::vector<PathSeed> &seeds,
		ref<Timer> timer, Scene* scene, std::ofstream& timeFile, int& currentPass,
		MLTAccumulBuffer* accumBuffer);

	void develop();

	/* ParallelProcess impl. */
	void processResult(const WorkResult *wr, bool cancelled);
	ref<WorkProcessor> createWorkProcessor() const;
	void bindResource(const std::string &name, int id);
	EStatus generateWork(WorkUnit *unit, int worker);

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~PSSMLTProcess() { }

private:
	ref<const RenderJob> m_job;
	RenderQueue *m_queue;
	const PSSMLTConfiguration &m_config;
	const Bitmap *m_directImage;
	ref<Bitmap> m_developBuffer;
	ImageBlock *m_accum;
	ProgressReporter *m_progress;
	const std::vector<PathSeed> &m_seeds;
	ref<Mutex> m_resultMutex;
	ref<Film> m_film;
	int m_resultCounter, m_workCounter;
	unsigned int m_refreshTimeout;
	ref<Timer> m_timeoutTimer, m_refreshTimer;

	PSSMLTStats m_stats;

	// === Pass attributs
	ref<Timer> m_timer;
	Scene* m_scene;

    int& m_currentPass;
	std::ofstream& m_timeFile;
	MLTAccumulBuffer* m_accumBuffer;
};

MTS_NAMESPACE_END

#endif /* __PSSMLT_PROC */
