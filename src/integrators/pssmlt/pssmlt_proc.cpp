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

#include <mitsuba/bidir/util.h>
#include <mitsuba/bidir/path.h>
#include "pssmlt_proc.h"
#include "pssmlt_sampler.h"

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

MTS_NAMESPACE_BEGIN

// TODO: Faire heritage
class SeedWorkUnitPSSMLT : public SeedWorkUnit {
public:
	inline void set(const WorkUnit *wu) {
		SeedWorkUnit::set(wu);
		m_pLarge = static_cast<const SeedWorkUnitPSSMLT *>(wu)->m_pLarge;
	}

	inline void setProbLarge(Float v) {
		m_pLarge = v;
	}

	inline Float getProbLarge() const {
		return m_pLarge;
	}

	inline void load(Stream *stream) {
		SeedWorkUnit::load(stream);
		m_pLarge = stream->readFloat();
	}

	inline void save(Stream *stream) const {
		SeedWorkUnit::save(stream);
		stream->writeFloat(m_pLarge);
	}

	inline std::string toString() const {
		return "SeedWorkUnitPSSMLT[]";
	}

	MTS_DECLARE_CLASS()
private:
	Float m_pLarge; // For a given pLarge stats
};

class ResultsPSSMLT : public ImageBlock {
public:
	ResultsPSSMLT(Bitmap::EPixelFormat fmt, const Vector2i &size,
			const ReconstructionFilter *filter = NULL, int channels = -1):
				ImageBlock(fmt, size, filter, channels)
	{
		m_stats.clear();
	}

	virtual ~ResultsPSSMLT() {
	}

	/////////////////////////////////////
	// Driven the pLarge values
	/////////////////////////////////////
	PSSMLTStats& getStats() {
		return m_stats;
	}

	const PSSMLTStats& getStats() const {
		return m_stats;
	}

	MTS_DECLARE_CLASS()

private:
	PSSMLTStats m_stats;
};

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

StatsCounter largeStepRatio("Primary sample space MLT",
	"Accepted large steps", EPercentage);
StatsCounter largeStepNonZero("Primary sample space MLT",
	"large steps non zero contribution", EPercentage);
StatsCounter smallStepRatio("Primary sample space MLT",
	"Accepted small steps", EPercentage);
StatsCounter acceptanceRate("Primary sample space MLT",
	"Overall acceptance rate", EPercentage);
StatsCounter forcedAcceptance("Primary sample space MLT",
	"Number of forced acceptances");

class PSSMLTRenderer : public WorkProcessor {
public:
	PSSMLTRenderer(const PSSMLTConfiguration &conf)
		: m_config(conf) {
	}

	PSSMLTRenderer(Stream *stream, InstanceManager *manager)
		: WorkProcessor(stream, manager) {
		m_config = PSSMLTConfiguration(stream);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		m_config.serialize(stream);
	}

	ref<WorkUnit> createWorkUnit() const {
		return new SeedWorkUnitPSSMLT();
	}

	ref<WorkResult> createWorkResult() const {
		return new ResultsPSSMLT(Bitmap::ESpectrum,
			m_film->getCropSize(), m_film->getReconstructionFilter());
	}

	void prepare() {
		// === Basic initialisation
		Scene *scene = static_cast<Scene *>(getResource("scene"));
		m_origSampler = static_cast<PSSMLTSampler *>(getResource("sampler"));
		m_sensor = static_cast<Sensor *>(getResource("sensor"));
		m_scene = new Scene(scene);
		m_film = m_sensor->getFilm();
		m_scene->setSensor(m_sensor);
		m_scene->setSampler(m_origSampler);
		m_scene->removeSensor(scene->getSensor());
		m_scene->addSensor(m_sensor);
		m_scene->setSensor(m_sensor);
		m_scene->wakeup(NULL, m_resources);
		m_scene->initializeBidirectional();

		// === Create several sampler (Why ???)
		m_rplSampler = static_cast<ReplayableSampler*>(
			static_cast<Sampler *>(getResource("rplSampler"))->clone().get());
		m_sensorSampler = new PSSMLTSampler(m_origSampler);
		m_emitterSampler = new PSSMLTSampler(m_origSampler);
		m_directSampler = new PSSMLTSampler(m_origSampler);

		//TODO: Voir le path builder
		m_pathSampler = new PathSampler(m_config.technique, m_scene,
			m_emitterSampler, m_sensorSampler, m_directSampler, m_config.maxDepth,
			m_config.rrDepth, m_config.separateDirect, m_config.directSampling);
	}

	void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
		ResultsPSSMLT *result = static_cast<ResultsPSSMLT *>(workResult);
		const SeedWorkUnitPSSMLT *wu = static_cast<const SeedWorkUnitPSSMLT *>(workUnit);
		const PathSeed &seed = wu->getSeed();
		SplatList *current = new SplatList(), *proposed = new SplatList();

		m_emitterSampler->reset();
		m_sensorSampler->reset();
		m_directSampler->reset();

		// === Initialise the random number (Random*)
		m_sensorSampler->setRandom(m_rplSampler->getRandom());
		m_emitterSampler->setRandom(m_rplSampler->getRandom());
		m_directSampler->setRandom(m_rplSampler->getRandom());

		/* Generate the initial sample by replaying the seeding random
		   number stream at the appropriate position. Afterwards, revert
		   back to this worker's own source of random numbers */
		m_rplSampler->setSampleIndex(seed.sampleIndex);

		m_pathSampler->sampleSplats(Point2i(-1), *current);

		// === Clear results
		//TODO: Surcharge op
		result->clear();
		result->getStats().clear();

		ref<Random> random = m_origSampler->getRandom();
		m_sensorSampler->setRandom(random);
		m_emitterSampler->setRandom(random);
		m_directSampler->setRandom(random);

		// === Jump to frist unused random number
		m_rplSampler->updateSampleIndex(m_rplSampler->getSampleIndex()
			+ m_sensorSampler->getSampleIndex()
			+ m_emitterSampler->getSampleIndex()
			+ m_directSampler->getSampleIndex());

		// === Reset samplers
		m_sensorSampler->accept();
		m_emitterSampler->accept();
		m_directSampler->accept();

		/* Sanity check -- the luminance should match the one from
		   the warmup phase - an error here would indicate inconsistencies
		   regarding the use of random numbers during sample generation */
		if (std::abs((current->luminance - seed.luminance)
				/ seed.luminance) > Epsilon)
			Log(EError, "Error when reconstructing a seed path: luminance "
				"= %f, but expected luminance = %f", current->luminance, seed.luminance);

		ref<Timer> timer = new Timer();

		/* MLT main loop */
		Float cumulativeWeight = 0;
		current->normalize(m_config.importanceMap);
		for (uint64_t mutationCtr=0; mutationCtr<m_config.nMutations && !stop; ++mutationCtr) {
			// === For time constant comparison
			if (wu->getTimeout() > 0 && (mutationCtr % 8192) == 0
					&& (int) timer->getMilliseconds() > wu->getTimeout())
				break;

			// === pLarge state ?
			bool largeStep = random->nextFloat() < wu->getProbLarge();
			// Propagate the decision to all samplers
			m_sensorSampler->setLargeStep(largeStep);
			m_emitterSampler->setLargeStep(largeStep);
			m_directSampler->setLargeStep(largeStep);

			// === Create proposed path
			m_pathSampler->sampleSplats(Point2i(-1), *proposed);
			proposed->normalize(m_config.importanceMap);

			if(largeStep) {
				result->getStats().largeTotal++;
				if(proposed->luminance != 0) {
					result->getStats().largeNoZero++;
					result->getStats().luminanceLargeTotal += proposed->luminance ;
				}
			} else {
				result->getStats().smallTotal++;
			}

			// === M-H Acceptance ratio
			// In this case the mutation strategy is symetric so leads to some simplifications
			Float a = std::min((Float) 1.0f, proposed->luminance / current->luminance);

			// === Check proposed luminance for consistancy
			if (std::isnan(proposed->luminance) || proposed->luminance < 0) {
				Log(EWarn, "Encountered a sample with luminance = %f, ignoring!",
						proposed->luminance);
				a = 0;
			}

			bool accept;
			Float currentWeight, proposedWeight;

			if (a > 0) {
				// Le truc avec MIS sur le pLarge
				if (m_config.kelemenStyleWeights && !m_config.importanceMap) {
					/* Kelemen-style MLT weights (these don't work for 2-stage MLT) */
					currentWeight = (1 - a) * current->luminance
						/ (current->luminance/m_config.luminance + wu->getProbLarge());
					proposedWeight = (a + (largeStep ? 1 : 0)) * proposed->luminance
						/ (proposed->luminance/m_config.luminance + wu->getProbLarge());
				} else {
					/* Veach-style use of expectations */
					currentWeight = 1-a;
					proposedWeight = a;
				}
				accept = (a == 1) || (random->nextFloat() < a);
			} else {
				if (m_config.kelemenStyleWeights && !m_config.importanceMap)
					currentWeight = current->luminance
						/ (current->luminance/m_config.luminance + wu->getProbLarge());
				else
					currentWeight = 1;
				proposedWeight = 0; // Normal
				accept = false;
			}

			// Update the weight until is accepted
			cumulativeWeight += currentWeight;
			if (accept) {
				// === For all splat, splat it one the screen
				for (size_t k=0; k<current->size(); ++k) {
					Spectrum value = current->getValue(k) * cumulativeWeight;
					if (!value.isZero())
						result->put(current->getPosition(k), &value[0]);
				}

				// === Update current state
				cumulativeWeight = proposedWeight;
				std::swap(proposed, current);

				// === Update samplers
				m_sensorSampler->accept();
				m_emitterSampler->accept();
				m_directSampler->accept();

				// === Update statistics
				if (largeStep) {
					largeStepRatio.incrementBase(1);
					++largeStepRatio;
					result->getStats().largeAccepted++;
				} else {
					smallStepRatio.incrementBase(1);
					++smallStepRatio;
					result->getStats().smallAccepted++;
				}

				acceptanceRate.incrementBase(1);
				++acceptanceRate;
			} else {
				// === For all splat, splat it one the screen
				// (Only for the rejected path)
				for (size_t k=0; k<proposed->size(); ++k) {
					Spectrum value = proposed->getValue(k) * proposedWeight;
					if (!value.isZero())
						result->put(proposed->getPosition(k), &value[0]);
				}

				// === Update sampler
				m_sensorSampler->reject();
				m_emitterSampler->reject();
				m_directSampler->reject();

				// === Update statistics
				acceptanceRate.incrementBase(1);
				if (largeStep)
					largeStepRatio.incrementBase(1);
				else
					smallStepRatio.incrementBase(1);
			}
		} // Nouvelle mutation

		// === Pour ne pas oublier la derniere splat
		/* Perform the last splat */
		for (size_t k=0; k<current->size(); ++k) {
			Spectrum value = current->getValue(k) * cumulativeWeight;
			if (!value.isZero())
				result->put(current->getPosition(k), &value[0]);
		}


		delete current;
		delete proposed;
	}

	ref<WorkProcessor> clone() const {
		return new PSSMLTRenderer(m_config);
	}

	MTS_DECLARE_CLASS()
private:
	PSSMLTConfiguration m_config;
	ref<Scene> m_scene;
	ref<Sensor> m_sensor;
	ref<Film> m_film;
	ref<PathSampler> m_pathSampler;
	ref<PSSMLTSampler> m_origSampler;
	ref<PSSMLTSampler> m_sensorSampler;
	ref<PSSMLTSampler> m_emitterSampler;
	ref<PSSMLTSampler> m_directSampler;
	ref<ReplayableSampler> m_rplSampler;
};

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

PSSMLTProcess::PSSMLTProcess(const RenderJob *parent, RenderQueue *queue,
	const PSSMLTConfiguration &conf, const Bitmap *directImage,
	const std::vector<PathSeed> &seeds, ref<Timer> timer, Scene* scene,
	std::ofstream& timeFile, int& currentPass,
	MLTAccumulBuffer* accumBuffer) : m_job(parent), m_queue(queue),
		m_config(conf), m_progress(NULL), m_seeds(seeds), m_scene(scene),
		m_currentPass(currentPass), m_timeFile(timeFile), m_accumBuffer(accumBuffer)
{
	m_directImage = directImage;
	m_timeoutTimer = new Timer();
	m_refreshTimer = new Timer();
	m_resultMutex = new Mutex();
	m_resultCounter = 0;
	m_workCounter = 0;
	m_refreshTimeout = 1;
	m_stats.clear();

	m_timer = timer;
}

ref<WorkProcessor> PSSMLTProcess::createWorkProcessor() const {
	return new PSSMLTRenderer(m_config);
}

void PSSMLTProcess::develop() {
	LockGuard lock(m_resultMutex);
	size_t pixelCount = m_accum->getBitmap()->getPixelCount();
	const Spectrum *accum = (Spectrum *) m_accum->getBitmap()->getData();
	const Spectrum *direct = m_directImage != NULL ?
		(Spectrum *) m_directImage->getData() : NULL;
	const Float *importanceMap = m_config.importanceMap != NULL ?
			m_config.importanceMap->getFloatData() : NULL;
	Spectrum *target = (Spectrum *) m_developBuffer->getData();

	Float luminanceTargeted = m_config.luminance;
	if(m_config.recomputeNormalisation) {
	  Float fact = m_config.luminanceSamples / (Float)(m_config.luminanceSamples + m_stats.largeTotal);
	  Float approxLuminance = m_stats.luminanceLargeTotal /  m_stats.largeTotal;
	  luminanceTargeted = fact*luminanceTargeted + (1.f - fact)*approxLuminance;
	  Log(EInfo, "Luminance approximated: %f", luminanceTargeted);
	}

	/* Compute the luminance correction factor */
	Float avgLuminance = 0;
	if (importanceMap) {
		for (size_t i=0; i<pixelCount; ++i)
			avgLuminance += accum[i].getLuminance() * importanceMap[i];
	} else {
		for (size_t i=0; i<pixelCount; ++i)
			avgLuminance += accum[i].getLuminance();
	}

	avgLuminance /= (Float) pixelCount;
	Float luminanceFactor = luminanceTargeted / avgLuminance; // Find the correct ratio

	// === Scale all pixels values to ratio
	for (size_t i=0; i<pixelCount; ++i) {
		Float correction = luminanceFactor;
		if (importanceMap)
			correction *= importanceMap[i];
		Spectrum value = accum[i] * correction;
		if (direct)
			value += direct[i];
		target[i] = value;
	}

	if(m_accumBuffer == NULL) {
	  m_film->setBitmap(m_developBuffer);
	} else {
	  Float advancement = m_resultCounter / (Float)m_config.workUnits;
	  m_accumBuffer->add(m_developBuffer, advancement );
	  m_film->setBitmap(m_accumBuffer->accumulation.get());
	}


	m_refreshTimer->reset();

	m_queue->signalRefresh(m_job);
}

void PSSMLTProcess::processResult(const WorkResult *wr, bool cancelled) {
	LockGuard lock(m_resultMutex);
	// === Process results
	const ResultsPSSMLT *result = static_cast<const ResultsPSSMLT *>(wr);
	m_accum->put(result);
	m_progress->update(++m_resultCounter);
	m_refreshTimeout = std::min(2000U, m_refreshTimeout * 2);

	m_stats.add(result->getStats());

	bool developped = false;
	/* Re-develop the entire image every two seconds if partial results are
	   visible (e.g. in a graphical user interface). */
	if (m_job->isInteractive() && m_refreshTimer->getMilliseconds() > m_refreshTimeout) {
	  develop();
	  developped = true;
		if(m_config.pLarge < 0) {
			Log(EInfo, "PSSMLT Stats: %s", m_stats.toString().c_str());
		}
	}
	Float timerSec = m_timer->getMilliseconds() / 1000;
	if(timerSec > m_config.maxTimeImgDump) {
	  // === Develop image if necessary
	  if(!developped)
	    develop();
	  // === writing down image
	  {
       /// Path computation
       std::stringstream ss;
       ss << m_scene->getDestinationFile().c_str() << "_pass_" << (m_currentPass+1);
       std::string path = ss.str();

       /// Develop image
       m_film->setDestinationFile(path,0);
       m_film->develop(m_scene,0.f);

       /// Revert destination file
       m_film->setDestinationFile(m_scene->getDestinationFile(),0);
     }
	  // === writing down time
	  unsigned int milliseconds = m_timer->getMilliseconds();
	  m_timeFile << (milliseconds / 1000.f) << ",\n";
	  m_timeFile.flush();
    Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
      milliseconds % 1000);

    // === Print the statistic at each step of the rendering
    // to see the algorithm behaviors.
    Statistics::getInstance()->printStats();

	  ++m_currentPass;
    m_timer->reset();
	}
}

ParallelProcess::EStatus PSSMLTProcess::generateWork(WorkUnit *unit, int worker) {
	int timeout = 0;
	if (m_config.timeout > 0) {
		timeout = static_cast<int>(static_cast<int64_t>(m_config.timeout*1000) -
		          static_cast<int64_t>(m_timeoutTimer->getMilliseconds()));
	}

	if (m_workCounter >= m_config.workUnits || timeout < 0) {
		Log(EInfo, "Finish the working load? %i / %i", m_workCounter, m_config.workUnits);
	    return EFailure;
	}
	// === Create by choosing new seed
	SeedWorkUnitPSSMLT *workUnit = static_cast<SeedWorkUnitPSSMLT *>(unit);
	workUnit->setSeed(m_seeds[m_workCounter++]);
	workUnit->setTimeout(timeout); // For time comparison

	if(m_config.pLarge < 0) {
		workUnit->setProbLarge(m_stats.getProbLarge());
	} else {
		workUnit->setProbLarge(m_config.pLarge);
	}

	return ESuccess;
}

void PSSMLTProcess::bindResource(const std::string &name, int id) {
	ParallelProcess::bindResource(name, id);
	if (name == "sensor") {
		m_film = static_cast<Sensor *>(Scheduler::getInstance()->getResource(id))->getFilm();
		if (m_progress)
			delete m_progress;
		m_progress = new ProgressReporter("Rendering", m_config.workUnits, m_job);
		m_accum = new ImageBlock(Bitmap::ESpectrum, m_film->getCropSize());
		m_accum->clear();
		m_developBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_film->getCropSize());
	}
}

MTS_IMPLEMENT_CLASS(ResultsPSSMLT, false, ImageBlock)
MTS_IMPLEMENT_CLASS(SeedWorkUnitPSSMLT, false, WorkResult)
MTS_IMPLEMENT_CLASS_S(PSSMLTRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(PSSMLTProcess, false, ParallelProcess)
MTS_IMPLEMENT_CLASS(SeedWorkUnit, false, WorkUnit)

MTS_NAMESPACE_END
