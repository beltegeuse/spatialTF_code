/*
 This file is part of Mitsuba, a physically based rendering system.

 Copyright (c) 2007-2012 by Wenzel Jakob and others.

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

#include <fstream>

#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>

#include "splatting.h"
#include "sppm_proc.h"
#include "sppm_splat.h"

#include "../pixeldata.h"
#include "../misWeights.h"
#include "../initializeRadius.h"
#include "../tools.h"

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

template <>
bool PixelData<GatherPoint>::usePhiStatistics = false;
template <>
bool PixelData<GatherPoint>::limitMaxGatherPoints = false;
template <>
int PixelData<GatherPoint>::nbChains = 1;
template <>
int PixelData<GatherPoint>::maxGatherPoints = 1; // Can be set to 1 even though we will use more than 1 gather point
Float MISHelper::lightPathRatio = 1.f;

#define VERBOSE_INFO 0

class SPPMSplatIntegrator : public Integrator {
 public:

  SPPMSplatIntegrator(const Properties &props)
      : Integrator(props) {
    /* Initial photon query radius (0 = infer based on the gather point bbox size and sensor size) */
    m_config.initialScale = props.getFloat("initialScale",1.f);
    /* Alpha parameter from the paper (influences the speed, at which the photon radius is reduced) */
    m_config.alpha = props.getFloat("alpha", .7);
    /* Number of photons to shoot in each iteration */
    m_photonCount = props.getInteger("photonCount", 250000);
    /* Granularity of the work units used in parallelizing the
     particle tracing task (default: choose automatically). */
    m_granularity = props.getInteger("granularity", 0);
    /* Longest visualized path length (<tt>-1</tt>=infinite). When a positive value is
     specified, it must be greater or equal to <tt>2</tt>, which corresponds to single-bounce
     (direct-only) illumination */
    m_config.maxDepth = props.getInteger("maxDepth", -1);
    m_config.minDepth = props.getInteger("minDepth", 0);  //< For rendering in some case indirect only
    /* Depth to start using russian roulette */
    m_config.rrDepth = props.getInteger("rrDepth", 3);
    /* Indicates if the gathering steps should be canceled if not enough photons are generated. */
    m_autoCancelGathering = props.getBoolean("autoCancelGathering", true);
    m_mutex = new Mutex();
    /* If the user want to finish the rendering process at a given time */
    m_maxRenderingTime = props.getInteger("maxRenderingTime", INT_MAX);
    m_maxPass = props.getInteger("maxPass", INT_MAX);
    m_stepSnapshot = props.getInteger("stepSnapshot", INT_MAX);
    m_stepDensitySnapshot = props.getInteger("stepDensity", 10);
    m_dumpAtEachIterationRadii = true;

    // Sampling configuration of the reproductibility
    // and the reference Mod
    m_referenceMod = props.getBoolean("referenceMod", false);

    m_computeDirect = props.getBoolean("computeDirect", true);
    m_dumpImagePass = props.getBoolean("dumpImagePass", false);

	// Techniques (VCM, SPPM..)
	std::string technique = props.getString("technique","sppm");
	MISHelper::parse(technique, m_config.usedTechniques, m_config.multiGatherPoints);

	m_config.removeDeltaPaths = props.getBoolean("removeDeltaPaths",true);

    //////////////////////////////////////////////////////////
    // Testing ...
    //////////////////////////////////////////////////////////
    if (m_config.maxDepth <= 1 && m_config.maxDepth != -1)
      Log(EError, "Maximum depth must be set to \"2\" or higher!");
    if (m_maxRenderingTime != INT_MAX && m_maxPass != INT_MAX) {
      Log(EError, "Max pass and time is incompatible!");
    }

    m_gpManager = new RadiusInitializer(props);

    // Internal values
    m_running = false;
    m_totalEmittedPath = 0;
  }

  SPPMSplatIntegrator(Stream *stream, InstanceManager *manager)
      : Integrator(stream, manager) {
    Log(EError, "Network rendering is not supported!");
  }

  virtual ~SPPMSplatIntegrator() {
    delete m_gpManager;
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Integrator::serialize(stream, manager);
    Log(EError, "Network rendering is not supported!");
  }

  void cancel() {
    m_running = false;
  }

  bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
                  int sceneResID, int sensorResID, int samplerResID) {
    Integrator::preprocess(scene, queue, job, sceneResID, sensorResID,
                           samplerResID);
    // Show the configuration
    // of the intergration technique
    m_config.dump();

    // If we want to reproduce the sampling
    // Autotune the granularity of shooting photons
    if (!m_referenceMod) {
      m_granularity = (int)std::max(
          (size_t) 1,
          m_photonCount / (Scheduler::getInstance()->getWorkerCount()));
      Log(EInfo, "Automatic granularity choice (to be reproductible): %i",
          m_granularity);
    }

    return true;
  }

  bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
              int sceneResID, int sensorResID, int unused) {

    // Force to use correct emitter weight
    scene->recomputeWeightEmitterFlux();

    // Get all data from MTS
    // to render the scene
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = scene->getSensor();
    ref<Film> film = sensor->getFilm();
    size_t nCores = sched->getCoreCount();
    Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
        film->getCropSize().x, film->getCropSize().y, nCores,
        nCores == 1 ? "core" : "cores");

    // Image related informations
    Vector2i cropSize = film->getCropSize();
    Point2i cropOffset = film->getCropOffset();
    int blockSize = scene->getBlockSize();
    MISHelper::lightPathRatio = m_photonCount / (Float)(cropSize.x * cropSize.y);

    // Clear all internal representation
    // before launch the rendering process
    m_gatherBlocks.clear();
    m_running = true;
    m_totalEmittedPath = 0;

    // Allocate the memory of
    //  - the bitmap to show the results
    //  - gather points (regrouped in image's blocks)
    m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
    m_bitmap->clear();
    int totalPixels = 0;
	// Prealloc for gather points
	char * gatherPointsData = new char[cropSize.x * cropSize.y * GatherPointsList::allocTempSize((int)nCores)];
	char * ptr = gatherPointsData;
	m_gatherBlocks.reserve(((cropSize.y + blockSize - 1) / blockSize) * ((cropSize.x + blockSize - 1) / blockSize) );
    for (int yofs = 0; yofs < cropSize.y; yofs += blockSize) {
      for (int xofs = 0; xofs < cropSize.x; xofs += blockSize) {
        m_gatherBlocks.push_back(GatherBlock());
        m_offset.push_back(Point2i(cropOffset.x + xofs, cropOffset.y + yofs));

        GatherBlock &gatherLists = m_gatherBlocks.back();  // Get the last
        int nPixels = std::min(blockSize, cropSize.y - yofs)
            * std::min(blockSize, cropSize.x - xofs);
        totalPixels += nPixels;

        // Create all GP and initialized it
        gatherLists.resize(nPixels);
        for (int i = 0; i < nPixels; ++i) {
		  // Allocate and init first gather point
		  gatherLists[i].allocTemp((int)nCores, ptr);
		  gatherLists[i].scale = m_config.initialScale;
        }
      }
    }

    // Create the independent samplers
    // These samplers will be used to generate path
    // from the light sources
    Log(EInfo, "Create SPPM samplers ... ");
    Properties propsIndepent("independent");
    if (m_referenceMod) {
      propsIndepent.setBoolean("randInit", true);
    }
    ref<Sampler> samplerIndependent =
        static_cast<Sampler *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(Sampler),propsIndepent));

    std::vector<SerializableObject *> samplersIndependent(
        sched->getCoreCount());
    for (size_t i = 0; i < sched->getCoreCount(); ++i) {
      ref<Sampler> clonedIndepSampler = samplerIndependent->clone();
      clonedIndepSampler->incRef();
      samplersIndependent[i] = clonedIndepSampler.get();
    }
    int samplerResID = sched->registerMultiResource(samplersIndependent);

#ifdef MTS_DEBUG_FP
    enableFPExceptions();
#endif

#if defined(MTS_OPENMP)
    Thread::initializeOpenMP(nCores);
#endif

    // Initialize the class responsible to the GP genereation
    // and the radii initialization
    m_gpManager->init(scene, m_config.maxDepth, m_gatherBlocks, m_offset, 
		m_config.usedTechniques);

    // Create the files to dump information about the rendering
    // Also create timer to track algorithm performance
    std::string timeFilename = scene->getDestinationFile().string()
        + "_time.csv";
    std::ofstream timeFile(timeFilename.c_str());
    ref<Timer> renderingTimer = new Timer;

    // === For all iterations ...
    int currentIteration = 1;
    while (m_running && m_maxPass > 0) {
      ////////////////////////////////
      // Step 1: Generation GP
      ////////////////////////////////

      // Generate GP in the scene
      // Also, if needed, computes the initialize radii
      // Then create the gather map
	  Log(EInfo, "Regenerating gather points positions and radius!");
      m_gpManager->regeneratePositionAndRadius();
	  Log(EInfo, "Done regenerating!");
      m_gpManager->rescaleFlux();
      Log(EInfo, "Build the gather map... pass: %i", currentIteration);
      ref<GatherPointMap> gatherMap = new GatherPointMap(m_gatherBlocks,
														 scene,
														 &samplersIndependent,
                                                         (Float)m_config.maxDepth,
														 m_config.usedTechniques,
														 (int)m_photonCount);
      int gatherMapID = sched->registerResource(gatherMap);
      if (currentIteration == 1) {
        writeRadii(currentIteration, scene, true);  //< dump initialize radii init
      }

      ////////////////////////////////
      // Step 2: Rendering
      ////////////////////////////////

      // Shoot photon procedure
      // returns the number of path shooted
      int nbEmittedPath = shootPhotons(currentIteration, sceneResID,
                                       sensorResID, samplerResID, gatherMapID,
                                       job);
      m_totalEmittedPath += nbEmittedPath;

      ////////////////////////////////
      // Step 3: Update statistics
      ////////////////////////////////
      updatePixelsStatistics(currentIteration, nbEmittedPath, film, queue, job,
                             scene, nCores);

      ////////////////////////////////
      // Addition step: Dump data +
      // prepare the next iteration
      ////////////////////////////////

      // Write down some
      writeSnapshot(currentIteration, scene);
#if VERBOSE_INFO
      writeDensity<GatherPoint>(scene, currentIteration, m_stepDensitySnapshot,
                                m_gatherBlocks);
      writeRadii(currentIteration, scene);
#endif

      // === Update the log time
      unsigned int milliseconds = renderingTimer->getMilliseconds();
      timeFile << (milliseconds / 1000.f) << ",\n";
      timeFile.flush();
      Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
          milliseconds % 1000);
      m_maxRenderingTime -= (milliseconds / 1000);
      if (m_maxRenderingTime < 0) {
        m_running = false;
        Log(EInfo, "Max time reaching !");
      }

      // === update variables && free memory
      renderingTimer->reset();
      sched->unregisterResource(gatherMapID);
      m_maxPass--;
      ++currentIteration;  //< For the next iteration
    }

#ifdef MTS_DEBUG_FP
    disableFPExceptions();
#endif

    ////////////////////////////////
    // Free memory
    ////////////////////////////////
    for (size_t i = 0; i < samplersIndependent.size(); ++i)
      samplersIndependent[i]->decRef();
    sched->unregisterResource(samplerResID);

	delete [] gatherPointsData;

    timeFile.close();
    return true;
  }

  int shootPhotons(int currentIteration, int sceneResID, int sensorResID,
                   int samplerResID, int gatherResID, const RenderJob *job) {
	if ( !(m_config.usedTechniques & MISHelper::LIGHTPATHS) ) { 
		return 0;
	}
    Log(EInfo, "Performing a photon mapping pass %i P COunt: %i", currentIteration,m_photonCount);
    ref<Scheduler> sched = Scheduler::getInstance();

    // Create the process responsible to shoot photon
    // and cumulate their contributions into the gather points
    ref<SplattingPhotonProcess> proc = new SplattingPhotonProcess(
        m_photonCount, m_granularity,
        m_config.maxDepth == -1 ? -1 : m_config.maxDepth - 1, m_config.rrDepth,
        job);

    // Give to the process all the information needed
    proc->bindResource("scene", sceneResID);
    proc->bindResource("sensor", sensorResID);
    proc->bindResource("sampler", samplerResID);
    proc->bindResource("gathermap", gatherResID);

    // Launch the process and wait that its finish
    sched->schedule(proc);
    sched->wait(proc);

	Log(EInfo, "Finished with count: %i", proc->getNbEmittedPath());

    return proc->getNbEmittedPath();
  }

  void updatePixelsStatistics(int currentIteration, int nbEmittedPath,
                              Film *film, RenderQueue *queue,
                              const RenderJob *job, Scene* scene,
                              size_t nCores) {
    Log(EInfo, "Update Gather points ..");
    // This is the bitmap
    // to accumulate only the iteration contribution
    ref<Bitmap> bitmapOnlyIter = m_bitmap->clone();

    film->clear();
//#if defined(MTS_OPENMP)
//    // schedule(dynamic)
//#pragma omp parallel for
//#endif
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
      GatherBlock &gatherBlock = m_gatherBlocks[blockIdx];

      Spectrum *targetOnlyIter = (Spectrum *) bitmapOnlyIter->getUInt8Data();
      Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
      for (size_t i = 0; i < gatherBlock.size(); ++i) {
        GatherPointsList &gps = gatherBlock[i];
		// Iterate through all gather points corresponding to pixel "i" in gather block "blockIdx"
		Spectrum contribPixel(0.f), contribOnlyIterPixel(0.f);
		Float M = 0; // M: Number of photons collected to this pass
		// collect all the statistics
		for (size_t idThread = 0; idThread < nCores; idThread++) {
			M += (Float) gps.tempM[idThread];
		}
		gps.nPhotons += M;
		Spectrum flux = gps.getFlux(0); // For now flux 0
		// Iterate through all gps
		if (m_config.usedTechniques & MISHelper::SPPM_ONLY) {
			for (GatherPointsList::iterator it = gps.begin(); it != gps.end(); ++it) {
				// Add contribution for SPPM
				if (it->depth != -1 && it->its.isValid()) {
					// Update GP
					Float N = gps.N;
					if (N + M == 0)
						break;
					Float ratio = (N + m_config.alpha * M) / (N + M);
					gps.N = N + m_config.alpha * M;
					gps.scale = gps.scale * std::sqrt(ratio);
					// === Debug code part
					// This code is responsible to computing the
					// contribution image for only the current
					// iteration
					if (m_dumpImagePass && it->points->radius != 0.f) {
						Spectrum fluxPass = it->weight * flux;
						fluxPass /= nbEmittedPath * it->points->radius * it->points->radius * M_PI;
						contribOnlyIterPixel += fluxPass;
					}
					////////////
					// Step a: Compute the indirect component
					////////////
					if (it->points->radius == 0.f) {
						if (M != 0) {
							SLog(EError, "Null radius but collected some photons");
						}
					}
					else {
						gps.flux += it->weight * flux;
						gps.flux /= (it->points->radius * it->points->radius * M_PI);
					}
					break;
				}
			}
		}
		if ( !(m_config.usedTechniques & MISHelper::SPPM_ONLY) &&
			m_config.usedTechniques & (MISHelper::MERGE | MISHelper::CONNECT))
		{
			// MERGE & CONNECT from VCM, BPM
			gps.scale = (Float)(m_config.initialScale * std::pow((Float)currentIteration, (m_config.alpha - 1.f) * 0.5f));
			contribOnlyIterPixel += flux;
			Float denom = 1.f / (Float) currentIteration;
			gps.flux *= (Float) currentIteration - 1.f;
			gps.flux += flux;
			gps.flux *= denom;
			contribPixel += gps.flux;
		}
		else if ( m_config.usedTechniques & MISHelper::SPPM_ONLY )
		{
			////////////
			// Step b: Normalize and merge the two component
			////////////
			contribPixel = gps.flux / ((Float) m_totalEmittedPath);
		}
		// Handle direct illumination
		Spectrum directIllum = 
			m_config.usedTechniques & MISHelper::DIRLIGHT ?
				MISHelper::computeWeightedDirectIllum(gps,m_config.usedTechniques,(int)m_photonCount,1.f, m_config.removeDeltaPaths, 0):
				Spectrum(0.f);
		contribOnlyIterPixel += directIllum;
		Float denom = 1.f;
		if (m_config.minDepth <= 1) {
			denom = 1.f / (Float) currentIteration;
			gps.fluxDirect = (gps.fluxDirect * ((Float) currentIteration - 1.f)
				+ directIllum) * denom;
			contribPixel += gps.fluxDirect;
		}
        // No filter, we write directly in the bitmap
        target[gps.pos.y * m_bitmap->getWidth() + gps.pos.x] = contribPixel;
        targetOnlyIter[gps.pos.y * m_bitmap->getWidth() + gps.pos.x] =
            contribOnlyIterPixel;
      }
    }

    // === Debug: If it was requested
    // dump the contrib image
    if (m_dumpImagePass) {
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_contrib_"
         << currentIteration;
      std::string path = ss.str();

      film->setBitmap(bitmapOnlyIter);
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }
#if VERBOSE_INFO
	  saveBitmap(scene, passClamped.get(), "Clamped", currentIteration);
#endif

    /* Update the bitmap  */
    film->setBitmap(m_bitmap);
    queue->signalRefresh(job);
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "SPPMSplat[" << endl << "]";
    return oss.str();
  }

  MTS_DECLARE_CLASS()
 protected:
  void writeSnapshot(int it, Scene* scene) {
    if (((it - 1) % m_stepSnapshot) == 0) {
      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_pass_" << it;
      std::string path = ss.str();
      ////////////////////////////////////////////////
      // Write image
      ////////////////////////////////////////////////
      Film* film = scene->getFilm();
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
    }
  }

  void writeRadii(int it, Scene* scene, bool initialize = false) {
	  if ( !(m_config.usedTechniques & MISHelper::MERGE) ) {
		  return;
	  }
    if (!initialize && !m_dumpAtEachIterationRadii) {
      return;  //< No operation is needed
    }
    if (((it - 1) % m_stepSnapshot) != 0) {
      return;
    }
	Float norm = 1.f / std::max(it - 1, 1 );


    Film* film = scene->getFilm();

    // === Print out image radius
    // Build name
#if VERBOSE_INFO
    std::stringstream ss;
    if (initialize) {
      ss << scene->getDestinationFile().c_str() << "_radiusInit";
    } else {
      ss << scene->getDestinationFile().c_str() << "_radius_" << it;
    }
    std::string path = ss.str();

    // Build the bitmap contains all radius
    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
                                    film->getSize());
	ref<Bitmap> bitmapC = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
		film->getSize());
    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
	  Spectrum *targetC = (Spectrum *) bitmapC->getData();
      for (int j = 0; j < (int) gps.size(); j++) {
        GatherPointsList &gatherPointsList = gps[j];
		for( GatherPointsList::iterator gp = gatherPointsList.begin(); gp != gatherPointsList.end(); ++gp ) {
			target[gatherPointsList.pos.y * bitmap->getWidth() + gatherPointsList.pos.x] =
				Spectrum(gp->points->radius);
		}
		targetC[gatherPointsList.pos.y * bitmapC->getWidth() + gatherPointsList.pos.x] = Spectrum(
			gatherPointsList.nPhotons) * norm;
      }
    }
	saveBitmap(scene, bitmapC.get(), "count", it);
	film->setBitmap(bitmap);
	film->setDestinationFile(path, 0);
  film->develop(scene, 0.f);
#endif
  }

  void saveBitmap(Scene* scene, Bitmap* bitmap, const std::string& suffix, int idPass) {
	  Film* film = scene->getFilm();
	  std::stringstream ss;
	  ss << scene->getDestinationFile().c_str() << "_" << suffix <<  "_pass_" <<idPass;
	  std::string path = ss.str();

	  film->setBitmap(bitmap);
	  fs::path oldPath = scene->getDestinationFile();
	  film->setDestinationFile(path, 0);
	  film->develop(scene, 0.f);
	  // Revert name image
	  film->setDestinationFile(oldPath, 0);
  }

 private:
  GatherBlocks m_gatherBlocks;
  std::vector<Point2i> m_offset;
  ref<Mutex> m_mutex;
  ref<Bitmap> m_bitmap;
  RadiusInitializer* m_gpManager;

  SPPMSplatConfig m_config;
  int m_photonCount, m_granularity;
  int m_maxPass;
  bool m_computeDirect;
  bool m_dumpAtEachIterationRadii;

  size_t m_totalEmittedPath;

  bool m_running;
  bool m_autoCancelGathering;
  int m_maxRenderingTime;
  int m_stepSnapshot;
  int m_stepDensitySnapshot;

  /// == Debug display
  bool m_dumpImagePass;
  bool m_referenceMod;
};

MTS_IMPLEMENT_CLASS_S(SPPMSplatIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(SPPMSplatIntegrator,
                  "Stochastic progressive photon mapper with splatting");
MTS_NAMESPACE_END
