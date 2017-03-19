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

#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>
#include "gpstruct.h"
#include "../initializeRadius.h"
#include "apa_splatting_proc.h"

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

/*!\plugin{sppm}{Stochastic progressive photon mapping integrator}
 * \order{8}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *     \parameter{photonCount}{\Integer}{Number of photons to be shot per iteration\default{250000}}
 *     \parameter{initialRadius}{\Float}{Initial radius of gather points in world space units.
 *         \default{0, i.e. decide automatically}}
 *     \parameter{alpha}{\Float}{Radius reduction parameter \code{alpha} from the paper\default{0.7}}
 *     \parameter{granularity}{\Integer}{
		Granularity of photon tracing work units for the purpose
		of parallelization (in \# of shot particles) \default{0, i.e. decide automatically}
 *     }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{maxPasses}{\Integer}{Maximum number of passes to render (where \code{-1}
 *        corresponds to rendering until stopped manually). \default{\code{-1}}}
 * }
 * This plugin implements stochastic progressive photon mapping by Hachisuka et al.
 * \cite{Hachisuka2009Stochastic}. This algorithm is an extension of progressive photon
 * mapping (\pluginref{ppm}) that improves convergence
 * when rendering scenes involving depth-of-field, motion blur, and glossy reflections.
 *
 * Note that the implementation of \pluginref{sppm} in Mitsuba ignores the sampler
 * configuration---hence, the usual steps of choosing a sample generator and a desired
 * number of samples per pixel are not necessary. As with \pluginref{ppm}, once started,
 * the rendering process continues indefinitely until it is manually stopped.
 *
 * \remarks{
 *    \item Due to the data dependencies of this algorithm, the parallelization is
 *    limited to to the local machine (i.e. cluster-wide renderings are not implemented)
 *    \item This integrator does not handle participating media
 *    \item This integrator does not currently work with subsurface scattering
 *    models.
 * }
 */
class SPPMIntegrator : public Integrator {
public:

	SPPMIntegrator(const Properties &props) : Integrator(props) {
		/* Initial photon query radius (0 = infer based on scene size and sensor resolution) */
		m_initialRadius = props.getFloat("initialRadius", 0);
		/* Alpha parameter from the paper (influences the speed, at which the photon radius is reduced) */
		m_alpha = props.getFloat("alpha", .7);
		/* Number of photons to shoot in each iteration */
		m_photonCount = props.getInteger("photonCount", 65500);
		/* Granularity of the work units used in parallelizing the
		   particle tracing task (default: choose automatically). */
		m_granularity = props.getInteger("granularity", 0);
		/* Longest visualized path length (<tt>-1</tt>=infinite). When a positive value is
		   specified, it must be greater or equal to <tt>2</tt>, which corresponds to single-bounce
		   (direct-only) illumination */
		m_maxDepth = props.getInteger("maxDepth", -1);
		/* Depth to start using russian roulette */
		m_rrDepth = props.getInteger("rrDepth", 3);
		/* Indicates if the gathering steps should be canceled if not enough photons are generated. */
		m_autoCancelGathering = props.getBoolean("autoCancelGathering", true);
		/* Maximum number of passes to render. -1 renders until the process is stopped. */
		m_maxPasses = props.getInteger("maxPasses", -1);
		m_mutex = new Mutex();
		m_beginPass = props.getInteger("beginPass", 0);

		/// Initialisation
    m_gpManager = getRadiusInitializer(props);

    /// === Radii image
    m_useRadiiImage = false;
    if(props.hasProperty("radiiImage")) {
      m_useRadiiImage = true;
      m_radiiImage = props.getAsString("radiiImage");
    }

    /// === Use Splatting rendering ?
    m_useSplatting = props.getBoolean("useSplatting", true);


		if (m_maxDepth <= 1 && m_maxDepth != -1)
			Log(EError, "Maximum depth must be set to \"2\" or higher!");
		if (m_maxPasses <= 0 && m_maxPasses != -1)
			Log(EError, "Maximum number of Passes must either be set to \"-1\" or \"1\" or higher!");
	}

	SPPMIntegrator(Stream *stream, InstanceManager *manager)
	 : Integrator(stream, manager) { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		Integrator::serialize(stream, manager);
		Log(EError, "Network rendering is not supported!");
	}

	void cancel() {
		m_running = false;
	}


	bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
			int sceneResID, int sensorResID, int samplerResID) {
		Integrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);

    if (m_initialRadius == 0) {
      /* Guess an initial radius if not provided
        (use scene width / horizontal or vertical pixel count) * 5 */
      Float rad = scene->getBSphere().radius;
      Vector2i filmSize = scene->getSensor()->getFilm()->getSize();

      m_initialRadius = std::min(rad / filmSize.x, rad / filmSize.y) * 5;
    }
		return true;
	}

	bool render(Scene *scene, RenderQueue *queue,
		const RenderJob *job, int sceneResID, int sensorResID, int unused) {
		ref<Scheduler> sched = Scheduler::getInstance();
		ref<Sensor> sensor = scene->getSensor();
		ref<Film> film = sensor->getFilm();
		size_t nCores = sched->getCoreCount();
		Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
			film->getCropSize().x, film->getCropSize().y,
			nCores, nCores == 1 ? "core" : "cores");

		Vector2i cropSize = film->getCropSize();
		Point2i cropOffset = film->getCropOffset();

		m_gatherBlocks.clear();
		m_running = true;
		m_totalEmitted = 0;
		m_totalPhotons = 0;

		ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
			createObject(MTS_CLASS(Sampler), Properties("independent")));

		/* Avoid seed problems */
		for(size_t m = 0; m < m_beginPass*nCores; m++) {
		  // Advance the sampler
		  sampler->next2D();
		}

		int blockSize = scene->getBlockSize();

		/* Allocate memory */
		m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
		m_bitmap->clear();
		int totalPixels = 0;
		for (int yofs=0; yofs<cropSize.y; yofs += blockSize) {
			for (int xofs=0; xofs<cropSize.x; xofs += blockSize) {
				m_gatherBlocks.push_back(std::vector<GatherPoint*>());
				m_offset.push_back(Point2i(cropOffset.x + xofs, cropOffset.y + yofs));
				std::vector<GatherPoint*> &gatherPoints = m_gatherBlocks[m_gatherBlocks.size()-1];
				int nPixels = std::min(blockSize, cropSize.y-yofs)
							* std::min(blockSize, cropSize.x-xofs);
				totalPixels += nPixels;
				gatherPoints.resize(nPixels);
			  for (int i=0; i<nPixels; ++i) {
          gatherPoints[i] = new GatherPoint(nCores);
          gatherPoints[i]->initRadius = m_initialRadius; // Just for the initialisation ...
        }
			}
		}

		/* Create a sampler instance for every core */
		std::vector<SerializableObject *> samplers(sched->getCoreCount());
		for (size_t i=0; i<sched->getCoreCount(); ++i) {
			ref<Sampler> clonedSampler = sampler->clone();
			clonedSampler->incRef();
			samplers[i] = clonedSampler.get();
		}

		int samplerResID = sched->registerMultiResource(samplers);

#ifdef MTS_DEBUG_FP
		enableFPExceptions();
#endif

#if defined(MTS_OPENMP)
		Thread::initializeOpenMP(nCores);
#endif

		m_gpManager->init(scene, m_maxDepth,
		      m_gatherBlocks, m_offset, samplers);

		m_gpManager->regenerateGP(false);
		// === Radii Image
		if(m_useRadiiImage) {
		  std::string scenePath = scene->getSourceFile().parent_path().string();
		  std::string totalPath = scenePath + "/" +m_radiiImage;
		  Log(EInfo, "Read the radii Image: %s", totalPath.c_str());
		  ref<FileStream> radiiStream = new FileStream(totalPath, FileStream::EReadOnly);
		  ref<Bitmap> radiiBitmap = new Bitmap(Bitmap::ERGBE, radiiStream);
		  // Block reading
		  for(size_t idBlock = 0; idBlock < m_gatherBlocks.size(); idBlock++) {
        // Local block reading
        std::vector<GatherPoint*> gps = m_gatherBlocks[idBlock];
        for(size_t idGP = 0; idGP < gps.size(); idGP++) {
          GatherPoint* gp = gps[idGP];
          gp->initRadius = radiiBitmap->getPixel(gp->pos).getLuminance();
        }
		  }
		}

		// Compute the current radius :
		m_currentRadiiReduction = 1.f;
		for(int i =0; i < m_beginPass; i++) {
		  m_currentRadiiReduction *= sqrt((i+m_alpha)/(i+1));
		}

		int it = 0;
		while (m_running && (m_maxPasses == -1 || it < m_maxPasses)) {
		  ref<GatherPointMap> gatherMap;

		  // === Regenerate GP
		  m_gpManager->regenerateGP(false);

		  // === If use Splatting, build the GP map
		  if(m_useSplatting) {
		    Log(EInfo, "Build GP Map... ");
		    gatherMap = new GatherPointMap(m_gatherBlocks, m_maxDepth);
		    m_gatherResID = sched->registerResource(gatherMap);
		  }

			photonMapPass(++it, queue, job, film, sceneResID,
					sensorResID, samplerResID);
			m_currentRadiiReduction *= sqrt((it+m_beginPass+m_alpha)/(it+m_beginPass+1));

			// === Free memory
			if(m_useSplatting) {
			  sched->unregisterResource(m_gatherResID);
			}
		}

#ifdef MTS_DEBUG_FP
		disableFPExceptions();
#endif

		for (size_t i=0; i<samplers.size(); ++i)
			samplers[i]->decRef();

		sched->unregisterResource(samplerResID);
		return true;
	}

	void photonMapPass(int it, RenderQueue *queue, const RenderJob *job,
			Film *film, int sceneResID, int sensorResID, int samplerResID) {
	  Log(EInfo, "Performing a photon mapping pass %i (" SIZE_T_FMT " photons so far)",
				it, m_totalPhotons);
		ref<Scheduler> sched = Scheduler::getInstance();
		ref<PhotonMap> photonMap = 0;
	  size_t nCores = sched->getCoreCount();

		int nbParticulesShooted;
		/* APA - Classical : Generate the global photon map */
		if(!m_useSplatting) {
      ref<GatherPhotonProcess> proc = new GatherPhotonProcess(
        GatherPhotonProcess::EAllSurfacePhotons, m_photonCount,
        m_granularity, m_maxDepth == -1 ? -1 : m_maxDepth-1, m_rrDepth, true,
        m_autoCancelGathering, job);

      proc->bindResource("scene", sceneResID);
      proc->bindResource("sensor", sensorResID);
      proc->bindResource("sampler", samplerResID);

      sched->schedule(proc);
      sched->wait(proc);

      Log(EInfo, "Building photon map..");
      photonMap = proc->getPhotonMap();
      photonMap->setScaleFactor(1 / (Float) proc->getShotParticles());
      photonMap->build();
      Log(EDebug, "Photon map full. Shot " SIZE_T_FMT " particles, excess photons due to parallelism: "
        SIZE_T_FMT, proc->getShotParticles(), proc->getExcessPhotons());
      m_totalEmitted += proc->getShotParticles();
      m_totalPhotons += photonMap->size();
      nbParticulesShooted = proc->getShotParticles();
		} else {
      /*
       * Make the rendering
       */
		  ref<SplattingPhotonProcess> proc = new SplattingPhotonProcess(
          m_photonCount, m_granularity, m_maxDepth == -1 ? -1 : m_maxDepth-1,
          m_rrDepth, job);

      proc->bindResource("scene", sceneResID);
      proc->bindResource("sensor", sensorResID);
      proc->bindResource("sampler", samplerResID);
      proc->bindResource("gathermap", m_gatherResID);

      sched->schedule(proc);
      sched->wait(proc);

      m_totalEmitted += proc->getNbEmittedPath();
      m_totalPhotons = m_totalEmitted;
      nbParticulesShooted = proc->getNbEmittedPath();
		}
		Log(EInfo, "Gathering ..");
		film->clear();
		#if defined(MTS_OPENMP)
			#pragma omp parallel for schedule(dynamic)
		#endif
		for (int blockIdx = 0; blockIdx<(int) m_gatherBlocks.size(); ++blockIdx) {
			std::vector<GatherPoint*> &gatherPoints = m_gatherBlocks[blockIdx];

			Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
			for (size_t i=0; i<gatherPoints.size(); ++i) {
				GatherPoint &gp = (*gatherPoints[i]);
				Spectrum flux(0.f), contrib(0.f);

				if (gp.depth != -1) {
				  if(!m_useSplatting) { //< If use classical APA (Photon map)
            Float currentRadii = m_currentRadiiReduction*gp.initRadius;
            photonMap->estimateRadianceRaw(gp.its,currentRadii,flux,
                                           m_maxDepth == -1 ? INT_MAX : (m_maxDepth-gp.depth));
            flux *= gp.weight / (nbParticulesShooted*currentRadii*currentRadii*M_PI);
				  } else { //< If use splatting version APA
				    /// Make the gathering
				    for(size_t idThread = 0; idThread < nCores; idThread++) {
              flux += gp.tempFlux[idThread];
            }
				    /// Compute Flux !
				    Float currentRadii = m_currentRadiiReduction*gp.initRadius;
				    flux *= gp.weight / (nbParticulesShooted*currentRadii*currentRadii*M_PI);
				  }
				}
				gp.reset();
				gp.flux = (gp.flux*(it-1))/it + (flux + gp.emission)/it;
				contrib = gp.flux;

				target[gp.pos.y * m_bitmap->getWidth() + gp.pos.x] = contrib;
			}
		}
		film->setBitmap(m_bitmap);
		queue->signalRefresh(job);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "APAIntegerator[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  photonCount = " << m_photonCount << "," << endl
			<< "  granularity = " << m_granularity << "," << endl
			<< "  maxPasses = " << m_maxPasses << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	// === Internal data
	std::vector<std::vector<GatherPoint*> > m_gatherBlocks;
	std::vector<Point2i> m_offset;
	ref<Mutex> m_mutex;
	ref<Bitmap> m_bitmap;
  RadiusInitializer* m_gpManager;

	// === Main parameters
	int m_photonCount, m_granularity;
	int m_maxDepth, m_rrDepth;
	bool m_autoCancelGathering;
	int m_maxPasses;
	Float m_initialRadius;
	Float m_alpha;
	int m_beginPass;

	// === Running data
	size_t m_totalEmitted, m_totalPhotons;
	bool m_running;

	// === Radii Management
	bool m_useRadiiImage;
	Float m_currentRadiiReduction;
	std::string m_radiiImage;

	// === Splatting management
	bool m_useSplatting;
	int m_gatherResID;
};

MTS_IMPLEMENT_CLASS_S(SPPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(SPPMIntegrator, "Stochastic progressive photon mapper");
MTS_NAMESPACE_END
