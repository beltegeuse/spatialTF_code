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
#include <mitsuba/core/fstream.h>
#include "mlt_proc.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{mlt}{Path Space Metropolis Light Transport}
 * \order{10}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{directSamples}{\Integer}{
 *	       By default, the implementation renders direct illumination component
 *	       separately using the \pluginref{direct} plugin, which
 *	       uses low-discrepancy number sequences for superior performance
 *	       (in other words, it is \emph{not} handled by MLT). This
 *	       parameter specifies the number of samples allocated to that method. To
 *	       force MLT to be responsible for the direct illumination
 *	       component as well, set this to \code{-1}. \default{16}
 *	   }
 *	   \parameter{luminanceSamples}{\Integer}{
 *	      MLT-type algorithms create output images that are only
 *	      \emph{relative}. The algorithm can e.g. determine that a certain pixel
 *	      is approximately twice as bright as another one, but the absolute
 *	      scale is unknown. To recover it, this plugin computes
 *	      the average luminance arriving at the sensor by generating a
 *	      number of samples. \default{\code{100000} samples}
 *     }
 *     \parameter{twoStage}{\Boolean}{Use two-stage MLT?
 *       See \pluginref{pssmlt} for details.\!\default{{\footnotesize\code{false}}}\!}
 *	   \parameter{bidirectional\showbreak\newline Mutation,\vspace{1mm}
 *	      [lens,multiChain,\newline caustic,manifold]\showbreak\newline Perturbation}{\Boolean}{
 *	     These parameters can be used to pick the individual mutation and perturbation
 *	     strategies that will be used to explore path space. By default, the original set
 *	     by Veach and Guibas is enabled (i.e. everything except the manifold
 *	     perturbation). It is possible to extend
 *	     this integrator with additional custom perturbations strategies if needed.
 *	   }
 *	   \parameter{lambda}{\Float}{
 *	       Jump size of the manifold perturbation \default{50}}
 * }
 * Metropolis Light Transport (MLT) is a seminal rendering technique proposed by Veach and
 * Guibas \cite{Veach1997Metropolis}, which applies the Metropolis-Hastings
 * algorithm to the path-space formulation of light transport.
 * Please refer to the \pluginref{pssmlt} page for a general description of MLT-type
 * algorithms and a list of caveats that also apply to this plugin.
 *
 * Like \pluginref{pssmlt}, this integrator explores the space of light paths,
 * searching with preference for those that carry a significant amount of
 * energy from an emitter to the sensor. The main difference is that PSSMLT
 * does this exploration by piggybacking on another rendering technique and
 * ``manipulating'' the random number stream that drives it, whereas MLT does
 * not use such an indirection: it operates directly on the actual light
 * paths.
 *
 * This means that the algorithm has access to considerably more
 * information about the problem to be solved, which allows it to perform a
 * directed exploration of certain classes of light paths. The main downside
 * is that the implementation is rather complex, which may make it more
 * susceptible to unforeseen problems.
 * Mitsuba reproduces the full MLT
 * algorithm except for the lens subpath mutation\footnote{In experiments,
 * it was not found to produce sigificant convergence improvements and was
 * subsequently removed.}. In addition, the plugin also provides the
 * manifold perturbation proposed by Jakob and Marschner \cite{Jakob2012Manifold}.
 *
 * \renderings{
 *    \includegraphics[width=\textwidth]{images/integrator_mlt_sketch.pdf}\hfill\,
 * }
 *
 * To explore the space of light paths, MLT iteratively makes changes
 * to a light path, which can either be large-scale \emph{mutations} or small-scale
 * \emph{perturbations}. Roughly speaking, the \emph{bidirectional mutation} is used
 * to jump between different classes of light paths, and each one of the perturbations is
 * responsible for efficiently exploring some of these classes.
 * All mutation and perturbation strategies can be mixed and matched as
 * desired, though for the algorithm to work properly, the bidirectional
 * mutation must be active and perturbations should be selected
 * as required based on the types of light paths that are present in the
 * input scene. The following perturbations are available:
 *
 * \begin{enumerate}[(a)]
 * \item \emph{Lens perturbation}: this perturbation slightly varies the outgoing
 * direction at the camera and propagates the resulting ray until it encounters
 * the first non-specular object. The perturbation then attempts to create a connection to the
 * (unchanged) remainder of the path.
 * \item \emph{Caustic perturbation}: essentially a lens perturbation
 * that proceeds in the opposite direction.
 * \item \emph{Multi-chain perturbation}: used when there are several chains
 * of specular interactions, as seen in the swimming pool example above.
 * After an initial lens perturbation, a cascade of additional perturbations
 * is required until a connection to the
 * remainder of the path can finally be established. Depending on the
 * path type, the entire path may be changed by this.
 * \item \emph{Manifold perturbation}: this perturbation was designed to
 * subsume and extend the previous three approaches.
 * It creates a perturbation at an arbitrary
 * position along the path, proceeding in either direction. Upon encountering
 * a chain of specular interactions, it numerically solves for a
 * connection path (as opposed to the cascading mechanism employed by the
 * multi-chain perturbation).
 * \end{enumerate}
 */
class MLT : public Integrator {
public:
	MLT(const Properties &props) : Integrator(props) {
		/* Longest visualized path length (<tt>-1</tt>=infinite).
		   A value of <tt>1</tt> will visualize only directly visible light
		   sources. <tt>2</tt> will lead to single-bounce (direct-only)
		   illumination, and so on. */
		m_config.maxDepth = props.getInteger("maxDepth", -1);

		/* This setting can be very useful to reduce noise in dark regions
		   of the image: it activates two-stage MLT, where a nested MLT renderer
		   first creates a tiny version of the output image. In a second pass,
		   the full version is then rendered, while making use of information
		   about the image-space luminance distribution found in the first
		   pass. Two-stage MLT is very useful in making the noise characteristics
		   more uniform over time image -- specifically, since MLT tends to get
		   stuck in very bright regions at the cost of the remainder of the image.*/
		//m_config.twoStage = props.getBoolean("twoStage", false);
		m_config.stagedTechnique = MLTStageTechnique(props.getString("stagedTechnique"));
        if(m_config.stagedTechnique == EUnknowStage) {
          Log(EError, "Unknow staged technique");
        }

		/* When running two-stage MLT, this parameter influences the size
		   of the downsampled image created in the first pass (i.e. setting this
		   to 16 means that the horizontal/vertical resolution will be 16 times
		   lower). When the two-stage process introduces noisy halos around
		   very bright image regions, it might might be good to reduce this
		   parameter to 4 or even 1. Generally though, it should be safe to leave
		   it unchanged. */
		m_config.firstStageSizeReduction = props.getInteger("firstStageSizeReduction", 16);

		/* Used internally to let the nested rendering process of a
		   two-stage MLT approach know that it is running the first stage */
		m_config.firstStage= props.getBoolean("firstStage", false);

		/* Number of samples used to estimate the total luminance
		   received by the scene's sensor */
		m_config.luminanceSamples = props.getInteger("luminanceSamples", 100000);

		/* This parameter can be used to specify the samples per pixel used to
		   render the direct component. Should be a power of two (otherwise, it will
		   be rounded to the next one). When set to zero or less, the
		   direct illumination component will be hidden, which is useful
		   for analyzing the component rendered by MLT. When set to -1,
		   MLT will handle direct illumination as well */
		m_config.directSamples = props.getInteger("directSamples", 16);
		m_config.separateDirect = m_config.directSamples >= 0;

		/* Specifies the number of parallel work units required for
		   multithreaded and network rendering. When set to <tt>-1</tt>, the
		   amount will default to four times the number of cores. Note that
		   every additional work unit entails a significant amount of
		   communication overhead (a full-sized floating put image must be
		   transmitted), hence it is important to set this value as low as
		   possible, while ensuring that there are enough units to keep all
		   workers busy. */
		m_config.workUnits = props.getInteger("workUnits", -1);

		/* Selectively enable/disable the bidirectional mutation */
		m_config.bidirectionalMutation = props.getBoolean("bidirectionalMutation", true);

		/* Selectively enable/disable the lens perturbation */
		m_config.lensPerturbation = props.getBoolean("lensPerturbation", true);

		/* Selectively enable/disable the caustic perturbation */
		m_config.causticPerturbation = props.getBoolean("causticPerturbation", true);

		/* Selectively enable/disable the multi-chain perturbation */
		m_config.multiChainPerturbation = props.getBoolean("multiChainPerturbation", true);

		/* Selectively enable/disable the manifold perturbation */
		m_config.manifoldPerturbation = props.getBoolean("manifoldPerturbation", false);
		m_config.probFactor = props.getFloat("probFactor", props.getFloat("lambda", 50));

		/* Stop MLT after X seconds -- useful for equal-time comparisons */
		m_config.timeout = props.getInteger("timeout", 0);

		m_config.maxTimeImgDump = props.getInteger("maxTimeImgDump", INT_MAX);

		if(m_config.separateDirect && m_config.stagedTechnique == EMultiStage) {
           Log(EError, "Impossible to separte the direct with multi-stage");
        }
	}

	/// Unserialize from a binary data stream
	MLT(Stream *stream, InstanceManager *manager)
	 : Integrator(stream, manager) {
		m_config = MLTConfiguration(stream);
	}

	virtual ~MLT() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		Integrator::serialize(stream, manager);
		m_config.serialize(stream);
	}

	bool preprocess(const Scene *scene, RenderQueue *queue,
			const RenderJob *job, int sceneResID, int sensorResID,
			int samplerResID) {
		Integrator::preprocess(scene, queue, job, sceneResID,
				sensorResID, samplerResID);

		if (scene->getSubsurfaceIntegrators().size() > 0)
			Log(EError, "Subsurface integrators are not supported by MLT!");

		if (scene->getSampler()->getClass()->getName() != "IndependentSampler")
			Log(EError, "Metropolis light transport requires the independent sampler");

		return true;
	}

	void cancel() {
		ref<RenderJob> nested = m_nestedJob;
		if (nested)
			nested->cancel();
		Scheduler::getInstance()->cancel(m_process);
	}

	bool launchRendering(Scene *scene, RenderQueue *queue, const RenderJob *job,
	                         int sceneResID, int sensorResID, int samplerResID,
	                         // New attributes
	                         ref<Timer> renderingTimer, int& currentPass,  std::ofstream& timeFile,
	                         bool nested, Random* randomNumber, const Film *film,
	                         size_t sampleCount, MLTAccumulBuffer* accumBuff,
	                         int currentIteration) {
	    ref<Scheduler> scheduler = Scheduler::getInstance();
	    size_t nCores = scheduler->getCoreCount();

	    Vector2i cropSize = film->getCropSize();
        Assert(cropSize.x > 0 && cropSize.y > 0);
        Log(EInfo, "Starting %srender job (%ix%i, " SIZE_T_FMT
            " %s, " SSE_STR ", approx. " SIZE_T_FMT " mutations/pixel) ..",
            nested ? "nested " : "", cropSize.x, cropSize.y,
            nCores, nCores == 1 ? "core" : "cores", sampleCount);

        if (m_config.workUnits <= 0) {
            const size_t desiredMutationsPerWorkUnit = 200000;
            const size_t cropArea  = (size_t) cropSize.x * cropSize.y;
            const size_t workUnits = ((desiredMutationsPerWorkUnit - 1) +
                (cropArea * sampleCount)) / desiredMutationsPerWorkUnit;
            Assert(workUnits <= (size_t) std::numeric_limits<int>::max());
            m_config.workUnits = (int) std::max(workUnits, (size_t) 1);
        }

        size_t luminanceSamples = m_config.luminanceSamples;
        if (luminanceSamples < (size_t) m_config.workUnits * 10) {
            luminanceSamples = (size_t) m_config.workUnits * 10;
            Log(EWarn, "Warning: increasing number of luminance samples to " SIZE_T_FMT,
                luminanceSamples);
        }

        m_config.nMutations = (cropSize.x * cropSize.y *
            sampleCount) / m_config.workUnits;

        ref<Bitmap> directImage;
        if (m_config.separateDirect && m_config.directSamples > 0 && !nested) {
            directImage = BidirectionalUtils::renderDirectComponent(scene,
                sceneResID, sensorResID, queue, job, m_config.directSamples);
            if (directImage == NULL)
                return false;
        }

        ref<ReplayableSampler> rplSampler = new ReplayableSampler(false, currentIteration);
        ref<PathSampler> pathSampler = new PathSampler(PathSampler::EBidirectional, scene,
            rplSampler, rplSampler, rplSampler, m_config.maxDepth, 10,
            m_config.separateDirect, true);

        std::vector<PathSeed> pathSeeds;
        ref<MLTProcess> process = new MLTProcess(job, queue,
                m_config, directImage, pathSeeds, renderingTimer, scene,
                timeFile, currentPass, accumBuff);

        m_config.luminance = pathSampler->generateSeeds(luminanceSamples,
            m_config.workUnits, true, m_config.importanceMap, pathSeeds, 10.f);
        m_config.luminanceSamples = luminanceSamples; // Re put the modification

        if(accumBuff != NULL) {
          accumBuff->addNormalization(m_config.luminance, luminanceSamples);
          m_config.luminance = accumBuff->getNormalization();
        }

        if (!nested)
            m_config.dump();

        int rplSamplerResID = scheduler->registerResource(rplSampler);

        process->bindResource("scene", sceneResID);
        process->bindResource("sensor", sensorResID);
        process->bindResource("sampler", samplerResID);
        process->bindResource("rplSampler", rplSamplerResID);

        m_process = process;
        scheduler->schedule(process);
        scheduler->wait(process);
        m_process = NULL;
        process->develop();
        scheduler->unregisterResource(rplSamplerResID);

        return process->getReturnStatus() == ParallelProcess::ESuccess;
	}

	void saveImportance(Scene* scene, const std::string& path) {
      ////////////////////////////////////////
      /// Compute Importance function to display it
      ////////////////////////////////////////
      // === Compute the importance function
      Vector2i size = m_config.importanceMap->getSize();
      ref<Bitmap> impMap = m_config.importanceMap->clone();
      const Float *luminanceValues = m_config.importanceMap->getFloatData();
      Float *importanceValues = impMap->getFloatData();
      for (int i = 0; i < impMap->getWidth(); ++i) {
            for (int j = 0; j < impMap->getHeight(); j++) {
              if(luminanceValues[i + j * size.x] > 0) {
                importanceValues[i + j * size.x] = 1.f / luminanceValues[i + j * size.x];
              } else {
                importanceValues[i + j * size.x] = 0.f;
              }
            }
      }

      // === Write down the importance function
      Film* filmSave = scene->getFilm();
      filmSave->setBitmap(impMap.get());
      fs::path oldPath = scene->getDestinationFile();
      filmSave->setDestinationFile(path, 0);
      filmSave->develop(scene, 0.f);
      // Revert name image
      filmSave->setDestinationFile(oldPath, 0);
    }

	/**
	 * A method to compute the number of samples needed
	 * to acheive the rendering for a dedicated time
	 *
	 * The algorithm work as follow:
	 *  1) Run with few samples and look how many time it takes
	 *  2) Scale up with 30% extra samples to deduce how many samples needed
	 */
    size_t estimateSampleCount(Scene *scene, RenderQueue *queue, const RenderJob *job,
                               int sceneResID, int sensorResID, int samplerResID,

                               // New attributes
                               ref<Timer> renderingTimer,  std::ofstream& timeFile,
                               Random* randomNumber, const Film *film) {
        if(m_config.timeout == 0) {
            Log(EError, "Impossible to estimate without timeout value");
            return 0;
        } else {
            Log(EInfo, "Run to know how many samples needed (without a lot of samples)");

            const size_t numberSampleCount = 4;
            ref<Timer> samplingTimer = new Timer;
            int currentIter = 0;
            int oldCountWU = m_config.workUnits;
            launchRendering(scene, queue, job, sceneResID, sensorResID, samplerResID,
                            renderingTimer, currentIter, timeFile,
                            false, randomNumber, film, numberSampleCount, 0, 0);
            m_config.workUnits = oldCountWU;

            // 30 % of offset
            return numberSampleCount * (m_config.timeout / samplingTimer->getSeconds()) * 1.3;
        }

    }

	bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
			int sceneResID, int sensorResID, int samplerResID) {

		ref<Sensor> sensor = scene->getSensor();
		ref<Sampler> sampler = sensor->getSampler();
		const Film *film = sensor->getFilm();
		size_t sampleCount = sampler->getSampleCount();
		m_config.importanceMap = NULL;

		// Create data to track down performances
		ref<Timer> renderingTimer = new Timer;
		std::string timeFilename = scene->getDestinationFile().string() + "_time.csv";
        std::ofstream timeFile;
        timeFile.open(timeFilename.c_str());
        ref<Random> randomNumbers = new Random;

#if 1
        // Automatically scale the number of samples
        // This is important for choosing the number of seed
        // and how many different seed we need
        if(m_config.timeout != 0) {
            // Use automatic estimation of samples
            size_t newSampleCount = estimateSampleCount(scene, queue, job, sceneResID, sensorResID, samplerResID,
                                    renderingTimer, timeFile,
                                    randomNumbers.get(), film);
            Log(EInfo, "Change the number of samples: " SIZE_T_FMT " (old: " SIZE_T_FMT ")", newSampleCount, sampleCount);
            sampleCount = newSampleCount;
            scene->getSampler()->setSampleCount(sampleCount);
        }
#endif

		if (m_config.stagedTechnique == ETwoStage && !m_config.firstStage) {
			Log(EInfo, "Executing first MLT stage");
			ref<Timer> timer = new Timer();
			Assert(m_config.firstStageSizeReduction > 0);
			m_config.importanceMap = BidirectionalUtils::mltLuminancePass(
					scene, sceneResID, queue, m_config.firstStageSizeReduction,
					m_nestedJob);

			if (!m_config.importanceMap) {
				Log(EWarn, "First-stage MLT process failed!");
				return false;
			}

			////////////////////////////////////////
      /// Compute Importance function to display it
      ////////////////////////////////////////
      // === Compute the importance function
      Vector2i size = m_config.importanceMap->getSize();
      ref<Bitmap> impMap = m_config.importanceMap->clone();
      const Float *luminanceValues = m_config.importanceMap->getFloatData();
      Float *importanceValues = impMap->getFloatData();
      for (int i = 0; i < impMap->getWidth(); ++i) {
            for (int j = 0; j < impMap->getHeight(); j++) {
              if(luminanceValues[i + j * size.x] > 0) {
                importanceValues[i + j * size.x] = 1.f / luminanceValues[i + j * size.x];
              } else {
                importanceValues[i + j * size.x] = 0.f;
              }
            }
      }

      std::stringstream ss;
      ss << scene->getDestinationFile().c_str() << "_imp";
      saveImportance(scene, ss.str());

			Log(EInfo, "First MLT stage took %i ms", timer->getMilliseconds());
		}

		bool nested = (m_config.stagedTechnique == ETwoStage) && m_config.firstStage;
		bool success = true;
		int currentPass = 0;

//    if (m_config.stagedTechnique != EMultiStage) {
//
//      success = launchRendering(scene, queue, job, sceneResID, sensorResID,
//                                samplerResID, renderingTimer, currentPass,
//                                timeFile, nested, randomNumbers.get(), film,
//                                sampleCount, accumBuffer, 0);
//    } else {

    size_t totalTimeout = m_config.timeout;  // Save the total timeout
    MLTAccumulBuffer* accumBuffer = new MLTAccumulBuffer(film->getCropSize());
    MLTMultiStage multiStage(film, sampleCount);
    if (m_config.stagedTechnique != EMultiStage) {
      if(m_config.stagedTechnique == ETwoStage && nested) {
        multiStage.setLinearCount(film, sampleCount, 1);
      } else {
        multiStage.setLinearCount(film, sampleCount, 10);
      }
    }
    size_t cumulatedBudget = 0;
    ref<Timer> renderingTimerTotal = new Timer;

    bool firstPass = true;
    Float budgetScale = 1.f;

    // Now do the rendering loop
    for (size_t i = 0; i < multiStage.budget.size() && success; i++) {
      // If no budget, go one step futher
      if (multiStage.budget[i] == 0) {
        continue;
      }

      // Change the timeout value
      // Some minimal values is required
      // However, for the first valid iteration, no time out is apply
      if (!firstPass) {
        m_config.timeout = (multiStage.budget[i] / (Float) sampleCount)
            * totalTimeout;
        if (m_config.timeout < 10 && totalTimeout != 0) {
          // In this case, we add a minimum of 10 sec
          // To be sure that we received enought samples
          m_config.timeout = 10;
          // This is to avoid no sampling for one iteration
          // Which can be really a problem
        }
        Log(EInfo, "Set timeout to: %i", m_config.timeout);
      }
      firstPass = false;

      accumBuffer->currWeight = multiStage.budget[i];
      m_config.workUnits = 0;  // In order to recompute the good number of workUnits
      // 1) Launch rendering
      success = launchRendering(scene, queue, job, sceneResID, sensorResID,
                                samplerResID, renderingTimer, currentPass,
                                timeFile, nested, randomNumbers.get(), film,
                                multiStage.budget[i], accumBuffer, i);
      if (!nested && (m_config.stagedTechnique == EMultiStage))
        nested = true;

      accumBuffer->next(multiStage.budget[i]);
      if (m_config.stagedTechnique == EMultiStage) {
        m_config.importanceMap = multiStage.updateLuminanceMap(film, i);
        std::stringstream ss;
        ss << scene->getDestinationFile().c_str() << "_imp_" << i;
        saveImportance(scene, ss.str());
      }

      if(!(m_config.stagedTechnique == ETwoStage && nested)) {
        cumulatedBudget += multiStage.budget[i];
        Float progressSampling = cumulatedBudget / (Float) sampleCount;
        Log(EInfo, "Cumulated samples budget: %f (%i / %i)", progressSampling,
            cumulatedBudget, sampleCount);

        int secPerPourcentage = renderingTimerTotal->getSeconds()
            / progressSampling;
        Log(EInfo, "Estimated rendering time: %i sec (Tot: %i sec)",
            (int )(secPerPourcentage * (1 - progressSampling)),
            secPerPourcentage);

        // Rules to scale up automatically the samples
        if (totalTimeout != 0) {
          // Need to scale up to be sure everything is Ok
          Float progressTimeout = renderingTimerTotal->getSeconds()
              / (Float) totalTimeout;
          budgetScale = (progressSampling / progressTimeout);

          Log(EInfo, "New budget scale is found: %f", budgetScale);

          sampleCount = multiStage.scaleBudget(budgetScale, i);
        }
      }
    }
    Log(EInfo, "Total time of rendering (multiStage): %i",
        renderingTimerTotal->getSeconds());
    delete accumBuffer;
//    }

    return success;


	}

	MTS_DECLARE_CLASS()
private:
	ref<ParallelProcess> m_process;
	ref<RenderJob> m_nestedJob;
	MLTConfiguration m_config;
};

MTS_IMPLEMENT_CLASS_S(MLT, false, Integrator)
MTS_EXPORT_PLUGIN(MLT, "Path Space Metropolis Light Transport");
MTS_NAMESPACE_END
