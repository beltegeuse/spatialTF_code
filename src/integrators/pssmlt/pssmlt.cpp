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
#include <mitsuba/core/plugin.h>
#include "pssmlt_proc.h"
#include "pssmlt_sampler.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{pssmlt}{Primary Sample Space Metropolis Light Transport}
 * \order{9}
 * \parameters{
 *	   \parameter{bidirectional}{\Boolean}{
 *	   PSSMLT works in conjunction with another rendering
 *	   technique that is endowed with Markov Chain-based sample generation.
 *	   Two choices are available (Default: \code{true}):
 *	    \begin{itemize}
 *	    \item \code{true}: Operate on top of a fully-fleged bidirectional
 *	      path tracer with multiple importance sampling.
 *	    \item \code{false}: Rely on a unidirectional
 *	    volumetric path tracer (i.e. \pluginref{volpath})
 *	    \vspace{-4mm}
 *	    \end{itemize}
 *	   }
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{directSamples}{\Integer}{
 *	       By default, this plugin renders the direct illumination component
 *	       separately using an optimized direct illumination sampling strategy
 *	       that uses low-discrepancy number sequences for superior performance
 *	       (in other words, it is \emph{not} rendered by PSSMLT). This
 *	       parameter specifies the number of samples allocated to that method. To
 *	       force PSSMLT to be responsible for the direct illumination
 *	       component as well, set this parameter to \code{-1}. \default{16}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
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
 *       See below for details. \default{{\footnotesize\code{false}}}}
 *	   \parameter{pLarge}{\Float}{
 *	     Rate at which the implementation tries to replace the current path
 *	     with a completely new one. Usually, there is little need to change
 *	     this. \default{0.3}
 *	   }
 * }
 * Primary Sample Space Metropolis Light Transport (PSSMLT) is a rendering
 * technique developed by Kelemen et al. \cite{Kelemen2002Simple} which is
 * based on Markov Chain Monte Carlo (MCMC) integration.
 * \renderings{
 *    \vspace{-2mm}
 *    \includegraphics[width=11cm]{images/integrator_pssmlt_sketch.pdf}\hfill\,
 *    \vspace{-3mm}
 *    \caption{PSSMLT piggybacks on a rendering method that can turn points
 *    in the primary sample space (i.e. ``random numbers'') into paths. By
 *    performing small jumps in primary sample space, it can explore the neighborhood
 *    of a path\vspace{-5mm}}
 * }
 * In contrast to simple methods like path tracing that render
 * images by performing a na\"ive and memoryless random search for light paths,
 * PSSMLT actively searches for \emph{relevant} light paths (as is the case
 * for other MCMC methods). Once such a path is found, the algorithm tries to
 * explore neighboring paths to amortize the cost of the search. This can
 * significantly improve the convergence rate of difficult input.
 * Scenes that were already relatively easy to render usually don't benefit
 * much from PSSMLT, since the MCMC data management causes additional
 * computational overheads.
 *
 * An interesting aspect of PSSMLT is that it performs this exploration
 * of light paths by perturbing the ``random numbers'' that were initially
 * used to construct the path. Subsequent regeneration of the path using the
 * perturbed numbers yields a new path in a slightly different configuration, and
 * this process repeats over and over again.
 * The path regeneration step is fairly general and this is what makes
 * the method powerful: in particular, it is possible to use PSSMLT as a
 * layer on top of an existing method to create a new ``metropolized''
 * version of the rendering algorithm that is enhanced with a certain
 * degree of adaptiveness as described earlier.
 *
 * The PSSMLT implementation in Mitsuba can operate on top of either a simple
 * unidirectional volumetric path tracer or a fully-fledged bidirectional path
 * tracer with  multiple importance sampling, and this choice is controlled by the
 * \code{bidirectional} flag. The unidirectional path tracer is generally
 * much faster, but it produces lower-quality samples. Depending on the input, either may be preferable.
 * \vspace{-7mm}
 * \paragraph{Caveats:}
 * There are a few general caveats about MLT-type algorithms that are good
 * to know. The first one is that they only render ``relative'' output images,
 * meaning that there is a missing scale factor that must be applied to
 * obtain proper scene radiance values. The implementation in Mitsuba relies
 * on an additional Monte Carlo estimator to recover this scale factor. By
 * default, it uses 100K samples (controlled by the \code{luminanceSamples}
 * parameter), which should be adequate for most applications.
 *
 * The second caveat is that the amount of computational expense
 * associated with a pixel in the output image is roughly proportional to
 * its intensity. This means that when a bright object (e.g. the sun) is
 * visible in a rendering, most resources are committed to rendering the
 * sun disk at the cost of increased variance everywhere else. Since this is
 * usually not desired, the \code{twoStage} parameter can be used to
 * enable \emph{Two-stage MLT} in this case.
 *
 * In this mode of operation, the renderer first creates a low-resolution
 * version of the output image to determine the approximate distribution of
 * luminance values. The second stage then performs the actual rendering, while
 * using the previously collected information to ensure that
 * the amount of time spent rendering each pixel is uniform.
 *
 * The third caveat is that, while PSMLT can work with scenes that are extremely
 * difficult for other methods to handle, it is not particularly efficient
 * when rendering simple things such as direct illumination (which is more easily
 * handled by a brute-force type algorithm). By default, the
 * implementation in Mitsuba therefore delegates this to such a method
 * (with the desired quality being controlled by the \code{directSamples} parameter).
 * In very rare cases when direct illumination paths are very difficult to find,
 * it is preferable to disable this separation so that PSSMLT is responsible
 * for everything. This can be accomplished by setting
 * \code{directSamples=-1}.
 */

class PSSMLT : public Integrator {
public:
	PSSMLT(const Properties &props) : Integrator(props) {
		//// DEV parameters
		m_config.useKelemenMutation = props.getBoolean("useKelemenMutation", true);
		m_config.useAMCMC = props.getBoolean("useAMCMC", false);

		if(m_config.useAMCMC && m_config.useKelemenMutation) {
		  // This configuration is not supported
		  Log(EError, "Impossible to use AMCMC and kelemen Mutation in the same time");
		}

		desiredSeedPerWorkUnit = props.getFloat("desiredSeed", 10.f);

		/* Note: a bunch of the parameters below are not publicly exposed,
		   because there is really little sense for most users to ever change them. */

		/* Longest visualized path length (<tt>-1</tt>=infinite).
		   A value of <tt>1</tt> will visualize only directly visible light
		   sources. <tt>2</tt> will lead to single-bounce (direct-only)
		   illumination, and so on. */
		m_config.maxDepth = props.getInteger("maxDepth", -1);

		/* Depth to begin using russian roulette (set to -1 to disable) */
		m_config.rrDepth = props.getInteger("rrDepth", 5);

		/* If set to <tt>true</tt>, the MLT algorithm runs on top of a
		   bidirectional path tracer with multiple importance sampling.
		   Otherwise, the implementation reverts to a basic path tracer.
		   Generally, the bidirectinal path tracer should be noticably
		   better, so it's best to this setting at its default. */
		m_config.technique = props.getBoolean("bidirectional", true) ?
			PathSampler::EBidirectional : PathSampler::EUnidirectional;

		// Special case if we want to compute only the missing component
		// Indeed, these energy can be computed using PSSMLT with specialized PT integrator.
		bool computeMissing = props.getBoolean("missingComp", false);
		if(computeMissing && m_config.technique == PathSampler::EBidirectional) {
		  // Impossible setup because missing only works with
		  // unidirectional path tracer
		  Log(EError, "Bidirectional and missing technique are incompatibles");
		}
		if(computeMissing) {
		  m_config.technique = PathSampler::EUnidirectionalMissing;
		  m_config.bounceRoughness = props.getFloat("bounceRoughness");
		}

		/* This setting can be very useful to reduce noise in dark regions
		   of the image: it activates two-stage MLT, where a nested MLT renderer
		   first creates a tiny version of the output image. In a second pass,
		   the full version is then rendered, while making use of information
		   about the image-space luminance distribution found in the first
		   pass. Two-stage MLT is very useful in making the noise characteristics
		   more uniform over time image -- specifically, since MLT tends to get
		   stuck in very bright regions at the cost of the remainder of the image.*/
		// Other variation are possible
		m_config.stagedTechnique = MLTStageTechnique(props.getString("stagedTechnique"));
		if(m_config.stagedTechnique == EUnknowStage) {
		  Log(EError, "Unknow staged technique");
		}

		/* When running two-stage MLT, this parameter determines the size
		   of the downsampled image created in the first pass (i.e. setting this
		   to 16 means that the horizontal/vertical resolution will be 16 times
		   lower). Usually, it's fine to leave this parameter unchanged. When
		   the two-stage process introduces noisy halos around very bright image
		   regions, it can be set to a lower value */
		m_config.firstStageSizeReduction = props.getInteger(
			"firstStageSizeReduction", 16);

		/* Used internally to let the nested rendering process of a
		   two-stage MLT approach know that it is running the first stage */
		m_config.firstStage= props.getBoolean("firstStage", false);

		/* Number of samples used to estimate the total luminance
		   received by the sensor's sensor */
		m_config.luminanceSamples = props.getInteger("luminanceSamples", 100000);

		/* Probability of creating large mutations in the [Kelemen et. al]
		   MLT variant. The default is 0.3. */
		m_config.pLarge = props.getFloat("pLarge", 0.3f);

		/* This parameter can be used to specify the samples per pixel used to
		   render the direct component. Should be a power of two (otherwise, it will
		   be rounded to the next one). When set to zero or less, the
		   direct illumination component will be hidden, which is useful
		   for analyzing the component rendered by MLT. When set to -1,
		   PSSMLT will handle direct illumination as well */
		m_config.directSamples = props.getInteger("directSamples", 16);
		m_config.separateDirect = m_config.directSamples >= 0;

		/* Should the multiple importance sampling-based weight computation by
		   Kelemen et al. be used? Otherwise, the implementation falls back
		   to the 'use of expectations' technique from Veach-style MLT. */
		m_config.kelemenStyleWeights = props.getBoolean("kelemenStyleWeights", true);

		/* Should an optimized direct illumination sampling strategy be used
		   for s=1 paths? (as opposed to plain emission sampling). Usually
		   a good idea. Note that this setting only applies when the
		   bidirectional path tracer is used internally. The optimization
		   affects all paths, not just the ones contributing direct illumination,
		   hence it is completely unrelated to the <tt>separateDirect</tt>
		   parameter. */
		m_config.directSampling = props.getBoolean(
				"directSampling", true);

		/* Recommended mutation sizes in primary sample space */
		m_config.mutationSizeLow  = props.getFloat("mutationSizeLow",  1.0f/1024.0f);
		m_config.mutationSizeHigh = props.getFloat("mutationSizeHigh", 1.0f/64.0f);
		Assert(m_config.mutationSizeLow > 0 && m_config.mutationSizeHigh > 0 &&
		       m_config.mutationSizeLow < 1 && m_config.mutationSizeHigh < 1 &&
			   m_config.mutationSizeLow < m_config.mutationSizeHigh);

		/* Specifies the number of parallel work units required for
		   multithreaded and network rendering. When set to <tt>-1</tt>, the
		   amount will default to four times the number of cores. Note that
		   every additional work unit entails a significant amount of
		   communication overhead (a full-sized floating put image must be
		   transmitted), hence it is important to set this value as low as
		   possible, while ensuring that there are enough units to keep all
		   workers busy. */
		m_config.workUnits = props.getInteger("workUnits", -1);

		/* Stop MLT after X seconds -- useful for equal-time comparisons */
		m_config.timeout = props.getInteger("timeout", 0);

		m_config.maxTimeImgDump = props.getInteger("maxTimeImgDump", INT_MAX);
		m_config.recomputeNormalisation = props.getBoolean("computeNormalisation", false);

		if(m_config.separateDirect && m_config.stagedTechnique == EMultiStage) {
		  Log(EError, "Impossible to separte the direct with multi-stage");
		}
	}

	/// Unserialize from a binary data stream
	PSSMLT(Stream *stream, InstanceManager *manager)
	 : Integrator(stream, manager) {
		m_config = PSSMLTConfiguration(stream);
		configure();
	}

	virtual ~PSSMLT() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		Integrator::serialize(stream, manager);
		m_config.serialize(stream);
	}

	bool preprocess(const Scene *scene, RenderQueue *queue,
			const RenderJob *job, int sceneResID, int sensorResID,
			int samplerResID) {
		Integrator::preprocess(scene, queue, job, sceneResID,
				sensorResID, samplerResID);

		ref<const Sensor> sensor = scene->getSensor();

		if (scene->getSubsurfaceIntegrators().size() > 0)
			Log(EError, "Subsurface integrators are not supported by MLT!");

		if (sensor->getSampler()->getClass()->getName() != "IndependentSampler")
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
	                     size_t sampleCount, MLTAccumulBuffer* accumBuff, int currentIteration) {
	  ref<Scheduler> scheduler = Scheduler::getInstance();
	  size_t nCores = scheduler->getCoreCount();

	  Vector2i cropSize = film->getCropSize();
      Assert(cropSize.x > 0 && cropSize.y > 0);
      Log(EInfo, "Starting %srender job (%ix%i, " SIZE_T_FMT
          " %s, " SSE_STR ", approx. " SIZE_T_FMT " mutations/pixel) ..",
          nested ? "nested " : "", cropSize.x, cropSize.y,
          nCores, nCores == 1 ? "core" : "cores", sampleCount);

      size_t desiredMutationsPerWorkUnit =
          m_config.technique == PathSampler::EBidirectional ? 100000 : 200000;

      if (m_config.workUnits <= 0) {
          const size_t cropArea  = (size_t) cropSize.x * cropSize.y;
          const size_t workUnits = ((desiredMutationsPerWorkUnit - 1) +
              (cropArea * sampleCount)) / desiredMutationsPerWorkUnit;
          Assert(workUnits <= (size_t) std::numeric_limits<int>::max());
          m_config.workUnits = (int) std::max(workUnits, (size_t) 1);
          Log(EInfo, "Number of workUnits updated: %i", m_config.workUnits);
      }

      size_t luminanceSamples = m_config.luminanceSamples;
      if (luminanceSamples < (size_t) m_config.workUnits * 10) {
          luminanceSamples = (size_t) m_config.workUnits * 10;
          Log(EWarn, "Warning: increasing number of luminance samples to " SIZE_T_FMT,
              luminanceSamples);
          m_config.luminanceSamples = luminanceSamples;
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

      std::vector<PathSeed> pathSeeds;
      ref<ReplayableSampler> rplSampler = new ReplayableSampler(false, currentIteration);
      ref<PathSampler> pathSampler = new PathSampler(m_config.technique, scene,
          rplSampler, rplSampler, rplSampler, m_config.maxDepth, m_config.rrDepth,
          m_config.separateDirect, m_config.directSampling, m_config.bounceRoughness);

      ref<PSSMLTProcess> process = new PSSMLTProcess(job, queue,
              m_config, directImage, pathSeeds, renderingTimer, scene,
              timeFile, currentPass, accumBuff);

      m_config.luminance = pathSampler->generateSeeds(luminanceSamples,
          m_config.workUnits, false, m_config.importanceMap, pathSeeds, desiredSeedPerWorkUnit);
      m_config.luminanceSamples = luminanceSamples; // Re put the modification

      if(accumBuff != NULL) {
        accumBuff->addNormalization(m_config.luminance, luminanceSamples);
        m_config.luminance = accumBuff->getNormalization();
      }

      if (!nested)
          m_config.dump();

      /* Create a sampler instance for each worker */
      ref<PSSMLTSampler> mltSampler = new PSSMLTSampler(m_config);
      mltSampler->setRandom(randomNumber); // Set the random number to be sure
      // to not redo the same sequence

      std::vector<SerializableObject *> mltSamplers(scheduler->getCoreCount());
      for (size_t i=0; i<mltSamplers.size(); ++i) {
          ref<Sampler> clonedSampler = mltSampler->clone();
          clonedSampler->incRef();
          mltSamplers[i] = clonedSampler.get();
      }
      int mltSamplerResID = scheduler->registerMultiResource(mltSamplers);
      for (size_t i=0; i<scheduler->getCoreCount(); ++i)
          mltSamplers[i]->decRef();
      int rplSamplerResID = scheduler->registerResource(rplSampler);

      process->bindResource("scene", sceneResID);
      process->bindResource("sensor", sensorResID);
      process->bindResource("sampler", mltSamplerResID);
      process->bindResource("rplSampler", rplSamplerResID);

      m_process = process;
      scheduler->schedule(process);
      scheduler->wait(process);
      m_process = NULL;
      scheduler->unregisterResource(rplSamplerResID); // Samplers destructions
      process->develop();

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
                            true, randomNumber, film, numberSampleCount, 0, 0);
            m_config.workUnits = oldCountWU; // To be sure to not change
            // 30 % of offset
            return numberSampleCount * (m_config.timeout / samplingTimer->getSeconds()) * 1.3;
        }

    }

	bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
			int sceneResID, int sensorResID, int samplerResID) {
		ref<Scheduler> scheduler = Scheduler::getInstance();
		ref<Sensor> sensor = scene->getSensor();
		ref<Sampler> sampler = sensor->getSampler();
		const Film *film = sensor->getFilm();
		size_t sampleCount = sampler->getSampleCount();
		m_config.importanceMap = NULL;

		// Create data to track down performances
		ref<Timer> renderingTimer = new Timer;
		int currentPass = 0;
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

			std::stringstream ss;
			ss << scene->getDestinationFile().c_str() << "_imp";
			saveImportance(scene, ss.str());

			Log(EInfo, "First MLT stage took %i ms", timer->getMilliseconds());
		}

		bool nested = (m_config.stagedTechnique == ETwoStage) && m_config.firstStage;
		bool success = true;

/*		if(m_config.stagedTechnique != EMultiStage) {
		  success = launchRendering(scene, queue, job, sceneResID, sensorResID, samplerResID,
		                            renderingTimer, currentPass, timeFile,
		                            nested, randomNumbers.get(), film, sampleCount, 0, 0);
		} else {*/
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
//		}

		return success;
	}

	MTS_DECLARE_CLASS()
private:
	ref<ParallelProcess> m_process;
	ref<RenderJob> m_nestedJob;
	PSSMLTConfiguration m_config;
	float desiredSeedPerWorkUnit;
};

MTS_IMPLEMENT_CLASS_S(PSSMLT, false, Integrator)
MTS_EXPORT_PLUGIN(PSSMLT, "Primary Sample Space MLT");
MTS_NAMESPACE_END
