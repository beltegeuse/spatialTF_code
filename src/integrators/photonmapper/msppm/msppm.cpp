// MTS includes
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/bidir/rsampler.h>

// Project include
#include "msppm.h"
#include "splatting.h"
#include "utils.h"
#include "msppm_path.h"
#include "../misWeights.h"
#include "../initializeRadius.h"
#include "../spatialTree.h"
#include "imp/importanceFuncBuilder.h"
#include "msppm_proc.h"
#include "../tools.h"

// OpenMP include
#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

template <>
bool PixelData<GatherPoint>::usePhiStatistics = true;
template <>
bool PixelData<GatherPoint>::limitMaxGatherPoints = true;

// TODO: Can be false in two cases:
//  - Not using 3 chains
//  - Use multiples chains but use only the highest one (not handled)
template <>
int PixelData<GatherPoint>::nbChains = 1;
template <>
int PixelData<GatherPoint>::maxGatherPoints = 10; // Can be set to 1 even though we will use more than 1 gather point
Float MISHelper::lightPathRatio = 1.f;

// This structure will be used to communicate with the rendering process
// Indeed at each time, the rendering process compute severals values
struct PhotonShootingRes {
  std::vector<Float> bi;  //< Normalization factor for the two chains
  //float ratioUni;  //< Ratio between uni/mutated path
  std::vector<size_t> nbEmittedPath;  //< Number of path emitted

  PhotonShootingRes(int nbChain) {
       bi.resize(nbChain, 0.f);
       nbEmittedPath.resize(nbChain, 0);
   }

   void dumpInfo() {
       SLog(EInfo, "Normalisation and Emitted paths: ");
       for(size_t i = 0; i < bi.size(); i++) {
           SLog(EInfo, " - Chain "SIZE_T_FMT": (bi: %f | emitted: "SIZE_T_FMT")", i, bi[i], nbEmittedPath[i]);
       }
   }
};

#define VERBOSE_INFO 0

class MSPPMIntegrator : public Integrator {
 public:
  MSPPMIntegrator(const Properties &props)
      : Integrator(props) {
    // === Photon Tracing configuration
    // maximum depth, russian roulette, ...
    m_config.maxDepth = props.getInteger("maxDepth", -1);
    m_config.rrDepth = props.getInteger("rrDepth", 3);

    // === Setup of SPPM
    // initial radii strategy, alpha, number of photons emitted
    // per passes
    m_config.initialScale = props.getFloat("initialScale", 1.f);
    m_config.alpha = props.getFloat("alpha", .7);
    m_config.photonCount = props.getInteger("photonCount", 250000);

    // === reference Mod:
    // if true, all the sampler (gather points and light path) are random
    // and different launch create completly different results (random seed)
    // if false, the gather point sequence will be the same
    // moreover, the light path sampling will approximatively the same
    // (some deviation are due to float points issues)
    m_config.referenceMod = props.getBoolean("referenceMod", false);

    // === Algorithm behavior
    // --- Use the expected values (Veach thesis)
    // or waste recycling
    m_config.useExpectedValue = props.getBoolean("useExpectedValue", true);
    // --- The different strategy to compute the statistic Mi
    m_config.numberStrat = getNumberStrategy(
        props.getString("numberStrategy", "Different"));

    // --- Get AStar value
    m_config.AStar = props.getFloat("AStar", 0.5);

    // --- Epsilon used to compute the importance function (InvSurf and LocalImp)
    m_config.epsilonInvSurf = props.getFloat("epsilonSurf", Epsilon);
    m_config.epsilonLocalImp = props.getFloat("epsilonDensity", Epsilon);

    // --- Remove contributions from chains
    // To avoid burning period
    m_config.cancelPhotons = props.getInteger("cancelPhotons", 10000);

    // === Frequency and time configuration and algorithm behavior
    // max rendering passes, step to dump data, ...
    m_maxPass = props.getInteger("maxPass", INT_MAX);
    m_stepSnapshot = props.getInteger("stepSnapshot", INT_MAX);
    m_stepDensitySnapshot = props.getInteger("stepDensity", 10);
    m_dumpAtEachIterationRadii = false;
    m_dumpExtraImportanceInfo = false;
    // --- Compute the direct contribution
    m_computeDirect = props.getBoolean("computeDirect", true);
    // --- Dump an special image to show the contribution of the pass
    m_dumpImagePass = props.getBoolean("dumpImagePass", false);
    // --- Show the importance function
    m_showImpFunction = props.getBoolean("showImpFunc", true);

    // === Chains configuration
    // --- Number of chains
    // WARNING: In this implementation the uniform and visibility chain
    // is managed by a same chain. So for our approach with 4 chains,
    // this parameter will equal to 3
    m_config.numberChains = props.getInteger("numberChains");
    PixelData<GatherPoint>::nbChains = m_config.numberChains;
    // --- Use MIS to blend the different levels
    m_config.useMISLevel = props.getBoolean("useMIS");

    //
    m_config.REAllTime = props.getBoolean("REAllTime", true);
    // --- Use all samples to computes the normalisation factor
    m_config.strongNormalisation = props.getBoolean("strongNormalisation", true);
    // --- Use corrected form for MIS weights
    m_config.correctMIS = props.getBoolean("correctMIS", true);
    // --- Use power heuristic for MIS weightd
    m_config.usePowerHeuristic = props.getBoolean("usePowerHeuristic", false);
    // --- Use MIS to compute the psi statistic for our TF
    // The aim is to use visibility and inv. radii samples.
    m_config.useMISUniqueCount = props.getBoolean("useMISUniqueCount", true);
    // --- Make correction of previous normalisation factor
    // by cancel out the difference of the multiplication factors
    // made at each iteration
    m_config.rescaleLastNorm = props.getBoolean("rescaleLastNorm", true);


    // The different way to compute phi statistic
    int defaultPhiStrategy = 0;
    if(PixelData<GatherPoint>::nbChains == 2) {
      defaultPhiStrategy = 0; // Use second level to estimate the phi statistics
    } else if(PixelData<GatherPoint>::nbChains == 3) {
      defaultPhiStrategy = 2; // Use the two intermediate levels to estimate phi statistics
      // blend them using MIS (power heuristic)
    } else if(PixelData<GatherPoint>::nbChains == 1) {
      defaultPhiStrategy = 0; // See what's happens
    }
    m_config.phiStatisticStrategy = props.getInteger("phiStatisticStrategy", defaultPhiStrategy); // Default one

    // === Techniques (VCM, SPPM..)
    // Note that VCM can is broken in the current code version
    std::string technique = props.getString("technique","sppm");
    MISHelper::parse(technique, m_config.usedTechniques, m_config.multiGatherPoints);

    m_config.removeDeltaPaths = props.getBoolean("removeDeltaPaths",true);
    // --- The tree type used to compute the spatial clustering
    m_config.treeType = getTreeType(props.getString("treeType","median"));

    // === Error checking
    if (m_config.maxDepth <= 1 && m_config.maxDepth != -1) {
      Log(EError, "Maximum depth must be set to \"2\" or higher!");
    }
    if (m_config.initialScale == 0.f) {
      Log(EError, "Not possible to have no initial scale");
    }

    // Psi handling cases
    if(m_config.numberChains == 1 && m_config.phiStatisticStrategy != 0) {
      Log(EError, "Impossible Phi strategy");
    }
    if(m_config.numberChains == 3 &&
        (m_config.phiStatisticStrategy != 0 && m_config.phiStatisticStrategy != 1 && m_config.phiStatisticStrategy != 2)) {
      Log(EError, "Impossible Phi strategy (4 chains)");
    }

    if(m_config.numberChains == 1) {
      m_config.useMISLevel = false; //< Impossible to make MIS
    }

    // === Internal data instantation and initialization
    m_gpManager = new RadiusInitializer(props);  //< The GP generator (radii, position ... etc.)
    m_impFunc = getImpFunc(props);  // < The importance function used

    // Reduce memory consomption
    if (m_config.usedTechniques & MISHelper::SPPM_ONLY) {
      PixelData<GatherPoint>::maxGatherPoints = 1;
    }

    ///////////////////
    // Debug
    ///////////////////
    m_nameRef = props.getString("ref", "");

    // === Other values
    m_running = false;

    m_precomputedIMP = props.getString("precomputedImp", "");
    m_needSaveImp = false;
    m_stepSaveImp = 5;

    // --- Debug options
    m_config.showUpperLevels = props.getBoolean("showUpperLevels", false);

  }

  MSPPMIntegrator(Stream *stream, InstanceManager *manager)
      : Integrator(stream, manager) {
    Log(EError, "Network rendering is not supported!");
  }

  virtual ~MSPPMIntegrator() {
    delete m_impFunc;
    delete m_gpManager;
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    Integrator::serialize(stream, manager);
    Log(EError, "Network rendering is not supported!");
  }

  void cancel() {
    m_running = false;
  }

  /**
   * Preprocess procedure
   * Noting to do.
   */
  bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
                  int sceneResID, int sensorResID, int samplerResID) {
    m_running = true;  // < Launch the computation !

    // Call the father precomputation
    Integrator::preprocess(scene, queue, job, sceneResID, sensorResID,
                           samplerResID);

    // Reinit some values
    m_config.dump();
    m_totalEmittedPath = 0;
    return true;
  }

  /**
   * Main loop of the algorithm
   */
  bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
              int sceneResID, int sensorResID, int unused) {
    // Force to use correct emitter weight
    scene->recomputeWeightEmitterFlux();

    if (m_nameRef != "") {
      // Root directory
      std::string parentDir = scene->getSourceFile().parent_path().string();
      // === Read density reference image
      std::string refFilepath = parentDir + "/" + m_nameRef;
      m_refBitmap = new Bitmap(refFilepath, "");
    }

    //////////////////////////////////////////////////
    // Step 0: Preparation of the data
    //////////////////////////////////////////////////

    // Get all data associated to the scene
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = scene->getSensor();
    ref<Film> film = sensor->getFilm();
    size_t nCores = sched->getCoreCount();
    Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
        film->getCropSize().x, film->getCropSize().y, nCores,
        nCores == 1 ? "core" : "cores");

    // Get size of the rendering
    Vector2i cropSize = film->getCropSize();
    Point2i cropOffset = film->getCropOffset();
    int blockSize = scene->getBlockSize();
    MISHelper::lightPathRatio = m_config.photonCount / (Float)(cropSize.x * cropSize.y);

    // Reinitialize value for the rendering
    // (in order to be able to relaunch the algorithm)
    m_gatherBlocks.clear();  //< clear the list
    m_running = true;
    m_totalEmittedPath = 0;

    // Prepare bitmap used to display the rendering
    // results
    m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
    m_bitmap->clear();

    // Allocated and initialize the pixels
    // and the associated gather points
    // special list "m_gatherBlocks" and "m_offset"
    // is used in order to be able  to process the data in parallel
    int totalPixels = 0;
    PixelData<GatherPoint>::usePhiStatistics = m_impFunc->needPhiStatistic();

	// Prealloc for gather points
	size_t size = cropSize.x * cropSize.y * GatherPointsList::allocTempSize((int)nCores);
	char * gatherPointsData = new char[size];
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
	assert(ptr == gatherPointsData + size);

    // Initialize the class used to generate the gather points
    // and initialize the associated radii
    m_gpManager->init(scene, m_config.maxDepth, m_gatherBlocks, m_offset, 
		m_config.usedTechniques);

    // Creation of some output file to track the performance of the algorithm
    // Remark: I put this here because I want to include precomputation time ...
    std::string timeFilename = scene->getDestinationFile().string()
        + "_time.csv";
    std::ofstream timeFile(timeFilename.c_str());
#if VERBOSE_INFO
    std::string normFilename = scene->getDestinationFile().string()
            + "_norm.csv";
    std::ofstream normFile(normFilename.c_str());
    std::string normRescaleFilename = scene->getDestinationFile().string()
                + "_normRescale.csv";
    std::ofstream normRescaleFile(normRescaleFilename.c_str());
#endif

    ref<Timer> renderingTimer = new Timer;

    // Initialize and precompute the importance function
    // This is done outside the rendering loop because importance function
    // can be computated in a lot of different ways
    // === Always highest level
    m_impFunc->setImportanceID(m_config.numberChains-1);
    m_impFunc->initializeObject(m_config, m_gpManager, &m_gatherBlocks,
                                &m_offset);  //< Initialize internal attributes
    m_impFunc->precompute(scene, queue, job, sceneResID, sensorResID);  //< Precompute the importance function (if necessary)
    m_impFuncRessourceID = sched->registerResource(m_impFunc);

    // Reset all data associated to GP
    // Because some GP can be used and changed in the importance function precompute process
    // the importance function
    resetGatherPoints();

    // Create the metropolis sampler (for each cores)
    // This sampler is a Kelemen sampler with a different
    // mutation function (As same as hachisuka)
    // Moreover, in this sampler that Adaptive MCMC is done
    ref<Sampler> rplSampler = new MSPPMSampler(m_config);
    // Create this independent sampler for bidirectional connections (to determine which pixel to connect to)
    ref<Sampler> samplerIndependent =
		static_cast<Sampler *>(PluginManager::getInstance()->createObject(
		MTS_CLASS(Sampler),Properties(("independent"))));


    std::vector<SerializableObject *> MCDataList(nCores);
    std::vector<SerializableObject *> samplersIndependent(nCores);
    for (size_t i = 0; i < nCores; ++i) {
      // === Create MSPPM sampler
      // and wrap it inside MCData structure
      std::vector<MSPPMSampler*> clonedSamplers;
      clonedSamplers.reserve(PixelData<GatherPoint>::nbChains);
      for(int idChains = 0; idChains < PixelData<GatherPoint>::nbChains; idChains++) {
          ref<Sampler> cloned = rplSampler->clone();
          cloned->incRef(); // Avoid destruction
          clonedSamplers.push_back((MSPPMSampler*)cloned.get());
      }
      MCDataList[i] = new MCData(clonedSamplers);
      MCDataList[i]->incRef();

	  // Independent sampler
	  ref<Sampler> clonedIndepSampler = samplerIndependent->clone();
	  clonedIndepSampler->incRef();
	  samplersIndependent[i] = clonedIndepSampler.get();
    }
    int MCDataResID = sched->registerMultiResource(MCDataList);

    // Other data initialized
    int currentIteration = 0;

    ///////////////////////////////////
    ///////////////////////////////////
    // Main rendering loop
    ///////////////////////////////////
    ///////////////////////////////////
    std::vector<Float> lastNormalization(PixelData<GatherPoint>::nbChains, 1.f);
    std::vector<Float> lastScale(PixelData<GatherPoint>::nbChains, 1.f);
    std::vector<Float> newScale(PixelData<GatherPoint>::nbChains, 1.f);


    Float currentMSE = 0.f;
    while (m_running && currentIteration < m_maxPass) {
      ++currentIteration;  // This is the next iteration

      ///////////////////////////////////
      // Step 1: Generation gp and imp. association
      ///////////////////////////////////
      // G = GenerateGP() && build the GP map
      // arguments is to know if the radii need to be
      // recomputed
      Log(EInfo, "Rengenerate GP positions");
      m_gpManager->regeneratePositionAndRadius();

      // Now, we know the radius scaled attached to the gatherpoints
      // GP flux are normalized for now
      // So, we need to rescale it by the area of the gatherpoint
      // to correctly accumulate the flux
      m_gpManager->rescaleFlux();
      // Now, we have valid gatherpoint
      // So we push all of them in the accel structure
      // to be able to splat the light paths on them
      Log(EInfo, "Build the gather map... pass: %i", currentIteration);
      ref<GatherPointMap> gatherMap = new GatherPointMap(m_gatherBlocks,
														 scene,
														 &samplersIndependent,
                             m_config,
                             PixelData<GatherPoint>::nbChains);

      gatherMap->setNormalization(lastNormalization);
      int gatherMapID = sched->registerResource(gatherMap);

      // Special case in the first iteration
      // We dump the radius of the gather point to see
      // if everythings is OK.
      // Moreover, we initialise the spatial tree which is used to track severals statistics
      if (currentIteration == 1) {

        LocalImpTree * tree = NULL;
        if ( m_config.treeType == ETTSAH ) {
          tree = &(gatherMap->getAccel());
          //SLog(EError, "Unsupported tree option: SAH Tree");
        }
        else if ( m_config.treeType == ETTMedian ) {
          Float scaleRadii = 0.25;
          tree = new MedianTree<GatherPoint>(m_gatherBlocks, false, scaleRadii);
          Log(EInfo, "Generate additional GP position");

          for(int i = 0; i < 3; i++) {
              m_gpManager->regeneratePositionAndRadius();
              static_cast<MedianTree<GatherPoint>*>(tree)->expendGP(m_gatherBlocks, 0.25f);
          } // Generate more GP

          // Rebuild the gathermap from the last iteration
          Log(EInfo, "Build the gather map... pass: %i", currentIteration);
          gatherMap = new GatherPointMap(m_gatherBlocks,
                                     scene,
                                     &samplersIndependent,
                                     m_config,
                                     PixelData<GatherPoint>::nbChains);
          gatherMap->setNormalization(lastNormalization);
          gatherMapID = sched->registerResource(gatherMap);

          // Build median KD tree
          static_cast<MedianTree<GatherPoint>*>(tree)->forceBuild();

        } else {
          SLog(EError, "Unknown tree type!");
        }

        std::string GPSFilename = scene->getDestinationFile().string()
            + "_gps.bin";
        writeGPFile(GPSFilename);

        // === Initialize spatial tree to get the statistics
        // if we have localImp
        if (m_impFunc->getName() == "localImp") {
          if(m_precomputedIMP == "") {
          dynamic_cast<LocalImp*>(m_impFunc)->buildHierachy(scene, nCores,
                                                            *tree);
          } else {
            dynamic_cast<LocalImp*>(m_impFunc)->loadData( scene->getDestinationFile().parent_path().string()
                                                          + "/" + m_precomputedIMP);
          }
        }

        if ( m_config.treeType == ETTMedian ) {
          delete tree;
        }
      }



      // Update the importance function and normalize it if necessary
      m_impFunc->update(currentIteration - 1, m_totalEmittedPath);
      if(m_impFunc->getName() == "localImp") {
        if(m_config.numberChains == 2) {
          m_impFunc->setInvSurfaceImportanceGP(0);
        } else if(m_config.numberChains == 3) {
          m_impFunc->setConstantImportanceGP(0);
          m_impFunc->setInvSurfaceImportanceGP(1);
        }
      } else {
        if(m_config.numberChains != 1) {
          Log(EError, "Not possible to use 3 chain approach with other importance function than LocalImp");
        }
      }

        // We normalize the importance function
        // by scale it by an factor
        for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
          newScale[idChain] = nomarlizeImp(idChain);
        }

        if(m_config.rescaleLastNorm) {
          for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
            if(currentIteration != 1) {
              double factor =  lastScale[idChain] / newScale[idChain];
              lastNormalization[idChain] *= factor;
              Log(EInfo, "Rescale last normalisation: %f", factor);
            }

            lastScale[idChain] = newScale[idChain];
          }
          gatherMap->setNormalization(lastNormalization);
        }

        // Remark: Normally is not needed for all the importance function
        // However, large values in the importance function can provoke precision issues
        // In order to be sure to not meet this issue
        // normalisation of the importance is done.
        for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
          computeImportanceForGPMIS(gatherMap, idChain);
        }


      if (m_showImpFunction) {
        writeImportanceInfo(currentIteration, scene);
        writeImportance(currentIteration, scene, m_config.numberChains-1); // Write imp highest level
      }


      ///////////////////////////////////
      // Step 2: Rendering
      ///////////////////////////////////
      // Special case in the first iteration
      // PAPER: InitializeMC()
      // This is to initialise the markov chain
      // Other iterations, the markov chain is reinitialized
      // in the light paths shooting process.
      if (currentIteration == 1 && m_config.usedTechniques & MISHelper::LIGHTPATHS) {
        for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
          initializePhotonPaths(scene, nCores, gatherMap, totalPixels, MCDataList, idChain);
         }
      }

      // PAPER: ShootPhoton()
      Log(EInfo, "Performing a shooting photon %i", currentIteration);
      PhotonShootingRes resShoot = shootPhoton(currentIteration,
                                               job, sceneResID, sensorResID, gatherMapID,
                                               MCDataResID, nCores);
      resShoot.dumpInfo();
	  
      ///////////////////////////////////
      // Step 3: Udpate statistics
      ///////////////////////////////////
      // UpdatePixelsStatistics()
      UpdatePixelsStatistics(currentIteration, resShoot, film, queue, job,
                             nCores, scene, lastNormalization);
      for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
        lastNormalization[idChain] = 1.f / resShoot.bi[idChain];
      }

#if VERBOSE_INFO
      for(int idChain = 0; idChain < PixelData<GatherPoint>::nbChains; idChain++) {
        normFile << resShoot.bi[idChain] << ",";
        normRescaleFile << resShoot.bi[idChain] / newScale[idChain] << ",";
      }
      normFile << "\n";
      normFile.flush();
      normRescaleFile << "\n";
      normRescaleFile.flush();
#endif

      ///////////////////////////////////
      // Step 4: Heavy Debugging stage
      ///////////////////////////////////
      // Write data and some statistics
      writeSnapshot(currentIteration, scene);

#if VERBOSE_INFO
      if(currentIteration % m_stepSaveImp == 0 && m_needSaveImp && m_impFunc->getName() == "localImp") {
        std::stringstream ssImp;
        ssImp << scene->getDestinationFile().c_str() << "_impbin_pass_" << currentIteration << ".bin";
        dynamic_cast<LocalImp*>(m_impFunc)->saveData(ssImp.str());
      }
#endif

      // Log the time for the current pass
      // In order to plot RMSE curves over the rendering time
      unsigned int milliseconds = renderingTimer->getMilliseconds();
      timeFile << (milliseconds / 1000.f) << ",\n";
      timeFile.flush();
      Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
          milliseconds % 1000);

      // Free the memory and reset the timer
      // to compute the time of one rendering iteration.
      renderingTimer->reset();
      sched->unregisterResource(gatherMapID);

      /////////////
      // Debug
      /////////////
      if (m_refBitmap) {
        Float oldMSE = currentMSE;
        currentMSE = getMSE(/*currentIteration == 1*/true);
        Log(EInfo, "MSE: %f", currentMSE);
        if (currentIteration > 1) {
          Float diffMSE = currentMSE - oldMSE;
          Float pourcent = (diffMSE / oldMSE) * 100;
          Log(EInfo, "MSE change %f", diffMSE);
          // === Warning
          if (diffMSE > 0.f) {
            Log(EWarn, "RMSE increases !!! by: %f (pourcent: %f)", diffMSE,
                pourcent);
          }
        }
      }
    }

    ///////////////////////////////////
    // Free memory
    ///////////////////////////////////

    // --- Sampler rpl
    for (size_t i = 0; i < nCores; ++i) {
      MCDataList[i]->decRef();
	  samplersIndependent[i]->decRef();
	}
    sched->unregisterResource(MCDataResID);
	delete [] gatherPointsData;

    return true;
  }

  void reinitAllAMCMC(std::vector<SerializableObject *>& mcdatas) {
    for(size_t i = 0; i < mcdatas.size(); i++) {
      dynamic_cast<MCData*>(mcdatas[i])->reinitAMCMCSamplers();
    }
  }


  void computeImportanceForGPMIS(GatherPointMap* gatherMap, int idImportance) {
#if defined(MTS_OPENMP)
  #pragma omp parallel for
#endif
	  for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
		  GatherBlock &gatherBlock = m_gatherBlocks[blockIdx];
		  for (size_t i = 0; i < gatherBlock.size(); ++i) {
			  GatherPointsList &gps = gatherBlock[i];
			  Float importance = 0.f;
			  for (GatherPointsList::iterator it = gps.begin(); it != gps.end(); ++it ) {
				  if (it->depth != -1 && it->its.isValid()) {
					  ImportanceRes imp = gatherMap->queryGPImpactedImportance(it->its, it->depth, idImportance);
					  it->misInfo.m_importance[idImportance] = std::max(imp.importances[idImportance], importance);
					  importance = imp.importances[idImportance];
				  }
				  else
					  it->misInfo.m_importance[idImportance] = importance;
			  }
		  }
	  }
  }

  /**
   * Method to initialize the Markov chains.
   * Markov chains are described with two structure:
   *  - MSPPMSampler: The random generator ... etc.
   *  - PhotonPaths: The structure to store the light path of the current and proposed states
   *
   *  For now, this function is only call at the first iteration and have similarity with PSSMLT seed initialization.
   */
  void initializePhotonPaths(Scene* scene, size_t nCores,
                             GatherPointMap* gatherMap,
                             size_t totalPixels,
                             std::vector<SerializableObject*>& MCDataLists,
                             int idImportance) {

    // === Create objects
    Log(EInfo, "Seed generation & selection ...");
    // --- Sampler to generate random number and reselect an seed
    ref<ReplayableSampler> rplSampler = new ReplayableSampler(
        m_config.referenceMod, idImportance);
    // --- The path builder which use rplSampler to generate light paths
    ref<PhotonPathBuilder> pathBuilder = new PhotonPathBuilder(
        scene, m_config.maxDepth, m_config.rrDepth, rplSampler, 0, gatherMap);

    // === Generate 100 k light path and select nCores of them
    // The selection is proportional to the light path importance
    std::vector<SeedNormalisationPhoton> seeds;
    pathBuilder->generateSeeds(100000, nCores, seeds, idImportance);
    // Show the selection
    for (size_t i = 0; i < nCores; i++) {
      Log(EInfo, " - Selected Seed: (%f, " SIZE_T_FMT ")", seeds[i].importance,
          seeds[i].sampleIndex);
    }

    // === Initialize MSPPM Sampler and PhotonPaths
    // to be able to use them in the rest of the rendering
    // This is done for each selected seed
    Log(EInfo, "Regeneration path");
    for (size_t i = 0; i < nCores; i++) {
      // --- Instance new objects
      ref<PhotonPaths> path = new PhotonPaths(PixelData<GatherPoint>::nbChains);
      ref<ReplayableSampler> tempRpl =
          static_cast<ReplayableSampler*>(rplSampler->clone().get());
      tempRpl->incRef();  // < Pour etre sur ...
      MCData* mcData = static_cast<MCData*>(MCDataLists[i]);
      MSPPMSampler* samplerCore = mcData->getSampler(idImportance);
      ref<Random> oldSampler = samplerCore->getRandom();  // Save the created random sampler

      // --- Initial configuration sampler
      samplerCore->setLargeStep(true);
      tempRpl->setSampleIndex(seeds[i].sampleIndex);  //< Go to the good sample index by regenerating random number
      samplerCore->setRandom(tempRpl->getRandom());  //< TODO

      // --- Generate the current path of PhotonPaths
      // by using MSPPMSampler
      pathBuilder->setSampler(samplerCore);
      pathBuilder->samplePaths(*path->current, idImportance);

      // --- Error checking
      // We check if the path generated have the same importance as the seed selection step
      if (path->current->getImp(idImportance) != seeds[i].importance) {
        Log(EError, "Found inconstancies ... (%f != %f)", seeds[i].importance,
            path->current->getImp(idImportance));
      } else {
        Log(EInfo, " * Path is regenerated :)");
        mcData->setPath(idImportance, path);
        samplerCore->accept();
        path->incRef();
      }

      // Recover the created random sampler.
      // Indeed, it's important to have an new random sampler
      // In case that an initial state is selected twice.
      // To avoid similar chain (with not reference mode)
      samplerCore->setRandom(oldSampler.get());
    }

  }

  /**
   * Method to shoot photon and update the gather point
   * statistic. This method compute also the normalization factor
   * by using the sampling information from the uniform strategy.
   */
  PhotonShootingRes shootPhoton(int currentIteration,
                                const RenderJob *job, int sceneResID, int sensorResID,
                                int gatherMapID, int MCDataResID, size_t nCores) {
	if ( !(m_config.usedTechniques & MISHelper::LIGHTPATHS) ) {
	  PhotonShootingRes res(PixelData<GatherPoint>::nbChains);
    SLog(EError, "Not implemented: Check res initial value");
    return res;
	}
    ref<Scheduler> sched = Scheduler::getInstance();

    // Choose the granularity in the shooting
    // process
    size_t granularity = 0;
    if (!m_config.referenceMod) {
      granularity = std::max(
          (size_t) 1,
          m_config.photonCount / (Scheduler::getInstance()->getWorkerCount()));
    }

    // === Generate the proc responsible to shoot the photon
    // in the scene
    ref<SplattingMSPPMPhotonProcess> proc = new SplattingMSPPMPhotonProcess(
        m_config.photonCount, granularity, m_config, job);

    // --- Initialize it by given him
    // all the information
    proc->bindResource("scene", sceneResID);  //< 3D scene
    proc->bindResource("sensor", sensorResID);  //< Sensor (camera)
    proc->bindResource("gathermap", gatherMapID);  //< GP map (to splat photon)
    proc->bindResource("mcdata", MCDataResID);  //< Markov chain structure
    proc->bindResource("impFunc", m_impFuncRessourceID);

    // --- Launch and wait the computation
    sched->schedule(proc);
    sched->wait(proc);

    // --- Total emitted paths

    if(m_config.showUpperLevels) {
      m_totalEmittedPath += proc->getNbEmittedPath(PixelData<GatherPoint>::nbChains-1);
    } else {
      // Always take the first level
      m_totalEmittedPath += proc->getNbEmittedPath(0);

      if(!m_config.useMISLevel) {
        // If we not using MIS,
        // We need to add other level
        // To compute the final number of emitted path
        for(int i =1; i < PixelData<GatherPoint>::nbChains; i++) {
          // Begin at 1 (0 level is already added)
          m_totalEmittedPath += proc->getNbEmittedPath(i);
        }
      }
    }
    // === Create the results object
    // To be able to use it in the update pixel procedure
    PhotonShootingRes res(PixelData<GatherPoint>::nbChains);
    // Fill the structure with all info
    for(int i = 0; i < PixelData<GatherPoint>::nbChains; i++) {
        res.bi[i] = proc->getNormalisation(i);
        res.nbEmittedPath[i] = proc->getNbEmittedPath(i);
    }

    // Rescale normalisation factors
    // With depending of how the normalisation factor
    // are computed
    // FIXME: 4CHAINS
    if(m_config.numberChains == 1) {
    } else if(m_config.numberChains == 2) {
      res.bi[1] *= res.bi[0];
    } else if(m_config.numberChains == 3) {
      if(m_config.strongNormalisation) {
        res.bi[1] *= res.bi[0];
        res.bi[2] *= res.bi[1];
      } else {
        //res.bi[1] *= res.bi[0]; // FIXME: 4Chains
        res.bi[2] *= res.bi[1];
      }
    }


    // --- Info number uniform counts
    static size_t uniformCount = 0;
    static size_t uniformContributingCount = 0;

    uniformCount += proc->getNbUniformEmitted(0);
    uniformContributingCount += proc->getContributingUniform();
    SLog(EInfo, "Percentage of contributing uniform paths: %f %%",uniformContributingCount * 100.0f / (Float)uniformCount);

    return res;
  }

  /**
   * After photon shooting, we need to collect and update pixel statistics
   */
  void UpdatePixelsStatistics(int currentIteration, PhotonShootingRes resShoot,
                              Film *film, RenderQueue *queue,
                              const RenderJob *job, size_t nCores, Scene* scene,
                              std::vector<Float> normalizationMIS) {

    bool dumpAllInfo = m_dumpImagePass
        && (((currentIteration-1)  % m_stepSnapshot) == 0);

    film->clear();

    // --- Clone the display film
    // The two above bitmap will be using for debugging propose:
    //   - passContribBitmap: bitmap with the current iteration contribution only
    //   - passContribBitmapMi: bitmap with the Mi statistic (number of contributions)
    ref_vector<Bitmap> passContribs;
    for (int i = 0; i < PixelData<GatherPoint>::nbChains; i++) {
      passContribs.push_back(m_bitmap->clone());
    }
    ref<Bitmap> passContribBitmapMi = m_bitmap->clone();

    // === Loop on all the gather points
    // the aim this to compute the final image
    // TODO: For now the OpenMP implementation is off because localImp is NOT thread safe ! Need to fix it

    // If wwe are in MIS computation
    // Need to correct the number of samples
    Float scalingFlux = 1.f;
    if (m_config.numberChains >= 2) {
      scalingFlux = resShoot.nbEmittedPath[0]
          / (double) resShoot.nbEmittedPath[1];
      SLog(EInfo, "Scaling Flux by factor: %f", scalingFlux);
    }
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
      GatherBlock &gatherBlock = m_gatherBlocks[blockIdx];
      Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
      Spectrum *targetPassContribMi = (Spectrum *) passContribBitmapMi
          ->getUInt8Data();
      for (size_t i = 0; i < gatherBlock.size(); ++i) {
        GatherPointsList &gps = gatherBlock[i];
        // Iterate through all gather points corresponding to pixel "i" in gather block "blockIdx"
        Spectrum contribPixel(0.f);
        std::vector<Spectrum> contribPassOnly(PixelData<GatherPoint>::nbChains);
        std::vector<Spectrum> fluxChains(PixelData<GatherPoint>::nbChains);
        Float MTotal = 0.f;
        Float maxImportance = 0.f;

        for (int idChain = 0; idChain < PixelData<GatherPoint>::nbChains;
            idChain++) {
          fluxChains[idChain] = gps.getFlux(idChain);
          fluxChains[idChain] *= resShoot.bi[idChain];
        }

        if (m_config.useMISLevel && !m_config.showUpperLevels) {
          if (m_config.numberChains == 1) {
            Log(EError, "Invalid chain scaling flux: Only one chain");
          } else if (m_config.numberChains == 2) {
            fluxChains[1] *= scalingFlux;
          } else if (m_config.numberChains == 3) {
            fluxChains[1] *= scalingFlux;
            fluxChains[2] *= (resShoot.nbEmittedPath[0]
                / (double) resShoot.nbEmittedPath[2]);
          } else {
            Log(EError, "Invalid N chain scaling flux");
          }
        }

        for (GatherPointsList::iterator it = gps.begin(); it != gps.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            /////////////////////////////////////////
            // Treatement for one GP
            /////////////////////////////////////////
            GatherPoint & gp = *it;
            Float M = 0, N = gps.N;

            ///////////////////
            // Step 1: Collect all temporal data
            // collected by the gather point to this iteration
            // all data in duplicated for the number of cores to avoid
            // mutex operation
            ///////////////////

            /*
             * Temporal data:
             *  - tempFlux: Flux accumulated by this gp
             *  - M: Number of path impacted this gp
             *  - Phi: Number of contributive uniform paths
             */

            // Only if the gather point is valid
            for (size_t idThread = 0; idThread < nCores; idThread++) {
              size_t index = idThread + gp.tempIndex * nCores;
              M += (Float) gps.tempM[index];
            }

            Float PhiOdd = 0.f;
            Float PhiEven = 0.f;
            Float nbSamples = 0;

            if (m_impFunc->needPhiStatistic()) {
              if (m_config.phiStatisticStrategy == 0) {
                PhiOdd = gps.getPhi(0, gp.tempIndex, true) * resShoot.bi[0];
                PhiEven = gps.getPhi(0, gp.tempIndex, false) * resShoot.bi[0];

                nbSamples = gps.getNSamplePhi(0, gp.tempIndex);
              } else if (m_config.phiStatisticStrategy == 1) {
                PhiOdd = gps.getPhi(1, gp.tempIndex, true) * resShoot.bi[1];
                PhiEven = gps.getPhi(1, gp.tempIndex, false) * resShoot.bi[1];

                nbSamples = gps.getNSamplePhi(1, gp.tempIndex);
              } else if (m_config.phiStatisticStrategy == 2) {
                PhiOdd = gps.getPhi(0, gp.tempIndex, true) * resShoot.bi[0]
                    + gps.getPhi(1, gp.tempIndex, true) * resShoot.bi[1]
                        * scalingFlux;
                PhiEven = gps.getPhi(0, gp.tempIndex, false) * resShoot.bi[0]
                    + gps.getPhi(1, gp.tempIndex, false) * resShoot.bi[1]
                        * scalingFlux;

                nbSamples = gps.getNSamplePhi(0, gp.tempIndex)
                    + gps.getNSamplePhi(1, gp.tempIndex);
              } else if (m_config.phiStatisticStrategy == 3) {
                PhiOdd = gps.getPhi(0, gp.tempIndex, true);
                PhiEven = gps.getPhi(0, gp.tempIndex, false);

                nbSamples = gps.getNSamplePhi(0, gp.tempIndex);
              } else {
                SLog(EError, "No Psi managed");
              }
            }

            // Case of Metropolis Mi statistic eval
            // need to normalize it by the normalization
            // factor
            if (m_config.numberStrat == ENbMetropolis) {
              M *= resShoot.bi[0];  // TODO Check normalisation factor, correct one?
              // FIXME: Statistic incompatible to 3 chains
              if (m_config.numberChains != 1) {
                SLog(EError, "Incompatible statistic");
              }
            }
            MTotal += M;
            gps.nPhotons += M;

            // Compute maxImportance for JaroMIS
            // FIXME: WARNING
            // FIXME: 4CHAINS
            maxImportance = std::max(gp.importance[0], maxImportance);

            // Update the cluster
            // Attached to the importance function
            // To compute later the importance function attached to the clusters
            if (m_impFunc->getName() == "localImp") {
              // For the moment, Phi is only computed with the 1st level of MC
              // Because, this level is constant and may reach more fast the equilibrium.
              static_cast<LocalImp*>(m_impFunc)->updateCluster(gp, M, PhiOdd,
                                                               PhiEven,
                                                               nbSamples);
            }

            if (N + M > 0) {
              if (m_config.usedTechniques & MISHelper::SPPM_ONLY) {
                // === Debug section
                // This is the computation of the contribution
                // only for this iteration
                if (dumpAllInfo) {
                  std::vector<Spectrum> fluxPass(fluxChains.size());
                  for (size_t idFlux = 0; idFlux < fluxPass.size(); idFlux++) {
                    fluxPass[idFlux] = gp.weight * fluxChains[idFlux];
                  }
                  if (gp.points->radius == 0.f) {
                    // No radius: Not valid GP, normally
                    // we check here that the GP doesn't collect something
                    for (size_t idFlux = 0; idFlux < fluxPass.size();
                        idFlux++) {
                      if (fluxPass[idFlux] != Spectrum(0.f)) {
                        Log(EError, "GP without radius but collected flux.");
                      }
                    }
                  } else {
                    size_t nbEmittedPathTotal = 0;
                    for (int idChain = 0;
                        idChain < PixelData<GatherPoint>::nbChains; idChain++) {
                      nbEmittedPathTotal += resShoot.nbEmittedPath[idChain];
                    }

                    if (m_config.useMISLevel) {  // Special case: MIS only lowest level
                      nbEmittedPathTotal = resShoot.nbEmittedPath[0];
                    }

                    for (size_t idFlux = 0; idFlux < fluxPass.size();
                        idFlux++) {
                      fluxPass[idFlux] /= nbEmittedPathTotal * gp.points->radius
                          * gp.points->radius * M_PI;
                      contribPassOnly[idFlux] = fluxPass[idFlux];
                    }
                  }
                }

                ///////////////////
                // Step 2: Indirect evaluation
                ///////////////////

                // === Update Radii and number of photon accumulated
                Float ratio = (N + m_config.alpha * M) / (N + M);
                gps.N = N + m_config.alpha * M;
                gps.scale = gps.scale * std::sqrt(ratio);

                // === Update accumulated flux
                // Rescale the flux by divided it
                // by the radius used.
                ////////////
                // Step a: Compute the indirect component
                ////////////
                if (it->points->radius == 0.f) {
                  if (M != 0) {
                    SLog(EError, "Null radius but collected some photons");
                  }
                } else {
                  Spectrum fluxIteration(0.f);
                  for (size_t idFlux = 0; idFlux < fluxChains.size();
                      idFlux++) {
                    fluxIteration += fluxChains[idFlux];
                  }

                  if (m_config.showUpperLevels) {
                    // Take the upper level
                    fluxIteration = fluxChains[m_config.numberChains - 1];
                  }

                  gps.flux += gp.weight * fluxIteration;
                  gps.flux /= (gp.points->radius * gp.points->radius * M_PI);
                }
              }
            }
            // Should we quit after first valid GP in pixel list?
            //if (!updatePerGP)
            break;
          }
        }
        //maxImportance *= resShoot.bi;
        gps.cumulImportance += maxImportance;
        if (!(m_config.usedTechniques & MISHelper::SPPM_ONLY)
            && m_config.usedTechniques
                & (MISHelper::MERGE | MISHelper::CONNECT)) {
          // Choose the good flux
          Spectrum fluxIteration(0.f);
          for (size_t idFlux = 0; idFlux < fluxChains.size(); idFlux++) {
            fluxIteration += fluxChains[idFlux];
          }
          // MERGE & CONNECT from VCM, BPM
          gps.scale = (Float) (m_config.initialScale
              * std::pow((Float) currentIteration,
                         (m_config.alpha - 1.f) * 0.5f));

          //contribOnlyIterPixel1 += fluxIteration; // FIXME
          Float denom = 1.f / (Float) currentIteration;
          gps.flux *= (Float) currentIteration - 1.f;
          gps.flux += fluxIteration;
          gps.flux *= denom;
          contribPixel += gps.flux;
          SLog(
              EError,
              "VCM or BPM not compatible with 3 chains, solve implementation issues");
        } else if (m_config.usedTechniques & MISHelper::SPPM_ONLY) {
          ////////////
          // Step b: Normalize and merge the two component
          ////////////
          contribPixel = gps.flux / ((Float) m_totalEmittedPath);
        }

        // Handle direct illumination
        // Use classical MC estimator
        Spectrum directIllum =
            m_config.usedTechniques & MISHelper::DIRLIGHT ?
                MISHelper::computeWeightedDirectIllum(
                    gps, m_config.usedTechniques, (int) m_config.photonCount,
                    normalizationMIS[1],  // FIXME MARTIN (NORMALISATION)
                    m_config.removeDeltaPaths, 1) :
                Spectrum(0.f);
        for (size_t idFlux = 0; idFlux < fluxChains.size(); idFlux++) {
          contribPassOnly[idFlux] += directIllum;
        }
        Float denom = 1.f / (Float) currentIteration;
        gps.fluxDirect = (gps.fluxDirect * ((Float) currentIteration - 1.f)
            + directIllum) * denom;
        contribPixel += gps.fluxDirect;
        // No filter, we write directly in the bitmap

        // Put the final result in the image
        target[gps.pos.y * m_bitmap->getWidth() + gps.pos.x] = contribPixel;

        for (size_t idFlux = 0; idFlux < fluxChains.size(); idFlux++) {
          ((Spectrum*) passContribs[idFlux]->getData())[gps.pos.y
              * m_bitmap->getWidth() + gps.pos.x] = contribPassOnly[idFlux];
        }
        targetPassContribMi[gps.pos.y * m_bitmap->getWidth() + gps.pos.x] =
            Spectrum(MTotal);
      }
    }

    // Finished to update all pixels statistics
    // Now, if we need, we update cluters statistics
    if (m_impFunc->getName() == "localImp") {
      dynamic_cast<LocalImp*>(m_impFunc)->compactClusterStatistic((int) nCores);
      dynamic_cast<LocalImp*>(m_impFunc)->printVariance();
    }

    // === Dump on the disk the current
    // iteration contribution if is needed
    if (dumpAllInfo) {
      for (int idChain = 0; idChain < PixelData<GatherPoint>::nbChains;
          idChain++) {
        std::stringstream ssContrib;
        ssContrib << "contrib" << idChain;
        saveBitmap(scene, passContribs[idChain].get(), ssContrib.str(),
                   currentIteration);
      }

      saveBitmap(scene, passContribBitmapMi.get(), "Mi", currentIteration);
    }

    // === Update the bitmap
    // and show the results
    film->setBitmap(m_bitmap);
    film->develop(scene, 0.f);
    queue->signalRefresh(job);
  }

  /**
   * Normalization procedure of the importance function
   * This function is for debugging propose and solve some
   * numerical issue due to too high accumulated values
   * in the gather points
   */
  Float nomarlizeImp(int idImportance) {
    double targetImp = m_bitmap->getSize().x * m_bitmap->getSize().y * 1.f;
    double sumImportance = 0.f;
    Float minImp = 10000.f;
    Float maxImp = 0.f;

    // Compute the min,max,sum of the importance associated to the gps
    // Non valid GP is ignored
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
      GatherBlock &gatherPoints = m_gatherBlocks[blockIdx];
      for (size_t i = 0; i < gatherPoints.size(); ++i) {
		GatherPointsList & gpl = gatherPoints[i];
		for(GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it) {
			if (it->depth != -1) {  // if gp.depth == -1 => non valid gp
			    /*if(rootImp) {
			        if(it->importance[idImportance] != 0) {
			            it->importance[idImportance] = math::safe_sqrt(it->importance[idImportance]);
			        } else {
			            SLog(EWarn, "0 Importance");
			        }
			    }*/
			  sumImportance += it->importance[idImportance];
			  targetImp += 1.f;
			  minImp = std::min(minImp, it->importance[idImportance]);
			  maxImp = std::max(maxImp, it->importance[idImportance]);
			}
        }
      }
    }

    // Compute the factor to renormalize the importance function
    double factor = sumImportance / targetImp;
    Log(EInfo, "Normalisation Imp: (min: %f, max: %f, sum: %f, factor: %f)",
        minImp, maxImp, sumImportance, factor);

    // Apply this factor for all importance associated to GP
    // do it for all, invalid gp will be ignored
    for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
	  GatherBlock &gatherPoints = m_gatherBlocks[blockIdx];
	  for (size_t i = 0; i < gatherPoints.size(); ++i) {
		GatherPointsList & gpl = gatherPoints[i];
		  for(GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it) {
		    it->importance[idImportance] = (Float)( it->importance[idImportance] / factor);
		  }
      }
    }

    return (Float) (1.f/factor);
  }

  //////////////////////////////////////
  /// Function from hachisuka
  //////////////////////////////////////
  void resetGatherPoints() {
	  for (int blockIdx = 0; blockIdx < (int) m_gatherBlocks.size(); ++blockIdx) {
		GatherBlock &gatherPoints = m_gatherBlocks[blockIdx];
		for (size_t i = 0; i < gatherPoints.size(); ++i) {
			GatherPointsList & gpl = gatherPoints[i];
			gpl.resetTemp();
			gpl.flux = Spectrum(0.f);
			gpl.fluxDirect = Spectrum(0.f);
			gpl.scale = m_config.initialScale;
		}
    }

    m_totalEmittedPath = 0;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "ISPPMIntegrator[" << endl << "]";
    return oss.str();
  }

  MTS_DECLARE_CLASS()

 protected:
  void writeRadii(int it, Scene* scene, bool initialize = false) {
    if (!initialize && !m_dumpAtEachIterationRadii) {
      return;  //< No operation is needed
    }
    if (((it - 1) % m_stepSnapshot) != 0) {
      return;
    }

    Film* film = scene->getFilm();

    // === Print out image radius
    // Build name
    std::string nameFile;
    if (initialize) {
        nameFile = "radiusInit";
    } else {
        nameFile = "radius";
    }

    // Build the bitmap contains all radius
    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
                                    film->getSize());
    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
      for (int j = 0; j < (int) gps.size(); j++) {
        GatherPointsList &list = (gps[j]);
		for ( GatherPointsList::iterator it = list.begin(); it != list.end(); ++it )
		{
			if ( it->depth != -1 ) {
				target[list.pos.y * bitmap->getWidth() + list.pos.x] =
					Spectrum(it->points->radius * it->points->radius * M_PI);
				break;
			}
		}
      }
    }

    saveBitmap(scene, bitmap.get(), nameFile, it);
  }

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

  void writeImportance(int idPass, Scene* scene, int idImportance) {
    if (((idPass - 1) % m_stepSnapshot) != 0)
      return;

    // === Don't need !
    // if the function is not dynamic
    // indeed, the importance function will not change.
    if (idPass > 1 && !m_impFunc->isDynamic())
      return;

    Float norm = 1.f / std::max(idPass - 1, 1);
    Film* film = scene->getFilm();
    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
                                    film->getSize());
    bitmap->clear();

    ref<Bitmap> bitmapK = bitmap->clone();
    ref<Bitmap> bitmapI = bitmap->clone();
    ref<Bitmap> bitmapD = bitmap->clone();
    ref<Bitmap> bitmapC = bitmap->clone();

    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      Spectrum *target = (Spectrum *) bitmap->getData();
      Spectrum *targetK = (Spectrum *) bitmapK->getData();
      Spectrum *targetI = (Spectrum *) bitmapI->getData();
      Spectrum *targetD = (Spectrum *) bitmapD->getData();
      Spectrum *targetC = (Spectrum *) bitmapC->getData();
      for (int j = 0; j < (int) gps.size(); j++) {
        GatherPointsList &list = (gps[j]);
        for (GatherPointsList::iterator it = list.begin(); it != list.end();
            ++it) {
          // Just write the importance for the first valid gatherpoint.
          if (it->depth != -1) {
            target[list.pos.y * bitmap->getWidth() + list.pos.x] = Spectrum(
                it->importance[idImportance]);
            targetK[list.pos.y * bitmapK->getWidth() + list.pos.x] = Spectrum(
                it->kappa);
            targetI[list.pos.y * bitmapI->getWidth() + list.pos.x] = Spectrum(
                it->invSurf);
            targetD[list.pos.y * bitmapD->getWidth() + list.pos.x] = Spectrum(
                it->density);
            break;
          } else {
            target[list.pos.y * bitmap->getWidth() + list.pos.x] = Spectrum(
                0.f);
            targetK[list.pos.y * bitmapK->getWidth() + list.pos.x] = Spectrum(
                0.f);
            targetI[list.pos.y * bitmapI->getWidth() + list.pos.x] = Spectrum(
                0.f);
            targetD[list.pos.y * bitmapD->getWidth() + list.pos.x] = Spectrum(
                0.f);
          }
        }
        targetC[list.pos.y * bitmapC->getWidth() + list.pos.x] = Spectrum(
            list.nPhotons) * norm;
      }
    }

    saveBitmap(scene, bitmap.get(), "imp", idPass);
#if VERBOSE_INFO
    saveBitmap(scene, bitmapC.get(), "count", idPass);
    if (m_impFunc->getName() == "localImp") {
      saveBitmap(scene, bitmapK.get(), "kappa", idPass);
      saveBitmap(scene, bitmapI.get(), "invSurf", idPass);
      saveBitmap(scene, bitmapD.get(), "density", idPass);
    } else if (m_impFunc->getName() == "invSurf") {
      saveBitmap(scene, bitmapI.get(), "invSurf", idPass);
    }
#endif
  }


  void writeImportanceInfo(int idPass, Scene* scene) {
      if (((idPass - 1) % m_stepSnapshot) != 0)
        return;

      if(!m_dumpExtraImportanceInfo)
          return;

      LocalImp* imp = 0;
      if(m_impFunc->getName() == "localImp") {
          imp = dynamic_cast<LocalImp*>(m_impFunc);
      } else {
          return;
      }

      // === Don't need !
      // if the function is not dynamic
      // indeed, the importance function will not change.
      if (idPass > 1 && !m_impFunc->isDynamic())
        return;

      Film* film = scene->getFilm();
      ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
                                      film->getSize());
      for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
        GatherBlock &gps = m_gatherBlocks[i];
        Spectrum *target = (Spectrum *) bitmap->getData();
        for (int j = 0; j < (int) gps.size(); j++) {
            GatherPointsList &list = (gps[j]);
            for ( GatherPointsList::iterator it = list.begin(); it != list.end(); ++it )
            {
              // Just write the importance for the first valid gatherpoint.
                if ( it->depth != -1 ) {
                    target[list.pos.y * bitmap->getWidth() + list.pos.x] = Spectrum(
                            imp->getDensity(*it));
                    break;
                }
            }
        }
      }

      saveBitmap(scene, bitmap.get(), "imp_density", idPass);
    }

  void writeGPAbsPosition(int idPass, Scene* scene) {
    if (((idPass - 1) % m_stepSnapshot) != 0)
      return;

    Film* film = scene->getFilm();
    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat,
                                    film->getSize());
    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
	  for (int j = 0; j < (int) gps.size(); j++) {
		  GatherPointsList &list = (gps[j]);
		  for ( GatherPointsList::iterator it = list.begin(); it != list.end(); ++it )
		  {
			  if ( it->depth != -1 ) {
				Float gpPosAbs[3] = { std::abs(it->its.p.x), std::abs(it->its.p.y),
					std::abs(it->its.p.z) };
				target[list.pos.y * bitmap->getWidth() + list.pos.x] = Spectrum(gpPosAbs);
				break;
			  }
		  }
      }
    }

    saveBitmap(scene, bitmap.get(), "gpPos", idPass);
  }

  void writeGPFile(const std::string& path) {
    std::ofstream outFile(path.c_str(), std::ofstream::out | std::ofstream::binary);
    outFile.write("#GPS",sizeof(char)*4);

    // Count nb of valid GP
    int nbGPValid = 0;
    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      for (int j = 0; j < (int) gps.size(); j++) {
        GatherPointsList &list = (gps[j]);
        for (GatherPointsList::iterator it = list.begin(); it != list.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            nbGPValid += 1;
          }
        }
      }
    }

    outFile.write((char*)&nbGPValid,sizeof(int));

    // Write all GP
    for (int i = 0; i < (int) m_gatherBlocks.size(); ++i) {
      GatherBlock &gps = m_gatherBlocks[i];
      for (int j = 0; j < (int) gps.size(); j++) {
        GatherPointsList &list = (gps[j]);
        for (GatherPointsList::iterator it = list.begin(); it != list.end();
            ++it) {
          if (it->depth != -1 && it->its.isValid()) {
            outFile.write((char*) &it->its.p.x, sizeof(float));
            outFile.write((char*) &it->its.p.y, sizeof(float));
            outFile.write((char*) &it->its.p.z, sizeof(float));

            outFile.write((char*) &list.pos.x, sizeof(int));
            outFile.write((char*) &list.pos.y, sizeof(int));

            outFile.write((char*) &list.N, sizeof(float));

            outFile.write((char*) &it->depth, sizeof(int));
          }
        }
      }
    }
  }

  /**
   * Compute the MSE value
   */
  Float getMSE(bool firstPass) {
    double MSE = 0.0f;
    for (int i = 0; i < m_bitmap->getWidth(); ++i) {
      for (int j = 0; j < m_bitmap->getHeight(); j++) {
        Spectrum refSpec = m_refBitmap->getPixel(Point2i(i, j));
        Spectrum diff = m_bitmap->getPixel(Point2i(i, j)) - refSpec;
        MSE += diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
      }
    }
    MSE /= m_bitmap->getWidth() * m_bitmap->getHeight();

    return (Float)MSE;
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
  // Global configuration
  MSPPMConfiguration m_config;

  // Running attributes
  // --- Rendering
  bool m_running;
  int m_maxPass;
  size_t m_totalEmittedPath;

  // --- Frequency of dumping stats
  int m_stepSnapshot;
  int m_stepDensitySnapshot;

  // --- Options behavior
  bool m_computeDirect;
  bool m_showImpFunction;
  bool m_dumpImagePass;
  bool m_dumpAtEachIterationRadii;
  bool m_dumpExtraImportanceInfo;

  // Object that control the importance function
  ImportanceFunction* m_impFunc;
  int m_impFuncRessourceID;

  // SPPM Internal data
  // Over the gather points (raddi, generation ... etc).
  // + Internal structure
  GatherBlocks m_gatherBlocks;
  std::vector<Point2i> m_offset;
  ref<Mutex> m_mutex;
  ref<Bitmap> m_bitmap;
  ref<Bitmap> m_refBitmap;  //< Case if the reference is given
  std::string m_nameRef;

  RadiusInitializer* m_gpManager;

  // The last term is importance because some importance function can be scaled differently.
  // This term is usefull to know the scaling factor of the importance function in order
  // to correct average normalisation factor over passes.

  bool m_needSaveImp;
  int m_stepSaveImp;
  std::string m_precomputedIMP;
};

MTS_IMPLEMENT_CLASS_S(MSPPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(MSPPMIntegrator, "Improved SPPM with Metropolis Sampling");

MTS_NAMESPACE_END
