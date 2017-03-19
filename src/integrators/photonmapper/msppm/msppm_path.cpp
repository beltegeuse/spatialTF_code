#include "msppm_path.h"

MTS_NAMESPACE_BEGIN

PhotonPathBuilder::PhotonPathBuilder(Scene *scene,
		int maxDepth, int rrDepth,
		Sampler* sampler, int idWorker,
		GatherPointMap* gatherMap):
	m_scene(scene),m_maxDepth(maxDepth),
	m_rrDepth(rrDepth), m_sampler(sampler),
	m_idWorker(idWorker),
	m_gatherMap(gatherMap) {

}

struct PathSeedNormalisationPhotonSortPredicate {
	bool operator()(const SeedNormalisationPhoton &left, const SeedNormalisationPhoton &right) {
		return left.sampleIndex < right.sampleIndex;
	}
};

Float PhotonPathBuilder::generateSeeds(size_t sampleCount, size_t seedCount,
			std::vector<SeedNormalisationPhoton> &seeds, int idImportance) {
	std::vector<SeedNormalisationPhoton> tempSeeds;
	tempSeeds.reserve(sampleCount);

	Float impAccum = 0.f;
	PhotonSplattingList list(PixelData<GatherPoint>::nbChains );
	for (size_t i=0; i<sampleCount; ++i) {
			size_t sampleIndex = m_sampler->getSampleIndex();
			samplePaths(list, idImportance);
			if(list.getImp(idImportance) != 0) {
				SeedNormalisationPhoton s;
				s.importance = list.getImp(idImportance);
				s.sampleIndex = sampleIndex;
				tempSeeds.push_back(s);
				impAccum += list.getImp(idImportance);
			}
	}

	DiscreteDistribution seedPDF(tempSeeds.size());
	for (size_t i=0; i<tempSeeds.size(); ++i)
		seedPDF.append(tempSeeds[i].importance);
	seedPDF.normalize();

	seeds.clear();
	seeds.reserve(seedCount);
	for (size_t i=0; i<seedCount; ++i)
		seeds.push_back(tempSeeds.at(seedPDF.sample(m_sampler->next1D())));

	std::sort(seeds.begin(), seeds.end(), PathSeedNormalisationPhotonSortPredicate());

	return impAccum / sampleCount;
}

void PhotonPathBuilder::samplePaths(PhotonSplattingList& list, int idImportance) {
	list.clear();

	Intersection its;
	ref<Sensor> sensor    = m_scene->getSensor();
	bool needsTimeSample  = sensor->needsTimeSample();
	PositionSamplingRecord pRec(sensor->getShutterOpen()
		+ 0.5f * sensor->getShutterOpenTime());

	/* Sample an emission */
	if (needsTimeSample)
		pRec.time = sensor->sampleTime(m_sampler->next1D());

	const Emitter *emitter = NULL;
	//const Medium *medium;

	Spectrum power;
	Ray ray;


	/* Sample both components together, which is potentially
	   faster / uses a better sampling strategy */
	power = m_scene->sampleEmitterRay(ray, emitter,
		m_sampler->next2D(), m_sampler->next2D(), pRec.time);

	int depth = 1, nullInteractions = 0;
	//bool delta = false;

	Spectrum throughput(1.0f); // unitless path throughput (used for russian roulette)
	while (!throughput.isZero() && (depth <= m_maxDepth || m_maxDepth < 0)) {
		m_scene->rayIntersectAll(ray, its);
		//Note: No Participating media

	    if (its.t == std::numeric_limits<Float>::infinity()) {
			/* There is no surface in this direction */
			break;
		} else {
			/* Sample
				tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
				Account for this and multiply by the proper per-color-channel transmittance.
			*/
			const BSDF *bsdf = its.getBSDF();

			/* Forward the surface scattering event to the attached handler */
			{
				int bsdfType = its.getBSDF()->getType();
				if (bsdfType & BSDF::EDiffuseReflection ||
					bsdfType & BSDF::EGlossyReflection) {

					//TODO
					ImportanceRes res = m_gatherMap->queryGPImpactedImportance(its, depth, idImportance);
					Float imp = res.importances[idImportance];
					if(imp != 0.f) {
						PhotonInfo p;
						p.depth = depth;
						p.its = its;
						p.power = throughput*power;
						list.add(p);
					}

					for(int idChains = 0; idChains < PixelData<GatherPoint>::nbChains; idChains++) {
					  list.setImp(idChains, std::max(res.importances[idChains], list.getImp(idChains)));
					}

				}
			}

			BSDFSamplingRecord bRec(its, m_sampler, EImportance);
			Spectrum bsdfWeight = bsdf->sample(bRec, m_sampler->next2D());
			if (bsdfWeight.isZero())
				break;

			/* Prevent light leaks due to the use of shading normals -- [Veach, p. 158] */
			Vector wi = -ray.d, wo = its.toWorld(bRec.wo);
			Float wiDotGeoN = dot(its.geoFrame.n, wi),
				  woDotGeoN = dot(its.geoFrame.n, wo);
			if (wiDotGeoN * Frame::cosTheta(bRec.wi) <= 0 ||
				woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
				break;

			/* Keep track of the weight, medium and relative
			   refractive index along the path */
			throughput *= bsdfWeight;
			if (its.isMediumTransition()) {
				SLog(EError, "Not Implemented");
			}


			if (bRec.sampledType & BSDF::ENull)
				++nullInteractions;
			//else
			//	delta = bRec.sampledType & BSDF::EDelta;

#if 0
			/* This is somewhat unfortunate: for accuracy, we'd really want the
			   correction factor below to match the path tracing interpretation
			   of a scene with shading normals. However, this factor can become
			   extremely large, which adds unacceptable variance to output
			   renderings.

			   So for now, it is disabled. The adjoint particle tracer and the
			   photon mapping variants still use this factor for the last
			   bounce -- just not for the intermediate ones, which introduces
			   a small (though in practice not noticeable) amount of error. This
			   is also what the implementation of SPPM by Toshiya Hachisuka does.

			   Ultimately, we'll need better adjoint BSDF sampling strategies
			   that incorporate these extra terms */

			/* Adjoint BSDF for shading normals -- [Veach, p. 155] */
			throughput *= std::abs(
						(Frame::cosTheta(bRec.wi) * woDotGeoN)/t'en f'
				(Frame::cosTheta(bRec.wo) * wiDotGeoN));
#endif

			ray.setOrigin(its.p);
			ray.setDirection(wo);
			ray.mint = Epsilon;
		}
		if (depth++ >= m_rrDepth) {
			/* Russian roulette: try to keep path weights equal to one,
			   Stop with at least some probability to avoid
			   getting stuck (e.g. due to total internal reflection) */

			Float q = std::min(throughput.max(), (Float) 0.95f);
			if (m_sampler->next1D() >= q)
				break;
			throughput /= q;
		}
	}
}

MTS_IMPLEMENT_CLASS(PhotonPathBuilder, false, Object);
MTS_IMPLEMENT_CLASS(PhotonSplattingList, false, SerializableObject);
MTS_IMPLEMENT_CLASS(PhotonPaths, false, SerializableObject);
MTS_IMPLEMENT_CLASS(MCData, false, SerializableObject);

MTS_NAMESPACE_END
