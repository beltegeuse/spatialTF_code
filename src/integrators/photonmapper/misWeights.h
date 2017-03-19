#pragma once

MTS_NAMESPACE_BEGIN

#include <mitsuba/mitsuba.h>

class MISHelper
{
public:

	enum Techniques
	{
		DIRLIGHT = 1,
		CONNECT = 2,
		MERGE = 4,
		SPPM_ONLY = 8,
		LIGHTPATHS = MERGE | CONNECT,
		BPT = DIRLIGHT | CONNECT,
		VCM = MERGE | BPT,
		SPPM = SPPM_ONLY | MERGE,
		SPPM_DIR = SPPM | DIRLIGHT
	};

	static Float lightPathRatio;

	// Parses value for techniques
	static void parse(const std::string & value, int &techniques, bool & multiGatherPoints)
	{
		if ( value.compare("sppm") == 0 )
		{
			techniques = SPPM;
			multiGatherPoints = false;
			return;
		}
		if ( value.compare("bpm") == 0 )
		{
			techniques = MERGE;
			multiGatherPoints = true;
			return;
		}
		if ( value.compare("sppm_dir") == 0 )
		{
			techniques = SPPM_DIR;
			multiGatherPoints = false;
			return;
		}
		if ( value.compare("bpm_dir") == 0 )
		{
			techniques = MERGE | DIRLIGHT;
			multiGatherPoints = true;
			return;
		}
		if ( value.compare("vcm") == 0 )
		{
			techniques = VCM;
			multiGatherPoints = true;
			return;
		}
		if ( value.compare("bpt") == 0 )
		{
			techniques = BPT;
			multiGatherPoints = true;
			return;
		}
		if ( value.compare("path") == 0 )
		{
			techniques = DIRLIGHT;
			multiGatherPoints = false;
			return;
		}
		// Default: sppm
		techniques = SPPM;
		multiGatherPoints = false;
	}

	static inline Float mergeAddhocMultiplier(int cameraLength, int totalLength)
	{
		/*Float w = (totalLength - cameraLength) / (Float) (totalLength - 1);
		if ( cameraLength != 1 )
			w *= 0.25f; // Blurring is over area of 4 pixels*/
		return 1.f;
	}

	// Accumulate camera vertices inverse weight
	static Float AccumulateCameraVerticesInverseWeight(
		// Path length
		int cameraPathLength,
		// Camera vertices
		const GatherPointsList &cameraVertices,
		// Last vertex reverse pdf with respect to area measure
		Float lastVertexReversePdfInAreaMeasure,
		// Last but one vertex reverse pdf with respect to area measure
		Float lastButOneVertexReversePdfInAreaMeasure,
		// Used techniques
		int usedTechniques,
		// Emitted paths count
		int emittedPaths,
		// Light path importance
		Float lightPathImportance,
		// Total length
		int totalPathLength
		)
	{
		Float invWeight = 0.0f; 
		Float product = 1.f;

		const Float mergeWeight = cameraVertices.radius * cameraVertices.radius * M_PI * emittedPaths;
		// We are going towards the camera
		for (int vertexIndex = cameraPathLength; vertexIndex >= 0; --vertexIndex)
		{
			// Two vertices of the current path segment 
			// next vertex is closer to the camera then previous
			const VertexMISInfo & current = cameraVertices[vertexIndex].misInfo;
			const VertexMISInfo & next = vertexIndex == 0 ? current : cameraVertices[vertexIndex - 1].misInfo;

			// None of these techniques works if the connection/merging vertex
			// is delta
			int techniques = usedTechniques;
			if (current.m_flags & VertexMISInfo::DELTA)
				techniques = 0;
			Float rev = vertexIndex == cameraPathLength ? 
						lastVertexReversePdfInAreaMeasure : 
						( vertexIndex == (cameraPathLength - 1) ? 
						lastButOneVertexReversePdfInAreaMeasure : current.m_vertexReversePdf );

			// Get reverse pdf (from previous vertex to the current )
			assert(rev); // TODO remove when debugged

			product *= rev;

			// Merging
			if (techniques & MERGE && cameraVertices[vertexIndex].depth != -1)
				invWeight += product * mergeWeight * mergeAddhocMultiplier(vertexIndex + 1, totalPathLength);

			assert(current.m_vertexInverseForwardPdf);
			
			product *= current.m_vertexInverseForwardPdf;

			// Connect
 			if (techniques & CONNECT && !(next.m_flags & VertexMISInfo::DELTA) && vertexIndex > 0)
				invWeight += product * lightPathRatio;
			
		}
		return invWeight;
	}

	// Accumulate light vertices inverse weight
	static Float AccumulateLightVerticesInverseWeight(
		// Path length
		int lightPathLength,
		// Light vertices
		const std::vector<PhotonInfo> & lightVertices,
		// Last vertex reverse pdf with respect to area measure
		Float lastVertexReversePdfInAreaMeasure,
		// Last but one vertex reverse pdf with respect to area measure
		Float lastButOneVertexReversePdfInAreaMeasure,
		// Used techniques
		int usedTechniques,
		// Are we merging?
		bool merging,
		// Direct illumination techniques inverse weight
		Float & directIllumInverseWeight,
		// Emitted paths count
		int emittedPaths,
		// Last gatherPoint
		const GatherPoint & gp,
		// Light path importance
		Float lightPathImportance,
		// Total length
		int totalPathLength
		)
	{
		Float invWeight = 0.0f; 
		Float product = 1.f;
		const Float mergeWeight = gp.points->radius * gp.points->radius * M_PI * emittedPaths;

		// We are going towards light
		for (int vertexIndex = lightPathLength; vertexIndex > 0; --vertexIndex)
		{
			// Two vertices of the current path segment 
			// next vertex is closer to the camera then previous
			const VertexMISInfo & current = lightVertices[vertexIndex].misInfo;
			const VertexMISInfo & next = lightVertices[vertexIndex - 1].misInfo;

			// None of these techniques works if the connection/merging vertex
			// is delta
			int techniques = usedTechniques;
			if (current.m_flags & VertexMISInfo::DELTA)
				techniques = 0;

			Float rev = vertexIndex == lightPathLength ? 
						lastVertexReversePdfInAreaMeasure : 
						( vertexIndex == (lightPathLength - 1) ? 
						lastButOneVertexReversePdfInAreaMeasure : current.m_vertexReversePdf );

			// Get reverse pdf (from previous vertex to the current )
			assert(rev); // TODO remove when debugged

			product *= rev;

			// Merging
			if (techniques & MERGE && (!merging || vertexIndex < lightPathLength) && !(techniques & SPPM_ONLY)) 
				invWeight += product * mergeWeight * mergeAddhocMultiplier(totalPathLength - vertexIndex, totalPathLength);

			assert(current.m_vertexInverseForwardPdf);

			product *= current.m_vertexInverseForwardPdf;

			// Connect
			if (techniques & CONNECT && !(next.m_flags & VertexMISInfo::DELTA) && vertexIndex > 1)
				invWeight += product * lightPathRatio;

		}
		directIllumInverseWeight = 0.f;
		if (usedTechniques & DIRLIGHT)
		{
			const VertexMISInfo & lastButOne = lightVertices[1].misInfo;
			// Direct illumination - last connect corresponds to DL
			if (!(lastButOne.m_flags & VertexMISInfo::DELTA) )
				directIllumInverseWeight += product; 
			// Direct hit of light source
			const VertexMISInfo & last = lightVertices[0].misInfo;
			directIllumInverseWeight += product * last.m_vertexReversePdf * last.m_vertexInverseForwardPdf;
		}
		return invWeight;
	}

	static Spectrum computeWeightedDirectIllum(
		// Camera vertices
		GatherPointsList &cameraVertices,
		// Used techniques
		int usedTechniques,
		// Emitted paths count
		int emittedPaths,
		// Importance normalization 
		Float importanceNormalization,
		// Remove completely delta paths
		bool removeCompletelyDelta,
		// The importance function used
		int idImportance
		)
	{
		Spectrum result(0.f);
		int length = 0;
		result += cameraVertices.emission;
		for(GatherPointsList::iterator it = cameraVertices.begin(); it != cameraVertices.end(); ++it, ++length)
		{
			if (!it->its.isValid())
				continue;
			unsigned int backupFlag = it->misInfo.m_flags;
			if ( !it->directIllumLightSample.isZero() ) {
				// Light sample
				it->misInfo.m_flags = it->lightSampleMIS.delta ? VertexMISInfo::DELTA : 0;
				Float wCamera = AccumulateCameraVerticesInverseWeight(
					length,
					cameraVertices,
					it->lightSampleMIS.m_lastVertexReversePdf,
					it->lightSampleMIS.m_lastButOneVertexReversePdf,
					usedTechniques,
					emittedPaths,
					it->misInfo.m_importance[idImportance],
					length + 2);
				Float misW = 1.f / (wCamera * it->misInfo.m_importance[idImportance] / importanceNormalization + it->lightSampleMIS.m_lightWeight);
				result += misW * it->directIllumLightSample;
			}
			// Bsdf sample
			if ( !( it->bsdfMIS.delta && it->completelyDelta && removeCompletelyDelta ) && !it->directIllumBSDF.isZero() )
			{
				it->misInfo.m_flags = it->bsdfMIS.delta ? VertexMISInfo::DELTA : 0;
				Float wCamera = AccumulateCameraVerticesInverseWeight(
					length,
					cameraVertices,
					it->bsdfMIS.m_lastVertexReversePdf,
					it->bsdfMIS.m_lastButOneVertexReversePdf,
					usedTechniques,
					emittedPaths,
					it->misInfo.m_importance[idImportance],
					length + 2);
				Float misW = 1.f / (wCamera * it->misInfo.m_importance[idImportance] / importanceNormalization + it->bsdfMIS.m_lightWeight);
				result += misW * it->directIllumBSDF;	
			}
			// Return original data
			it->misInfo.m_flags = backupFlag;
		}
		return result;
	}

	static void computeMergeWeights(const GatherPoint & gatherPoint, const PhotonSplattingList & photons, int photonIndex, Float bsdfForwardPdf, Float bsdfReversePdf,
		// Used techniques
		int usedTechniques,
		// Emitted paths count
		int emittedPaths,
		// Which importance function is used
		int idImportance,
		// Output
		Float & normalizedInverseWeight, Float & unnormalizedInverseWeight)
	{
		GatherPointsList &cameraVertices = *gatherPoint.points;
		int cameraPathLength = gatherPoint.pointIndex;
		Float mergeWeight = M_PI * cameraVertices.radius * cameraVertices.radius * emittedPaths * mergeAddhocMultiplier(cameraPathLength + 1, cameraPathLength + photonIndex + 1);
		Float inverseMergeWeight = 1.f / mergeWeight;
		Float wCameraMIS = (usedTechniques & SPPM_ONLY ) ? 1.f : AccumulateCameraVerticesInverseWeight(cameraPathLength,
			cameraVertices, inverseMergeWeight, 
			cameraPathLength == 0 ? 1.f : cameraVertices[cameraPathLength-1].misInfo.m_vertexReversePdfWithoutBSDF * bsdfReversePdf, 
			usedTechniques, emittedPaths, photons.getImp(idImportance), cameraPathLength + photonIndex + 1);

		Float wDirectLightMIS;
		Float wLightMIS = AccumulateLightVerticesInverseWeight(photonIndex,
			photons.photons, inverseMergeWeight, 
			photonIndex == 0 ? 1.f : photons.photons[photonIndex-1].misInfo.m_vertexReversePdfWithoutBSDF * bsdfForwardPdf,
			usedTechniques, true, wDirectLightMIS, emittedPaths, gatherPoint, photons.getImp(idImportance), cameraPathLength + photonIndex + 1);

		// Output
		normalizedInverseWeight = wCameraMIS + wLightMIS;
		unnormalizedInverseWeight = wDirectLightMIS / std::max(photons.photons[photonIndex].misInfo.m_importance[idImportance], gatherPoint.misInfo.m_importance[idImportance]);
	}

	static bool evaluteBSDF( const Intersection & its, const Vector & dir, Float & forwardPdf, Float & reversePdf, Float & cosine, Spectrum & contribution,
		ETransportMode forwardMode, ETransportMode reverseMode )
	{
		// Get bsdf pdfs
		Vector localDir = its.toLocal(dir);
		BSDFSamplingRecord bRecForward(its,its.wi,localDir,forwardMode);
		forwardPdf = its.getBSDF()->pdf(bRecForward);
		BSDFSamplingRecord bRecReverse(its,localDir,its.wi,reverseMode);
		reversePdf = its.getBSDF()->pdf(bRecReverse);
		// TODO: Multiply pdfs by RR
		cosine = Frame::cosTheta(bRecForward.wo);

		/* Prevent light leaks due to the use of shading normals */
		Vector worldWi = its.toWorld(its.wi);
		Float wiDotGeoN = dot(its.geoFrame.n, worldWi),
			woDotGeoN = dot(its.geoFrame.n, dir);

		if (wiDotGeoN * Frame::cosTheta(bRecForward.wi) <= 0 ||
			woDotGeoN * Frame::cosTheta(bRecForward.wo) <= 0)
			return false;

		contribution *= its.getBSDF()->eval(bRecForward);

		if (contribution.isZero())
			return false;

		if (forwardMode == EImportance) {
			/* Adjoint BSDF for shading normals */
			contribution *= std::abs(
				(Frame::cosTheta(bRecForward.wi) * woDotGeoN) /
				(Frame::cosTheta(bRecForward.wo) * wiDotGeoN));
		}

		contribution /= std::abs(cosine);

		return true;
	}

	static void computeConnections(GatherPointsList & cameraVertices, const PhotonSplattingList & lightVertices, int usedTechniques, int emittedPaths,
		int threadId,
		int maxDepth,
		Float normalization,
		Scene * scene,
		int idImportance
		)
	{
		// Try to connect each eye vertex with one light vertex (except the very first light vertex
		for ( GatherPointsList::iterator cV = cameraVertices.begin(); cV !=  cameraVertices.end(); ++cV )
		{
			if (!cV->its.isValid() || cV->misInfo.m_flags & VertexMISInfo::DELTA )
				continue;
			// Not have C++11 on the grid
			//for ( std::vector<PhotonInfo>::const_iterator lV = lightVertices.photons.cbegin() + 1; lV != lightVertices.photons.cend(); ++lV )
			for ( std::vector<PhotonInfo>::const_iterator lV = lightVertices.photons.begin() + 1; lV != lightVertices.photons.end(); ++lV )
			{
				if (!lV->its.isValid() || lV->misInfo.m_flags & VertexMISInfo::DELTA )
					continue;
				// Test the total length of path
				// Not have C++11 on the grid
				//int lightPathLength = (int)(lV - lightVertices.photons.cbegin());
				int lightPathLength = (int)(lV - lightVertices.photons.begin());
				int cameraPathLength = cV->pointIndex; 
				if (maxDepth > 0 && (cameraPathLength + lightPathLength + 2 ) > (maxDepth))
					break;
				// Compute direction between the two vertices
				Vector dir = lV->its.p - cV->its.p;
				const Float dist2 = dir.lengthSquared();
				const Float invDist2 = 1.f / dist2;
				const Float dist = sqrtf(dist2);
				dir /= dist;
				Spectrum contrib = cV->weight * lV->power;
				// Evaluate BSDF at camera vertex
				Float cameraForwardPdf, cameraReversePdf, cameraCosine;
				if (!evaluteBSDF(cV->its,dir,cameraForwardPdf,cameraReversePdf, cameraCosine, contrib, ERadiance, EImportance))
					continue;

				// Evaluate BSDF at light vertex
				Float lightForwardPdf, lightReversePdf, lightCosine;
				if (!evaluteBSDF(lV->its,-dir,lightForwardPdf,lightReversePdf, lightCosine, contrib, EImportance, ERadiance))
					continue;

				// Compute geometric term
				Float geometryTerm = invDist2 * absDot(cV->its.shFrame.n, dir) * absDot(lV->its.shFrame.n, dir);
				if (geometryTerm <= 0.f)
					continue;
				contrib *= geometryTerm;

				// Finally test occlusion
				Ray ray(cV->its.p, dir, Epsilon, dist * (1-ShadowEpsilon), cV->its.time);
				if (scene->rayIntersectAll(ray))
					continue;
				
				// As a last step compute MIS weights
				Float wCameraMIS = AccumulateCameraVerticesInverseWeight(cV->pointIndex,
					cameraVertices, lightForwardPdf * std::abs(cameraCosine) * invDist2 / lightPathRatio, 
					cameraPathLength == 0 ? 1.f : cameraVertices[cameraPathLength-1].misInfo.m_vertexReversePdfWithoutBSDF * cameraReversePdf, 
					usedTechniques, emittedPaths, lightVertices.getImp(idImportance), cameraPathLength + lightPathLength + 2);

				Float wDirectLightMIS = 0.f;
				Float wLightMIS = AccumulateLightVerticesInverseWeight(lightPathLength,
					lightVertices.photons, cameraForwardPdf * std::abs(lightCosine) * invDist2 / lightPathRatio, 
					lightPathLength == 0 ? 1.f : lightVertices.photons[lightPathLength-1].misInfo.m_vertexReversePdfWithoutBSDF * lightReversePdf,
					usedTechniques, false, wDirectLightMIS, emittedPaths, *cV, lightVertices.getImp(idImportance), cameraPathLength + lightPathLength + 2);

				Float mis = 1.f / ( 1.f + wCameraMIS + wLightMIS + wDirectLightMIS * normalization / std::max(lV->misInfo.m_importance[idImportance], cV->misInfo.m_importance[idImportance]) );
				
				// Store contribution
				Spectrum finalValue = mis * contrib / lightPathRatio;
				cameraVertices.addFlux(finalValue, threadId, idImportance);
			}
		}
	}
};

MTS_NAMESPACE_END
