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

#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

/*! \plugin{direct}{Direct illumination integrator}
 * \order{1}
 * \parameters{
 *     \parameter{shadingSamples}{\Integer}{This convenience parameter can be
 *         used to set both \code{emitterSamples} and \code{bsdfSamples} at
 *         the same time.
 *     }
 *     \parameter{emitterSamples}{\Integer}{Optional more fine-grained
 *        parameter: specifies the number of samples that should be generated
 *        using the direct illumination strategies implemented by the scene's
 *        emitters\default{set to the value of \code{shadingSamples}}
 *     }
 *     \parameter{bsdfSamples}{\Integer}{Optional more fine-grained
 *        parameter: specifies the number of samples that should be generated
 *        using the BSDF sampling strategies implemented by the scene's
 *        surfaces\default{set to the value of \code{shadingSamples}}
 *     }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See
 *        page~\pageref{sec:strictnormals} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 * \vspace{-1mm}
 * \renderings{
 *     \medrendering{Only BSDF sampling}{integrator_direct_bsdf}
 *     \medrendering{Only emitter sampling}{integrator_direct_lum}
 *     \medrendering{BSDF and emitter sampling}{integrator_direct_both}
 *     \caption{
 *         \label{fig:integrator-direct}
 *         This plugin implements two different strategies for computing the
 *         direct illumination on surfaces. Both of them are dynamically
 *         combined then obtain a robust rendering algorithm.
 *     }
 * }
 *
 * This integrator implements a direct illumination technique that makes use
 * of \emph{multiple importance sampling}: for each pixel sample, the
 * integrator generates a user-specifiable number of BSDF and emitter
 * samples and combines them using the power heuristic. Usually, the BSDF
 * sampling technique works very well on glossy objects but does badly
 * everywhere else (\subfigref{integrator-direct}{a}), while the opposite
 * is true for the emitter sampling technique
 * (\subfigref{integrator-direct}{b}). By combining these approaches, one
 * can obtain a rendering technique that works well in both cases
 * (\subfigref{integrator-direct}{c}).
 *
 * The number of samples spent on either technique is configurable, hence
 * it is also possible to turn this plugin into an emitter sampling-only
 * or BSDF sampling-only integrator.
 *
 * For best results, combine the direct illumination integrator with the
 * low-discrepancy sample generator (\code{ldsampler}). Generally, the number
 * of pixel samples of the sample generator can be kept relatively
 * low (e.g. \code{sampleCount=4}), whereas the \code{shadingSamples}
 * parameter of this integrator should be increased until the variance in
 * the output renderings is acceptable.
 *
 * \remarks{
 *    \item This integrator does not handle participating media or
 *          indirect illumination.
 * }
 */

class MINormalIntegrator : public SamplingIntegrator {
public:
	MINormalIntegrator(const Properties &props) : SamplingIntegrator(props) {

		/* Be strict about potential inconsistencies involving shading normals? */
		m_strictNormals = props.getBoolean("strictNormals", false);
		/* When this flag is set to true, contributions from directly
		 * visible emitters will not be included in the rendered image */
		m_hideEmitters = props.getBoolean("hideEmitters", false);

		m_showShadingNormal = props.getBoolean("showShadingNormal");
	}

	/// Unserialize from a binary data stream
	MINormalIntegrator(Stream *stream, InstanceManager *manager)
	 : SamplingIntegrator(stream, manager) {
		m_strictNormals = stream->readBool();
		m_hideEmitters = stream->readBool();
		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		SamplingIntegrator::serialize(stream, manager);
		stream->writeBool(m_strictNormals);
		stream->writeBool(m_hideEmitters);
	}

	void configure() {
		SamplingIntegrator::configure();
	}

	void configureSampler(const Scene *scene, Sampler *sampler) {
		SamplingIntegrator::configureSampler(scene, sampler);
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		Point2 sample;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		if (!rRec.rayIntersect(ray)) {
			/* If no intersection could be found, possibly return
			   radiance from a background emitter */
			if (rRec.type & RadianceQueryRecord::EEmittedRadiance && !m_hideEmitters)
				return scene->evalEnvironment(ray);
			else
				return Spectrum(0.0f);
		}

		/* Possibly include emitted radiance if requested */
		if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && !m_hideEmitters)
			Li += its.Le(-ray.d);

		/* Include radiance from a subsurface scattering model if requested */
		if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
			Li += its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

		if (!(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)
			|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
				* Frame::cosTheta(its.wi) >= 0)) {
			/* Only render the direct illumination component if
			 *
			 * 1. It was requested
			 * 2. The surface has an associated BSDF (i.e. it isn't an index-
			 *    matched medium transition -- this is not supported by 'direct')
			 * 3. If 'strictNormals'=true, when the geometric and shading
			 *    normals classify the incident direction to the same side
			 */
			return Li;
		}

		/* ==================================================================== */
		/*                          Emitter sampling                          */
		/* ==================================================================== */
		bool adaptiveQuery = (rRec.extra & RadianceQueryRecord::EAdaptiveQuery);

		if (rRec.depth > 1 || adaptiveQuery) {
			/* This integrator is used recursively by another integrator.
			   Be less accurate as this sample will not directly be observed. */

		}

		Normal n = its.geoFrame.n;
		if(m_showShadingNormal) {
			n = its.shFrame.n;
		}

		Float v[3];
		v[0] = n.x*0.5 + 0.5;
		v[1] = n.y*0.5 + 0.5;
		v[2] = n.z*0.5 + 0.5;
		Li += Spectrum(v);

		return Li;
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA; pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MINormalIntegrator[" << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	bool m_strictNormals;
	bool m_hideEmitters;
	bool m_showShadingNormal;
};

MTS_IMPLEMENT_CLASS_S(MINormalIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MINormalIntegrator, "Direct illumination integrator");
MTS_NAMESPACE_END
