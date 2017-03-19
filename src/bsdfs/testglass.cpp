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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{dielectric}{Smooth dielectric material}
 * \order{3}
 * \icon{bsdf_dielectric}
 * \parameters{
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{bk7} / 1.5046}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{air} / 1.000277}}
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 *     \parameter{specular\showbreak Transmittance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular transmission component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 *
 * \renderings{
 *     \medrendering{Air$\leftrightarrow$Water (IOR: 1.33) interface.
 *         See \lstref{dielectric-water}.}{bsdf_dielectric_water}
 *     \medrendering{Air$\leftrightarrow$Diamond (IOR: 2.419)}{bsdf_dielectric_diamond}
 *     \medrendering{Air$\leftrightarrow$Glass (IOR: 1.504) interface with absorption.
 *         See \lstref{dielectric-glass}.}{bsdf_dielectric_glass}
 * }
 *
 * This plugin models an interface between two dielectric materials having mismatched
 * indices of refraction (for instance, water and air). Exterior and interior IOR values
 * can be specified independently, where ``exterior'' refers to the side that contains
 * the surface normal. When no parameters are given, the plugin activates the defaults, which
 * describe a borosilicate glass BK7/air interface.
 *
 * In this model, the microscopic structure of the surface is assumed to be perfectly
 * smooth, resulting in a degenerate\footnote{Meaning that for any given incoming ray of light,
 * the model always scatters into a discrete set of directions, as opposed to a continuum.}
 * BSDF described by a Dirac delta distribution. For a similar model that instead describes a
 * rough surface microstructure, take a look at the \pluginref{roughdielectric} plugin.
 *
 * \begin{xml}[caption=A simple air-to-water interface, label=lst:dielectric-water]
 * <shape type="...">
 *     <bsdf type="dielectric">
 *         <string name="intIOR" value="water"/>
 *         <string name="extIOR" value="air"/>
 *     </bsdf>
 * <shape>
 * \end{xml}
 *
 * When using this model, it is crucial that the scene contains
 * meaningful and mutually compatible indices of refraction changes---see
 * \figref{glass-explanation} for a description of what this entails.
 *
 * In many cases, we will want to additionally describe the \emph{medium} within a
 * dielectric material. This requires the use of a rendering technique that is
 * aware of media (e.g. the volumetric path tracer). An example of how one might
 * describe a slightly absorbing piece of glass is shown below:
 * \begin{xml}[caption=A glass material with absorption (based on the
 *    Beer-Lambert law). This material can only be used by an integrator
 *    that is aware of participating media., label=lst:dielectric-glass]
 * <shape type="...">
 *     <bsdf type="dielectric">
 *         <float name="intIOR" value="1.504"/>
 *         <float name="extIOR" value="1.0"/>
 *     </bsdf>
 *
 *     <medium type="homogeneous" name="interior">
 *         <rgb name="sigmaS" value="0, 0, 0"/>
 *         <rgb name="sigmaA" value="4, 4, 2"/>
 *     </medium>
 * <shape>
 * \end{xml}
 * \vspace{1cm}
 *
 * \begin{table}[h!]
 *     \centering
 *     \begin{tabular}{>{\ttfamily}p{5cm}r@{.}lp{.8cm}>{\ttfamily}p{5cm}r@{.}l}
 *         \toprule
 *         \rmfamily \textbf{Name} & \multicolumn{2}{l}{\textbf{Value}}& &
 *         \rmfamily \textbf{Name} & \multicolumn{2}{l}{\textbf{Value}}\\
 *         \cmidrule{1-3} \cmidrule{5-7}
 *         vacuum               & 1 & 0 &  &
 *         bromine              & 1 & 661\\
 *         helium               & 1 & 00004 & &
 *         water ice            & 1 & 31\\
 *         hydrogen             & 1 & 00013& &
 *         fused quartz         & 1 & 458\\[-.8mm]
 *         \cmidrule{5-7}\\[-5.5mm]
 *         air                  & 1 & 00028& &
 *         pyrex                & 1 & 470\\
 *         carbon dioxide       & 1 & 00045& &
 *         acrylic glass        & 1 & 49\\[-.8mm]
 *         \cmidrule{1-3}\\[-5.5mm]
 *         water                & 1 & 3330& &
 *         polypropylene        & 1 & 49\\
 *         acetone              & 1 & 36 & &
 *         bk7                  & 1 & 5046\\
 *         ethanol              & 1 & 361& &
 *         sodium chloride      & 1 & 544\\
 *         carbon tetrachloride & 1 & 461& &
 *         amber                & 1 & 55\\
 *         glycerol             & 1 & 4729& &
 *         pet                  & 1 & 575\\
 *         benzene              & 1 & 501& &
 *         diamond              & 2 & 419\\
 *         silicone oil         & 1 & 52045\\
 *         \bottomrule
 *     \end{tabular}
 *     \caption{
 *         \label{tbl:dielectric-iors}
 *          This table lists all supported material names along with
 *          along with their associated index of refraction at standard
 *          conditions. These material names can be used with the plugins
 *          \pluginref{dielectric},\
 *          \pluginref{roughdielectric},\
 *          \pluginref{plastic}, \
 *          \pluginref{roughplastic}, as well as
 *          \pluginref{coating}.
 *     }
 * \end{table}
 * \remarks{
 *  \item Dispersion is currently unsupported but will be enabled in a future release.
 * }
 */
class TestGlass : public BSDF {
public:
	TestGlass(const Properties &props) : BSDF(props) {

		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));
		m_specularTransmittance = new ConstantSpectrumTexture(
			props.getSpectrum("specularTransmittance", Spectrum(1.0f)));
	}

	TestGlass(Stream *stream, InstanceManager *manager)
			: BSDF(stream, manager) {
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_specularReflectance.get());
		manager->serialize(stream, m_specularTransmittance.get());
	}

	void configure() {
		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);
		m_specularTransmittance = ensureEnergyConservation(
			m_specularTransmittance, "specularTransmittance", 1.0f);

		m_components.clear();
		m_components.push_back(EDeltaReflection | EFrontSide | EBackSide
			| (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));
		m_components.push_back(EDeltaTransmission | EFrontSide | EBackSide | ENonSymmetric
			| (m_specularTransmittance->isConstant() ? 0 : ESpatiallyVarying));

		m_usesRayDifferentials =
			m_specularReflectance->usesRayDifferentials() ||
			m_specularTransmittance->usesRayDifferentials();

		m_probBounce = 0.5f;//m_specularReflectance->getAverage().getLuminance() /
		    //(m_specularTransmittance->getAverage().getLuminance() + m_specularReflectance->getAverage().getLuminance());

		BSDF::configure();
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else if (name == "specularTransmittance")
				m_specularTransmittance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	/// Reflection in local coordinates
	inline Vector reflect(const Vector &wi) const {
		return Vector(-wi.x, -wi.y, wi.z);
	}

	/// Refraction in local coordinates
	inline Vector refract(const Vector &wi, Float cosThetaT) const {
		Float scale = -1.f;
		return Vector(scale*wi.x, scale*wi.y, cosThetaT);
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
				&& (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
		bool sampleTransmission = (bRec.typeMask & EDeltaTransmission)
				&& (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;

		Float cosThetaT = -Frame::cosTheta(bRec.wi);
		if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
			if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
				return Spectrum(0.0f);

			return m_specularReflectance->eval(bRec.its);
		} else {
			if (!sampleTransmission || std::abs(dot(refract(bRec.wi, cosThetaT), bRec.wo)-1) > DeltaEpsilon)
				return Spectrum(0.0f);

			return m_specularTransmittance->eval(bRec.its);
		}
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
				&& (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
		bool sampleTransmission = (bRec.typeMask & EDeltaTransmission)
				&& (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;

		Float cosThetaT = -Frame::cosTheta(bRec.wi);
		if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
			if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
				return 0.0f;

			return 1.0f;
		} else {
			if (!sampleTransmission || std::abs(dot(refract(bRec.wi, cosThetaT), bRec.wo)-1) > DeltaEpsilon)
				return 0.0f;

			return 1.0f;
		}
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
				&& (bRec.component == -1 || bRec.component == 0);
		bool sampleTransmission = (bRec.typeMask & EDeltaTransmission)
				&& (bRec.component == -1 || bRec.component == 1);

		Float cosThetaT = -Frame::cosTheta(bRec.wi);

		if (sampleTransmission && sampleReflection) {
			if (sample.x <= m_probBounce) {
				bRec.sampledComponent = 0;
				bRec.sampledType = EDeltaReflection;
				bRec.wo = reflect(bRec.wi);
				bRec.eta = 1.0f;
				pdf = m_probBounce;

				return m_specularReflectance->eval(bRec.its) / m_probBounce;
			} else {
				bRec.sampledComponent = 1;
				bRec.sampledType = EDeltaTransmission;
				bRec.wo = refract(bRec.wi, cosThetaT);
				bRec.eta = 1.f;
				pdf = 1.f-m_probBounce;

				return m_specularTransmittance->eval(bRec.its) / (1.f-m_probBounce);
			}
		} else if (sampleReflection) {
			bRec.sampledComponent = 0;
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;
			pdf = 1.0f;

			return m_specularReflectance->eval(bRec.its);
		} else if (sampleTransmission) {
			bRec.sampledComponent = 1;
			bRec.sampledType = EDeltaTransmission;
			bRec.wo = refract(bRec.wi, cosThetaT);
			bRec.eta = 1.0f;
			pdf = 1.0f;

			return m_specularTransmittance->eval(bRec.its);
		}

		return Spectrum(0.0f);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
				&& (bRec.component == -1 || bRec.component == 0);
		bool sampleTransmission = (bRec.typeMask & EDeltaTransmission)
				&& (bRec.component == -1 || bRec.component == 1);

		Float cosThetaT = -Frame::cosTheta(bRec.wi);

		if (sampleTransmission && sampleReflection) {
			if (sample.x <= 0.5) {
				bRec.sampledComponent = 0;
				bRec.sampledType = EDeltaReflection;
				bRec.wo = reflect(bRec.wi);
				bRec.eta = 1.0f;

				return m_specularReflectance->eval(bRec.its);
			} else {
				bRec.sampledComponent = 1;
				bRec.sampledType = EDeltaTransmission;
				bRec.wo = refract(bRec.wi, cosThetaT);
				bRec.eta = 1.0f;

				return m_specularTransmittance->eval(bRec.its);
			}
		} else if (sampleReflection) {
			bRec.sampledComponent = 0;
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;

			return m_specularReflectance->eval(bRec.its);
		} else if (sampleTransmission) {
			bRec.sampledComponent = 1;
			bRec.sampledType = EDeltaTransmission;
			bRec.wo = refract(bRec.wi, cosThetaT);
			bRec.eta = 1.0f;

			return m_specularTransmittance->eval(bRec.its);
		}

		return Spectrum(0.0f);
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.0f;
	}

	int sampleComponent(const BSDFSamplingRecord &bRec, Float &pdf,
	                    Point2 &sample, const Float roughtConst) const {
    pdf = 1.f;
    return -1; // Same decision for all
  }

	std::string toString() const {
		std::ostringstream oss;
		oss << "SmoothDielectric[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<Texture> m_specularTransmittance;
	ref<Texture> m_specularReflectance;
	Float m_probBounce;
};

/* Fake glass shader -- it is really hopeless to visualize
   this material in the VPL renderer, so let's try to do at least
   something that suggests the presence of a transparent boundary */
class TestGlassShader : public Shader {
public:
  TestGlassShader(Renderer *renderer) :
		Shader(renderer, EBSDFShader) {
		m_flags = ETransparent;
	}

	Float getAlpha() const {
		return 0.3f;
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return vec3(inv_pi * cosTheta(wo));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}


	MTS_DECLARE_CLASS()
};

Shader *TestGlass::createShader(Renderer *renderer) const {
	return new TestGlassShader(renderer);
}

MTS_IMPLEMENT_CLASS(TestGlassShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(TestGlass, false, BSDF)
MTS_EXPORT_PLUGIN(TestGlass, "Smooth dielectric BSDF");
MTS_NAMESPACE_END
