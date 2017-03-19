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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <boost/algorithm/string.hpp>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/math.h>

//#include <boost/math/special_functions/fpclassify.hpp>


MTS_NAMESPACE_BEGIN

class CoronaMaterial : public BSDF {
protected:
    struct MaterialRecord {
        Spectrum diffuse;
        Spectrum reflect;
        Spectrum refract;
        //float fresnelIor;
        float refractIor;
        float ashikhminExponent;
        float reflectGlossiness;
        float refractPhongExponent;
        float refractGlossiness;
        bool idealReflect;
        bool idealRefract;
        //relative index of refraction
        float eta;

        bool exitFromMtl;
    };

    inline bool isNormalized( const Vector & v1 ) const {
        return std::fabs( dot( v1, v1 ) - 1.f ) < 1e-4f;
    }

    //TODO trochu zvektorizovat
    // n1 = incident, n2 = exitant
    //podle http://en.wikipedia.org/wiki/Fresnel_equations
    static inline float getFresnelFactor(const float cosI, const float n1, const float n2) {
        const float iorFraction = n1/n2;                
        float sinISqr = 1 - cosI*cosI;
        Assert(cosI >= -1e-5f && cosI <= 1.00001f && n1 >= 1 && n2 >= 1 && sinISqr >= -1e-5f && sinISqr <= 1.00001f);
        sinISqr = clamp( sinISqr, 0.f, 1.f );

        const float sqrtTerm = 1 - sinISqr*(iorFraction*iorFraction);
        if(sqrtTerm <= 0) {    //total internal reflection
            return 1.f;
        }

        const float cosT = std::sqrt(sqrtTerm);
        const float n1cosI = n1*cosI;
        const float n2cosI = n2*cosI;
        const float n1cosT = n1*cosT;
        const float n2cosT = n2*cosT;
        Assert(cosT >= 0.f && cosT <= 1.f);

        const float tmp1 = (n1cosI-n2cosT)/(n1cosI+n2cosT);
        const float tmp2 = (n1cosT-n2cosI)/(n1cosT+n2cosI);

        const float rS = tmp1*tmp1;
        const float rP = tmp2*tmp2;
        Assert(rS >= 0 && rS <= 1 && rP >= 0 && rP <= 1 && cosT >= 0 && cosT <= 1);
        return 0.5f * (rS + rP);
    }
    inline Spectrum spectrumMax(const Spectrum& x, const Spectrum& y) const {
        Spectrum res;
        for(int i = 0; i < x.dim; ++i) {
            res[i] = std::max(x[i], y[i]);
        }
        return res;
    }


    inline Float checkPhongExponent( const Float phongExp ) const {        
        //Assert(phongExp >= 1.f && !std::isnan(phongExp));
        return phongExp;
    }

    inline Vector idealReflect(const Vector normal, const Vector toIncident) const {
        return normalize(-toIncident + normal * (2 * dot(toIncident, normal)));    //TODO je to fakt nutny?
    }

    inline Vector sampleAshikhmin( const Point2 sample, const  Vector wi, const float ashikhminExponent ) const {
        const float phi = 2*M_PI*sample.y;
        const float sinPhi = std::sin(phi);
        const float cosPhi = std::cos(phi);

        const float cosT = std::pow(1.f - sample.x, 1.f / (ashikhminExponent + 1));
        const float sinT = math::safe_sqrt(1.f - cosT*cosT);
        const Vector h(cosPhi*sinT, sinPhi*sinT, cosT);

        return idealReflect(h, wi);        
    }

    inline Vector _idealRefract(const Vector & wi, Float eta) const {        
        Assert(isNormalized(wi));
        const float cosTheta = Frame::cosTheta(wi);
       
        const float sqrtTerm = 1.f - eta*eta*(1.f - cosTheta*cosTheta);
        if(sqrtTerm >= 0) {                        
            const Vector res = Vector(-eta*wi.x, -eta*wi.y, -math::signum(cosTheta) * std::sqrt(sqrtTerm));
            //float length = res.length();
            Assert( isNormalized(res) );
            return res;
        } else {    //total internal reflection
            const Vector res = Vector(-wi.x, -wi.y, cosTheta);
            Assert(isNormalized(res));
            return res;
        } 
    }
    

    inline Vector samplePhongRefract(const Point2 sample, const  Vector wi, const float exponent, const float eta) const {
        const float r2 = 1-sample.x;
        const float cosTheta = std::pow(r2, 1.f/(1.f + exponent));
        const float sinTheta = math::safe_sqrt(1 - cosTheta*cosTheta);
        const float sinPhi = std::sin(2*M_PI*sample.y);
        const float cosPhi = std::cos(2*M_PI*sample.y);
        const Vector result(cosPhi*sinTheta, sinPhi*sinTheta, cosTheta);
        return Frame(_idealRefract(wi, eta)).toWorld(result);
    }


    void initMaterial(const BSDFSamplingRecord& bRec, MaterialRecord& mRec) const {
        mRec.refractIor = m_refractionIOR->eval(bRec.its).average();
            
        mRec.reflectGlossiness = evalReflectionGlossiness( bRec.its );
        if( mRec.reflectGlossiness < 1e7f ) {
            mRec.ashikhminExponent = checkPhongExponent(mRec.reflectGlossiness) + 7.5f;
            mRec.idealReflect = false;
        } else {
            mRec.idealReflect = true;
        }

        mRec.refractGlossiness = evalRefractionGlossiness( bRec.its );
        if( mRec.refractGlossiness < 1e7f ) {
            mRec.refractPhongExponent = checkPhongExponent(mRec.refractGlossiness);
            mRec.idealRefract = false;
        } else {
            mRec.idealRefract = true;
        }
            
        float dotShadeNormal = Frame::cosTheta(bRec.wi);
        mRec.exitFromMtl = dotShadeNormal < 0.f;
        if(mRec.exitFromMtl) {
            dotShadeNormal = -dotShadeNormal;
        }
        mRec.eta = mRec.exitFromMtl ? mRec.refractIor : 1.f/mRec.refractIor;
            
        float prevFresnelIor = 1.f;
        float nextFresnelIor = m_reflectionIOR->eval(bRec.its).average();
        if(mRec.exitFromMtl) {
            std::swap(prevFresnelIor, nextFresnelIor);
        }
        const float fresnelTerm = this->getFresnelFactor(dotShadeNormal, prevFresnelIor, nextFresnelIor);

        Spectrum diffuse = m_diffuseColor->eval(bRec.its);
        Spectrum reflect = m_reflectionColor->eval(bRec.its) * fresnelTerm;
        Spectrum refract = m_refractionColor->eval(bRec.its);

        Spectrum temp = reflect;
        float shrinkFactor = std::min(1.f, ((Spectrum(1.f) - temp)/spectrumMax(Spectrum(1e-6f), refract)).min());
        refract *= shrinkFactor;
        temp += refract;
        shrinkFactor = std::min(1.f, ((Spectrum(1.f) - temp)/spectrumMax(Spectrum(1e-6f), diffuse)).min());
        diffuse *= shrinkFactor;
        Assert((diffuse+reflect+refract).isValid() && (diffuse+reflect+refract).max() < 1+3e-6f);
  
        //if(fresnelTerm > 0.999f) {  // total internal reflection
        //    reflect += refract;
        //    refract = Spectrum(0.f);       
        //}
        mRec.diffuse = diffuse;
        mRec.reflect = reflect;
        mRec.refract = refract;
    };

    bool isOnSameSide( const Vector & u, const Vector & v ) const {
        return Frame::cosTheta( u ) * Frame::cosTheta( v ) >= 0.f;
    }

    void evalAshikhmin(const BSDFSamplingRecord& bRec, const MaterialRecord& mRec, const float reflectionProb, Spectrum& addBsdf, float& addPdf) const {        
        const float normalization = (mRec.ashikhminExponent+1)/(8*M_PI);           
        if(isOnSameSide(bRec.wo, bRec.wi)) {
            const Vector h = normalize((bRec.wi + bRec.wo));
            const float nhDot = std::min(Frame::cosTheta(h), 1.f);
            const float hiDot = fabs(dot(h, bRec.wi));
            const float common = normalization*std::pow(nhDot, mRec.ashikhminExponent)/hiDot;
            const float pdf = common * reflectionProb;
            if(pdf > 0.f) {
                const float boundFactor = std::max(fabs(Frame::cosTheta(bRec.wi)),
                    fabs(Frame::cosTheta(bRec.wo)));
                addBsdf += mRec.reflect * (common / boundFactor);
                addPdf += pdf;
            }
        }
    }

    inline Float radianceCorrection(const MaterialRecord & mRec, const BSDFSamplingRecord & bRec) const {
        if ( bRec.mode == ERadiance && 
             (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo)) < 0.f ) {            
            //if we are following path from camera and there was not a total internal reflection
            return mRec.eta*mRec.eta;            
        }
        return 1.f;
    }

    void evalPhongRefract(const BSDFSamplingRecord& bRec, const MaterialRecord& mRec, const float refractProb, Spectrum& addBsdf, float& addPdf) const {
        const Vector idealRefract = _idealRefract(bRec.wi, mRec.eta);
        const float niDot = fabs(Frame::cosTheta(bRec.wi));
        const float normalizationFactor = (mRec.refractPhongExponent+2)/(M_PI*2);

        const float radianceTerm = radianceCorrection(mRec, bRec);

        const float refractCoef = fabs(dot(bRec.wo, idealRefract));
        const float factor = std::pow(refractCoef, mRec.refractPhongExponent);

        const float pdf = refractProb*normalizationFactor*factor;
        const float noDot = fabs(Frame::cosTheta(bRec.wo));

        if(noDot > 0 && factor > 0 && pdf > 0) {
            const float boundFactor = std::max(niDot, noDot);
            addBsdf += radianceTerm*mRec.refract*normalizationFactor*factor/boundFactor;
            addPdf += pdf;
        }
    }

    const float* discreteSelectionLinear(const float* cdfBegin, const float* cdfEnd, float& random) const {
        const float searched = float(random * cdfEnd[-1]);
        const float* res = cdfBegin;
        for(; res != cdfEnd-1; ++res) {
            if(searched < *res) {
                break;
            }
        }
        const float prevCdf = (res == cdfBegin) ? 0.f : res[-1];
        random = (searched - prevCdf)/(*res - prevCdf);
        random = std::min(random, 1-Epsilon); //FLT_EPSILON
        return res;
    }


public:
	CoronaMaterial(const Properties &props)
		: BSDF(props) {
        m_glossyType = strToGlossyType( props.getString( "glossyType", "as" ) );

		m_diffuseColor = new ConstantSpectrumTexture(
			props.getSpectrum("diffuseColor", Spectrum(1.f)));
		m_reflectionColor = new ConstantSpectrumTexture(
			props.getSpectrum("reflectionColor", Spectrum(1.f)));
        m_reflectionIOR = new ConstantSpectrumTexture(
            props.getSpectrum("reflectionIOR", Spectrum(1.52f)));
        /*m_anisotropy = new ConstantSpectrumTexture(
        props.getSpectrum("anisotropy", Spectrum(0.f)));*/
        m_reflectionGlossiness = new ConstantSpectrumTexture(
            props.getSpectrum("reflectionGlossiness", Spectrum(1.f)));
       /* m_rotation = new ConstantSpectrumTexture(
            props.getSpectrum("rotation", Spectrum(0.f)));*/

        m_refractionColor = new ConstantSpectrumTexture(
            props.getSpectrum("refractionColor", Spectrum(1.f)));
        m_refractionIOR = new ConstantSpectrumTexture(
            props.getSpectrum("refractionIOR", Spectrum(1.52f)));
        m_refractionGlossiness = new ConstantSpectrumTexture(
            props.getSpectrum("refractionGlossiness", Spectrum(1.f)));

        //m_isOneSided = props.getBoolean( "oneSided", true );
	}

	CoronaMaterial(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
        m_glossyType = static_cast<EGlossyType>( stream->readInt() );

        m_diffuseColor = static_cast<Texture *>(manager->getInstance(stream));
        m_reflectionColor = static_cast<Texture *>(manager->getInstance(stream));
        m_reflectionIOR = static_cast<Texture *>(manager->getInstance(stream));
        /*m_anisotropy = static_cast<Texture *>(manager->getInstance(stream));*/
        m_reflectionGlossiness = static_cast<Texture *>(manager->getInstance(stream));
        /*m_rotation = static_cast<Texture *>(manager->getInstance(stream));*/

        m_refractionColor = static_cast<Texture *>(manager->getInstance(stream));
        m_refractionIOR = static_cast<Texture *>(manager->getInstance(stream));
        m_refractionGlossiness = static_cast<Texture *>(manager->getInstance(stream));

        /*m_isOneSided = static_cast<bool>(manager->getInstance(stream));*/
		configure();
	}

	void configure() {        
		m_components.clear();

        m_components.push_back(EDiffuseReflection | EFrontSide
            | (m_diffuseColor->isConstant() ? 0 : ESpatiallyVarying));
        m_components.push_back(EGlossyReflection | EFrontSide |
            ((!m_reflectionColor->isConstant() || !m_reflectionIOR->isConstant() 
            || !m_reflectionGlossiness->isConstant()) ? ESpatiallyVarying : 0));
        m_components.push_back(EGlossyTransmission | EFrontSide | EBackSide | ENonSymmetric |
            ((!m_refractionColor->isConstant() || !m_refractionIOR->isConstant() 
            || !m_refractionGlossiness->isConstant()) ? ESpatiallyVarying : 0));
        m_components.push_back(EDeltaReflection | EFrontSide | EBackSide |
            ((!m_reflectionColor->isConstant() || !m_reflectionIOR->isConstant() 
            || !m_reflectionGlossiness->isConstant()) ? ESpatiallyVarying : 0));
        m_components.push_back(EDeltaTransmission | EFrontSide | EBackSide | ENonSymmetric |
            ((!m_refractionColor->isConstant() || !m_refractionIOR->isConstant() 
            || !m_refractionGlossiness->isConstant()) ? ESpatiallyVarying : 0));        

		m_usesRayDifferentials =
			m_diffuseColor->usesRayDifferentials() ||
			m_reflectionColor->usesRayDifferentials() ||
			m_refractionColor->usesRayDifferentials();

		BSDF::configure();
	}

	Spectrum getDiffuseReflectance(const Intersection &its) const {
		return m_diffuseColor->eval(its);
	}

	/// Reflection in local coordinates
	inline Vector reflect(const Vector &wi) const {
		return Vector(-wi.x, -wi.y, wi.z);
	}

    inline Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        MaterialRecord mRec;
        this->initMaterial(bRec, mRec);
        return this->eval(bRec, measure, mRec);
    }

    inline Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        MaterialRecord mRec;
        this->initMaterial(bRec, mRec);
        return this->pdf(bRec, measure, mRec);
    }

    // normala je vzdycky z-up, vstup i vystup jsou v prostoru normaly
    // vraci smer v brec v wo
    // vraci brdf * cos / pdf
    // deltu vyrobim takhle bRec.sampledType = EDeltaReflection; pdf = 1.0f; vratit to co v corone
    // naplnit: m_components v configure
	inline Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
		Assert( bRec.component == -1 );

        bool hasSpecular = true; //(bRec.typeMask & EGlossyReflection) && (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse = true;  //(bRec.typeMask & EDiffuseReflection) && (bRec.component == -1 || bRec.component == 1);
        bool hasRefract = true;
       
        //Assert( std::fabs(bRec.wi.length() - 1.f) < 1e4f );
        MaterialRecord mRec;
        this->initMaterial(bRec, mRec);
        if(!hasSpecular) {
            mRec.reflect = Spectrum(0.f);
        }
        if(!hasDiffuse) {
            mRec.diffuse = Spectrum(0.f);
        }
        if(!hasRefract) {
            mRec.refract = Spectrum(0.f);
        }
        const float totalAlbedo = mRec.reflect.average()+mRec.diffuse.average()+mRec.refract.average();
        if(totalAlbedo < 1e-4f) {
            _pdf = 1.f;
            bRec.wo = Vector3(0, 0, 1);
            return Spectrum(0.f);
        }

        float pdfs[] = {
            mRec.diffuse.average()/totalAlbedo,
            mRec.reflect.average()/totalAlbedo,
            mRec.refract.average()/totalAlbedo,
        };
        pdfs[1] += pdfs[0];
        pdfs[2] += pdfs[1];
        Assert(pdfs[2] > 1.f - Epsilon && pdfs[2] < 1.f + Epsilon);
        
        Point2 sample = _sample;
        const int index = int(this->discreteSelectionLinear(pdfs, pdfs+3, sample.x) - pdfs);
                
        switch(index) {
        case 0: //diffuse
            bRec.wo = Warp::squareToCosineHemisphere(sample);
            if ( !isOnSameSide(bRec.wo, bRec.wi) ) {
                bRec.wo.z *= -1.f;
            };
            bRec.eta = 1.0f;            
            bRec.sampledType = EDiffuseReflection;            
            Assert(isOnSameSide(bRec.wo, bRec.wi));
            break;
        case 1: //reflect
            if ( !mRec.idealReflect ) {
                bRec.wo             = sampleAshikhmin( sample, bRec.wi, mRec.ashikhminExponent );
                bRec.sampledType    = EGlossyReflection;
            } else {
                bRec.wo             = reflect( bRec.wi );
                bRec.sampledType    = EDeltaReflection;
            }                                                            
            bRec.eta = 1.0f;                                    
            break;
        case 2: //refract
            bRec.eta = mRec.eta;
            if ( !mRec.idealRefract ) {
                bRec.wo             = samplePhongRefract( sample, bRec.wi, mRec.refractPhongExponent, bRec.eta );
                bRec.sampledType    = EGlossyTransmission;
            } else {
                bRec.wo             = _idealRefract( bRec.wi, bRec.eta );
                bRec.sampledType    = EDeltaTransmission;
            }                                                
            break;
        default:
            Assert(false);
        }
        bRec.sampledComponent = getComponentNumber( bRec.sampledType );

        EMeasure measure = BSDF::getMeasure(bRec.sampledType);
        _pdf = this->pdf(bRec, measure, mRec);
        if ( _pdf == 0.f ) {
            return Spectrum( 0.f );
        }

        Spectrum result = eval(bRec, measure, mRec) / _pdf;

        Assert( result.isValid() );
        //Assert( std::fabs(bRec.wo.length() - 1.f) < 1e4f );
        return result;        
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return CoronaMaterial::sample(bRec, pdf, sample);
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "diffuseColor") {
				m_diffuseColor = static_cast<Texture *>(child);
            } 
            else if ( name == "reflectionColor" ) {
                m_reflectionColor = static_cast<Texture*>( child );
            } 
            else if (name == "reflectionIOR") {
				m_reflectionIOR = static_cast<Texture *>(child);
            }
    //        else if (name == "anisotropy") {
				//m_anisotropy = static_cast<Texture *>(child);
    //        }
            else if (name == "reflectionGlossiness") {
                m_reflectionGlossiness = static_cast<Texture *>(child);
            }
            //else if (name == "rotation") {
            //    m_rotation = static_cast<Texture *>(child);
            //}
            else if (name == "refractionColor") {
                m_refractionColor = static_cast<Texture *>(child);
            }
            //else if (name == "anisotropy") {
            //    m_anisotropy = static_cast<Texture *>(child);
            //}
            else if (name == "refractionIOR") {
                m_refractionIOR = static_cast<Texture *>(child);
            }
            else if (name == "refractionGlossiness") {
                m_refractionGlossiness = static_cast<Texture *>(child);
            } else {
				BSDF::addChild(name, child);
            }
		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
        stream->writeInt( m_glossyType );
        manager->serialize(stream, m_diffuseColor.get());
        manager->serialize(stream, m_reflectionColor.get());
        manager->serialize(stream, m_reflectionIOR.get());
        //manager->serialize(stream, m_anisotropy.get());
        manager->serialize(stream, m_reflectionGlossiness.get());
        //manager->serialize(stream, m_rotation.get());
        manager->serialize(stream, m_refractionColor.get());
        manager->serialize(stream, m_refractionIOR.get());
        manager->serialize(stream, m_refractionGlossiness.get());
        //stream->writeBool( m_isOneSided );        
	}

    Float inline toPhongExp( Float val ) const {                
        val *= 1e7f;        
        return std::max( 1.f, std::min( val, 1e7f ) );        
    }

    Float evalReflectionGlossiness( const Intersection & its ) const {
        return toPhongExp( m_reflectionGlossiness->eval( its ).average() );
    }

    Float evalRefractionGlossiness( const Intersection & its ) const {
        return toPhongExp( m_refractionGlossiness->eval( its ).average() );
    }

	Float getRoughness(const Intersection &its, int component) const {
		Assert(component >= 0 && component <= 4);
        Float exponent = 0.f;
        if ( component == 0 ) {
            // Diffuse
            return std::numeric_limits<Float>::infinity();
        } else if ( component == 1 ) {
            exponent = evalReflectionGlossiness( its );
        } else if ( component == 2 ) {
            exponent = evalRefractionGlossiness( its );
        } else if ( component == 3 || component == 4 ) {
            // Delta reflection/refraction
            return 0.f;
        }        
        /* Find the Beckmann-equivalent roughness */
        return std::sqrt( 2 / (2 + exponent) );
    }

	std::string toString() const {
		std::ostringstream oss;
		oss << "CoronaMaterial[" << endl
   			<< "  id = \"" << getID() << "\"," << endl
            << "  glossyType = " << indent( glossyTypeToStr( m_glossyType ) ) << "," << endl
			<< "  diffuseColor = " << indent(m_diffuseColor->toString()) << "," << endl
            << "  reflectionColor = " << indent(m_reflectionColor->toString()) << "," << endl
            << "  reflectionIOR = " << indent(m_reflectionIOR->toString()) << "," << endl
            //<< "  anisotropy = " << indent(m_anisotropy->toString()) << "," << endl
            << "  reflectionGlossiness = " << indent(m_reflectionGlossiness->toString()) << "," << endl
           // << "  rotation = " << indent(m_rotation->toString()) << "," << endl
            << "  refractionColor = " << indent(m_refractionColor->toString()) << "," << endl
            << "  refractionIOR = " << indent(m_refractionIOR->toString()) << "," << endl
			<< "  refractionGlossiness = " << indent(m_refractionGlossiness->toString()) << "," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
    enum EGlossyType {
        EAs,        /* Ashikmin-Shirley*/
        EWard,
        EPhong
    };

    inline EGlossyType strToGlossyType( const std::string & str ) const {
        std::string ustr = boost::to_upper_copy( str );
        if ( ustr  == "AS" ) {
            return EAs;
        }

        if ( ustr == "WARD" ) {
            return EWard;
        }

        if ( ustr == "PHONG" ) {
            return EPhong;
        }

        Log( EWarn, "Unknown glossy type. Set to default value \"as\" (i.e. Ashikmin-Shirley)." );
        return EAs;
    }

    inline std::string glossyTypeToStr( EGlossyType type ) const {
        switch( type ) {
        case EAs:
            return "as";
        case EWard:
            Assert(false);
            return "ward";
        case EPhong:
            Assert(false);
            return "phong";
        }
        Assert( false );
        return "unknown";
    }

    inline void clampFloatProp( Float & val, const std::string & name, Float min, Float max ) {
        Float tmp = val;
        val = std::max( min, std::min( val, max ) );
        if ( val != tmp ) {
            Log( EWarn, "Property %s with value %f was clamped to %f." , name.c_str(), tmp, val );
        }
    }

    static int getComponentNumber( unsigned int type ) {
        switch(type) {
        case EDiffuseReflection:
            return 0;
        case EGlossyReflection:
            return 1;
        case EGlossyTransmission:
            return 2;
        case EDeltaReflection:
            return 3;
        case EDeltaTransmission:
            return 4;
        default:
            Log( EError, "Cannot determine components number. Reason: unknown components type." );
        }

        return -1;
    }    


    // normala je vzdycky z-up, vstup i vystup jsou v prostoru normaly
    inline Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure, MaterialRecord mRec) const {
        Assert( isNormalized( bRec.wi ) );
        Assert( isNormalized( bRec.wo ) );
        Assert( bRec.component == -1 );

        Float result = 0.f;
        EvalComponentsRecord eRec(bRec, measure);

        const float totalAlbedo = std::max(0.001f, mRec.reflect.average()+mRec.diffuse.average()+mRec.refract.average());
        const float diffuseProb = mRec.diffuse.average()/totalAlbedo;
        const float reflectProb = mRec.reflect.average()/totalAlbedo;
        const float refractProb = mRec.refract.average()/totalAlbedo;

        bool oneSide = isOnSameSide(bRec.wo, bRec.wi);
        if (eRec.hasDiffuseReflection && diffuseProb > 0 && oneSide) {
            result += diffuseProb * fabs( Warp::squareToCosineHemispherePdf(bRec.wo) );
        }

        if(eRec.hasGlossyReflection && reflectProb > 0 && !mRec.idealReflect && oneSide ) {
            Spectrum dummy;
            evalAshikhmin(bRec, mRec, reflectProb, dummy, result);
        } else if(eRec.hasDeltaReflection && reflectProb > 0 && oneSide && std::fabs(dot(reflect(bRec.wi), bRec.wo)-1.f) <= DeltaEpsilon) {
            result += reflectProb;
        }

        if(eRec.hasGlossyRefraction && refractProb > 0 && !mRec.idealRefract) {
            Spectrum dummy;
            evalPhongRefract(bRec, mRec, refractProb, dummy, result);
        } else if(eRec.hasDeltaRefraction && refractProb > 0 && std::fabs(dot(_idealRefract(bRec.wi, mRec.eta),bRec.wo)-1.f) <= DeltaEpsilon) {
            result += refractProb;
        }

        Assert( result >= 0.f );
        //Assert( boost::math::fpclassify(result) != FP_SUBNORMAL );

        return result;
    }


    // normala je vzdycky z-up, vstup i vystup jsou v prostoru normaly
    // vraci bsdf*cos
    // nahodny cisla vraci bRec.sampler
    inline Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure, MaterialRecord mRec) const {
        Assert( bRec.component == -1 );

        Spectrum result( 0.f );
        EvalComponentsRecord eRec(bRec, measure);

        bool oneSide = isOnSameSide(bRec.wo, bRec.wi);
        if (eRec.hasDiffuseReflection && oneSide) {
            result += mRec.diffuse * INV_PI;
        }   

        if(eRec.hasGlossyReflection && !mRec.idealReflect && oneSide) {
            float dummy;
            evalAshikhmin(bRec, mRec, 1.f, result, dummy);
        } else if(eRec.hasDeltaReflection && oneSide && std::fabs(dot(reflect(bRec.wi), bRec.wo)-1.f) <= DeltaEpsilon) {
            result += ( mRec.reflect / std::max( (Float)fabs( Frame::cosTheta(bRec.wo) ), 1e-6f ) );
        }

        if(eRec.hasGlossyRefraction && !mRec.idealRefract) {
            float dummy;
            evalPhongRefract(bRec, mRec, 1.f, result, dummy);
        } else if (eRec.hasDeltaRefraction && std::fabs(dot(_idealRefract(bRec.wi, mRec.eta),bRec.wo)-1.f) <= DeltaEpsilon) {
            const float radianceTerm = radianceCorrection(mRec, bRec);
            result += (mRec.refract * (radianceTerm / std::max( (Float)fabs( Frame::cosTheta(bRec.wo) ), 1e-6f )));
        }

        result *= fabs( Frame::cosTheta(bRec.wo) );
        Assert( result.isValid() );
        return result;
    }

    struct EvalComponentsRecord {
        bool hasDeltaReflection;
        bool hasDeltaRefraction;
        bool hasGlossyReflection;
        bool hasGlossyRefraction;
        bool hasDiffuseReflection;

        EvalComponentsRecord( const BSDFSamplingRecord & bRec, EMeasure measure ) {
            hasDeltaReflection = (bRec.typeMask & EDeltaReflection) && 
                (bRec.component == -1 || bRec.component == getComponentNumber(EDeltaReflection)) && 
                measure == EDiscrete;

            hasDeltaRefraction = (bRec.typeMask & EDeltaTransmission) && 
                (bRec.component == -1 || bRec.component == getComponentNumber(EDeltaTransmission)) && 
                measure == EDiscrete;

            hasGlossyReflection = (bRec.typeMask & EGlossyReflection) 
                && (bRec.component == -1 || bRec.component == getComponentNumber(EGlossyReflection))
                && measure == ESolidAngle;

            hasGlossyRefraction = (bRec.typeMask & EGlossyTransmission) 
                && (bRec.component == -1 || bRec.component == getComponentNumber(EGlossyTransmission))
                && measure == ESolidAngle;

            hasDiffuseReflection = (bRec.typeMask & EDiffuseReflection) 
                && (bRec.component == -1 || bRec.component == getComponentNumber(EDiffuseReflection))
                && measure == ESolidAngle;
        }
    };

private:
    EGlossyType m_glossyType; // inactive

	ref<Texture> m_diffuseColor;
	ref<Texture> m_reflectionColor;
    ref<Texture> m_refractionColor;


    ref<Texture> m_reflectionIOR;
    ref<Texture> m_reflectionGlossiness;
    ref<Texture> m_refractionIOR;
    ref<Texture> m_refractionGlossiness;


    //ref<Texture> m_anisotropy;
    
    //ref<Texture> m_rotation;

    
    //bool m_isOneSided;
};

// ================ Hardware shader implementation ================

/* Only diffuse shader */
class SmoothDiffuseShader : public Shader {
public:
    SmoothDiffuseShader(Renderer *renderer, const Texture *reflectance)
        : Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
            m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
    }

    bool isComplete() const {
        return m_reflectanceShader.get() != NULL;
    }

    void cleanup(Renderer *renderer) {
        renderer->unregisterShaderForResource(m_reflectance.get());
    }

    void putDependencies(std::vector<Shader *> &deps) {
        deps.push_back(m_reflectanceShader.get());
    }

    void generateCode(std::ostringstream &oss,
        const std::string &evalName,
        const std::vector<std::string> &depNames) const {
            oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
                << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
                << "    	return vec3(0.0);" << endl
                << "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
                << "}" << endl
                << endl
                << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
                << "    return " << evalName << "(uv, wi, wo);" << endl
                << "}" << endl;
    }

    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_reflectance;
    ref<Shader> m_reflectanceShader;
};

Shader *CoronaMaterial::createShader(Renderer *renderer) const {
    return new SmoothDiffuseShader( renderer, m_diffuseColor.get() );
}

MTS_IMPLEMENT_CLASS(SmoothDiffuseShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(CoronaMaterial, false, BSDF)
MTS_EXPORT_PLUGIN(CoronaMaterial, "Corona material");
MTS_NAMESPACE_END
