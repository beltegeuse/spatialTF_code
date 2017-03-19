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


#if !defined(__PSSMLT_SAMPLER_H)
#define __PSSMLT_SAMPLER_H

#include <mitsuba/render/sampler.h>
#include <mitsuba/core/random.h>
#include "msppm.h"

MTS_NAMESPACE_BEGIN

/**
 * Sampler implementation as described in
 * 'A Simple and Robust Mutation Strategy for the
 * Metropolis Light Transport Algorithm' by Kelemen et al.
 */
class MSPPMSampler : public Sampler {
public:
	// Construct a new MLT sampler
	MSPPMSampler(const MSPPMConfiguration &conf);

	/**
	 * \brief Construct a new sampler, which operates on the
	 * same random number generator as \a sampler.
	 */
	MSPPMSampler(MSPPMSampler *sampler);

	/// Unserialize from a binary data stream
	MSPPMSampler(Stream *stream, InstanceManager *manager);

	/// Set up the internal state
	void configure();

	/// Serialize to a binary data stream
	void serialize(Stream *stream, InstanceManager *manager) const;

	/// Set whether the current step should be large
	inline void setLargeStep(bool value) { m_largeStep = value; }

	/// Check if the current step is a large step
	inline bool isLargeStep() const { return m_largeStep; }

	/// Retrieve the next component value from the current sample
	virtual Float next1D();

	/// Retrieve the next two component values from the current sample
	virtual Point2 next2D();

	/// Return a string description
	virtual std::string toString() const;

	/// 1D mutation routine
	inline Float mutate(Float value) {
		Float sample = m_random->nextFloat();
		bool add;

		if (sample < 0.5f) {
			add = true;
			sample *= 2.0f;
		} else {
			add = false;
			sample = 2.0f * (sample - 0.5f);
		}
		Float dv = 0.f;

		dv = powf(sample,(1.f/(amcmc->theta*amcmc->scale)) + 1);


		if (add) {
			value += dv;
			if (value > 1)
				value -= 1;
		} else {
			value -= dv;
			if (value < 0)
				value += 1;
		}

		return value;

	}

	/// Return a primary sample
	Float primarySample(size_t i);

	/// Reset (& start with a large mutation)
	void reset();

	/// Accept a mutation
	void accept();

	/// Reject a mutation
	void reject();

	/// Replace the underlying random number generator
	inline void setRandom(Random *random) { m_random = random; }

	/// Return the underlying random number generator
	inline ref<Random> getRandom() { return m_random; }

	/* The following functions do nothing in this implementation */
	virtual void advance() { }
	virtual void generate(const Point2i &pos) { }

	/* The following functions are unsupported by this implementation */
	void request1DArray(size_t size) { Log(EError, "request1DArray(): Unsupported!"); }
	void request2DArray(size_t size) { Log(EError, "request2DArray(): Unsupported!"); }
	void setSampleIndex(size_t sampleIndex) {
		//Log(EError, "setSampleIndex(): Unsupported!"); }
	}
	ref<Sampler> clone();

	void updateMutationScale() {
		Float currAcc = amcmc->nbAccMut / (Float) amcmc->nbMut;
		amcmc->scale = amcmc->scale + ((currAcc - m_AStar) / (amcmc->nbMut));
		amcmc->scale = std::max(amcmc->scale, 0.0001f);
	}

	void reinitAMCMC() {
	  amcmc->theta = 0.8f;
	  amcmc->scale = 1.f;
	  amcmc->nbMut = 0;
    amcmc->nbAccMut = 0;
	}

	Float getScale() {
	  return amcmc->scale;
	}

	Float getAccRate() {
	  return amcmc->nbAccMut / (Float) amcmc->nbMut;
	}

	MTS_DECLARE_CLASS()

	// TODO
    struct AMCMC {
        size_t nbMut, nbAccMut;
        Float theta, scale;
    };

	AMCMC * amcmc;
protected:
	/// Virtual destructor
	virtual ~MSPPMSampler();
protected:
	struct SampleStruct {
		Float value;
		size_t modify;

		inline SampleStruct(Float value) : value(value), modify(0) { }
	};

	ref<Random> m_random;
	Float m_AStar;
	bool m_largeStep;
	std::vector<std::pair<size_t, SampleStruct> > m_backup;
	std::vector<SampleStruct> m_u;
	size_t m_time, m_largeStepTime;
	bool m_useKelemenMut;
};

MTS_NAMESPACE_END

#endif /* __PSSMLT_SAMPLER_H */
