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

#include "msppm_sampler.h"

MTS_NAMESPACE_BEGIN

MSPPMSampler::MSPPMSampler(const MSPPMConfiguration &config) : Sampler(Properties()) {
  if(config.referenceMod) {
    Log(EInfo, "Use random initialization");

    // Seed the system random generator
    // by the timing of the computation
    srand (10/*time(NULL)*/);

    // Generate the uint64 for MTS
    // random initalization
    uint64_t seed;
    seed = rand();
    seed = (seed << 32) | rand();

    // Print it to be sure that it change
    // at each rendering
    Log(EInfo, "Random seed: %llu", seed);

    // Create the random number generator
    m_random = new Random(seed);
  } else {
    m_random = new Random();
  }

    amcmc = new AMCMC;

    amcmc->theta = 0.8f;
    amcmc->scale = 1.f;
    amcmc->nbMut = 0;
    amcmc->nbAccMut = 0;

	m_AStar = config.AStar;
	m_useKelemenMut = true; // TODO
	configure();
}

MSPPMSampler::MSPPMSampler(MSPPMSampler *sampler) : Sampler(Properties()),
	m_random(sampler->m_random) {

    amcmc = new AMCMC;

    amcmc->theta = 0.8f;
    amcmc->scale = 1.f;
    amcmc->nbMut = 0;
    amcmc->nbAccMut = 0;

	m_AStar = sampler->m_AStar; //< Copy AStar

	m_useKelemenMut = sampler->m_useKelemenMut;
	configure();
}

MSPPMSampler::MSPPMSampler(Stream *stream, InstanceManager *manager)
	: Sampler(stream, manager) {
	m_random = static_cast<Random *>(manager->getInstance(stream));
	//m_s1 = stream->readFloat();
	//m_s2 = stream->readFloat();
	m_useKelemenMut = stream->readBool();
	configure();
}

void MSPPMSampler::serialize(Stream *stream, InstanceManager *manager) const {
	Sampler::serialize(stream, manager);
	manager->serialize(stream, m_random.get());
//	stream->writeFloat(m_s1);
//	stream->writeFloat(m_s2);
	stream->writeBool(m_useKelemenMut);
}

void MSPPMSampler::configure() {
    amcmc->theta = 0.8f;
    amcmc->scale = 1.f;
	//m_AStar = m_config.AStar;
	//	m_logRatio = -math::fastlog(m_s2/m_s1);
	m_time = 0;
	m_largeStepTime = 0;
	m_largeStep = false;
	m_sampleIndex = 0;
	m_sampleCount = 0;
}

MSPPMSampler::~MSPPMSampler() { }

void MSPPMSampler::accept() {
	if (m_largeStep) {
		m_largeStepTime = m_time;
	}
	m_time++;
	m_backup.clear();
	m_sampleIndex = 0;

	if(!m_largeStep) {
	    amcmc->nbMut++;
	    amcmc->nbAccMut++;
		updateMutationScale();
	}
}

void MSPPMSampler::reset() {
	m_time = m_sampleIndex = m_largeStepTime = 0;
	m_u.clear();
	amcmc->scale = 1.f;
	amcmc->nbMut = 0;
	amcmc->nbAccMut = 0;
}

void MSPPMSampler::reject() {
	for (size_t i=0; i<m_backup.size(); ++i)
		m_u[m_backup[i].first] = m_backup[i].second;
	m_backup.clear();
	m_sampleIndex = 0;

	if(!m_largeStep) {
	    amcmc->nbMut++;
		updateMutationScale();
	}
}

Float MSPPMSampler::primarySample(size_t i) {
	// === Lazy update
	while (i >= m_u.size())
		m_u.push_back(SampleStruct(m_random->nextFloat()));

	// === Overwise test number
	if (m_u[i].modify < m_time) {
		if (m_largeStep) {
			// Generate large step
			m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));
			m_u[i].modify = m_time;
			m_u[i].value = m_random->nextFloat();
		} else {
			// === If this random number is reseted by a large step
			// but we didn't alread do it.
			// we generate the random number for large step
			if (m_u[i].modify < m_largeStepTime) {
				m_u[i].modify = m_largeStepTime;
				m_u[i].value = m_random->nextFloat();
			}

			// === Rattraper le retard dans la mutation
			// Jusqu'a t-1
			while (m_u[i].modify + 1 < m_time) {
				m_u[i].value = mutate(m_u[i].value);
				m_u[i].modify++; // Said new value in modify
			}

			// === Save it
			m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));

			// Mut it
			m_u[i].value = mutate(m_u[i].value);
			m_u[i].modify++;
		}
	}

	return m_u[i].value;
}

ref<Sampler> MSPPMSampler::clone() {
	ref<MSPPMSampler> sampler = new MSPPMSampler(this);
	sampler->m_sampleCount = m_sampleCount;
	sampler->m_sampleIndex = m_sampleIndex;
	sampler->m_random = new Random(m_random);
	return sampler.get();
}

Float MSPPMSampler::next1D() {
	return primarySample(m_sampleIndex++);
}

Point2 MSPPMSampler::next2D() {
	/// Enforce a specific order of evaluation
	Float value1 = primarySample(m_sampleIndex++);
	Float value2 = primarySample(m_sampleIndex++);
	return Point2(value1, value2);
}

std::string MSPPMSampler::toString() const {
	std::ostringstream oss;
	oss << "PSSMLTSampler[" << endl
		<< "  sampleCount = " << m_sampleCount << endl
		<< "]";
	return oss.str();
}

MTS_IMPLEMENT_CLASS_S(MSPPMSampler, false, Sampler)
MTS_NAMESPACE_END
