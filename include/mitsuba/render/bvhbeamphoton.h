/*
 * bvhbeamphoton.h
 *
 *  Created on: 20 août 2012
 *      Author: Mickael Ribardiere
 */


#ifndef BVHBEAMPHOTON_H_
#define BVHBEAMPHOTON_H_


#include <mitsuba/render/beamkdtree.h>

MTS_NAMESPACE_BEGIN

struct SubBeamPhoton{
	Float tMin;
	Float tMax;
	int indexBeam;
};

struct BVHNode {
	Point pos; // center of the bounding sphere
	Float radius;
	SubBeamPhoton *subBeam;

	BVHNode *first;
	BVHNode *second;

	BVHNode(){
		first = NULL;
		second = NULL;
		subBeam = NULL;
	}
	~BVHNode(){
		if(first)
			delete first;
		if(second)
			delete second;
	}

	// === intersection with the bounding sphere
	bool rayIntersect(const Ray &ray) const {
		Vector o = ray.o - pos;
		Float A = ray.d.x*ray.d.x + ray.d.y*ray.d.y + ray.d.z*ray.d.z;
		Float B = 2 * (ray.d.x*o.x + ray.d.y*o.y + ray.d.z*o.z);
		Float C = o.x*o.x + o.y*o.y + o.z*o.z - radius*radius;

		if (A == 0) {
			if (B != 0)
				return true;

			return false;
		}

		Float discrim = B*B - 4.0f*A*C;

		if (discrim < 0)
			return false;

		return true;
	}
};

class MTS_EXPORT_RENDER BVHBeamPhoton {
public:
	BVHBeamPhoton(void);
	~BVHBeamPhoton();

	void push_back(BeamPhoton *beam) {
		m_beams.push_back(beam);
	}
	/// Reserve a certain amount of memory for the array
	inline void reserve(size_t size) {m_beams.reserve(size);}
	/**
	 * \brief Initialize the kdtree: if it was already initialized, the existing kdtree is first destroyed.
	 */
	void initialize(Float stepCut);
	/// Return the size of the kd-tree
	inline size_t size() const {return m_beams.size();}
	/// Clear the kd-tree array
	void clear(void);
	/// Compute Lvi in case of  Progressive Photon beam method
	void getLvi(Beam *bView, int maxDepth);
	void printStatistics(void);
	inline void updateRadius(Float radius) {m_radius = radius;}

private:
	int getElement(const std::vector<TransmittanceCell> &trans, Float t);
	inline Float K2(Float sqrParam) const {
		Float tmp = 1-sqrParam;
		return (3.f/M_PI) * tmp * tmp;
	}

	std::vector<BeamPhoton*> m_beams;
	std::vector<SubBeamPhoton*> m_subBeams;
	Float m_radius;
	BVHNode* m_root;
};

MTS_NAMESPACE_END

#endif /* BVHBEAMPHOTON_H_ */
