#pragma once

#include <mitsuba/core/spectrum.h>

MTS_NAMESPACE_BEGIN

// Holds list of gather points at given pixel
template <typename T>
class PixelData
{
 private:
  Spectrum* tempFlux;
  Float * tempPhi;
  Float * tempPhiNSamples;

 public:
	// Gather point associated data
	Point2i pos;  // Position on the image (pixel wise)
	Spectrum fluxDirect; // Holds flux from path-tracing for given pixel
	Spectrum flux; // Hold flux from other techniques
	Float scale; // Radius scale
	Float radius; // Radius of gather points in this list
	Float N; // Number of gathered photons (for SPPM only)
	size_t nPhotons; // Number of gathered photons without radius reduction

	// === Sync thread data - only 1 alloc per list!
	Float * tempM; 

	// Note: the two level are hard coded for now

	// Max thread information
	int maxThread;
	// Directly visible emission
	Spectrum emission;
	// Cumulated importance
	Float cumulImportance;

	// Maximum number of gather points per one list (only those that store tempM, tempPhi)
	static int maxGatherPoints;

	// If true, maxGatherPoints will limit the number of gather points used for merging
	static bool limitMaxGatherPoints;

	// True if phi statistics should be used
	static bool usePhiStatistics;

	// True if we need to accumulate different chains
	static int nbChains;

	static int allocTempSize(int mT)
	{
		return mT * (
		        maxGatherPoints * sizeof(Float) +
		        ((int)usePhiStatistics) * maxGatherPoints * nbChains * sizeof(Float) * 2 +
		        ((int)usePhiStatistics) * maxGatherPoints * nbChains * sizeof(Float) +
		        (nbChains)*sizeof(Spectrum));
	}

	// Alloc gather point inner structures
	void allocTemp(int mT, char * & allocPtr)
	{
		if (maxThread != -1) // Already allocated?
			return;
		tempFlux = (Spectrum *)allocPtr;
		allocPtr += sizeof(Spectrum)* mT * nbChains;
		tempM = (Float *)allocPtr;
		allocPtr += sizeof(Float)* mT * maxGatherPoints;
		if ( usePhiStatistics ) {
      tempPhiNSamples = (Float*)allocPtr;
      allocPtr += sizeof(Float) * mT * maxGatherPoints * nbChains;

			tempPhi = (Float *)allocPtr;
			allocPtr += sizeof(Float)* mT * maxGatherPoints * nbChains * 2;
		}
		maxThread = mT;
	}

	/// Reset the temp value associated to the gather point.
	inline void resetTemp() {
		memset(tempFlux, 0, nbChains*sizeof(Spectrum)* maxThread);
		memset(tempM, 0, sizeof(Float)* maxThread * maxGatherPoints);
		if ( usePhiStatistics ) {
			memset(tempPhi, 0, sizeof(Float)* maxThread * maxGatherPoints * nbChains * 2);
			memset(tempPhiNSamples, 0, sizeof(Float)* maxThread * maxGatherPoints * nbChains);
		}
	}

	inline void addFlux(const Spectrum flux, const int idThread, const int idImportance) {
	    if(idImportance > nbChains) {
	        SLog(EError, "Flux out bound");
	    }
	    tempFlux[idThread*nbChains + idImportance] += flux;
	}

	inline Spectrum getFlux(const int idImportance) const {
	    if(idImportance > nbChains) {
           SLog(EError, "Flux request out bound");
       }
	   Spectrum flux(0.f);
	   for (int idThread = 0; idThread < maxThread; idThread++) {
	       flux += tempFlux[idThread*nbChains + idImportance];
       }
	   return flux;
	}

	inline void addPhi(const Float v,
	                   const Float uniqueCount,
	                   const int idThread, const int idImportance, const int idGPS, bool isOdd) {
       if(idImportance > nbChains) {
           SLog(EError, "Flux out bound");
       }

       tempPhi[maxThread * nbChains * idGPS * 2 + idThread*nbChains *2 + idImportance *2 + (int)isOdd] += v;
       tempPhiNSamples[maxThread * nbChains * idGPS + idThread*nbChains + idImportance] += uniqueCount;
    }

	inline Float getPhi(const int idImportance, const int idGPS, bool isOdd) const {
     if(idImportance > nbChains) {
         SLog(EError, "Flux request out bound");
     }
     Float flux(0.f);
     for (int idThread = 0; idThread < maxThread; idThread++) {
       if(isOdd) {
         flux += tempPhi[maxThread * nbChains * idGPS * 2 + idThread*nbChains*2 + idImportance * 2];
       } else {
         flux += tempPhi[maxThread * nbChains * idGPS * 2 + idThread*nbChains*2 + idImportance * 2 + 1];
       }
     }
     return flux;
  }

	inline Float getNSamplePhi(const int idImportance, const int idGPS) {
	  if(idImportance > nbChains) {
      SLog(EError, "Flux request out bound");
    }
    Float nSamples = 0;
    for (int idThread = 0; idThread < maxThread; idThread++) {
      nSamples += tempPhiNSamples[maxThread * nbChains * idGPS + idThread*nbChains + idImportance];
    }
    return nSamples;
	}

	inline PixelData():
    tempFlux(NULL),
    tempPhi(NULL),
		pos(-1,-1),
		fluxDirect(0.f),
		N(0.f),
		nPhotons(0),
		tempM(NULL),
		maxThread(-1),
		emission(0.f),
		cumulImportance(0.f),
		m_innerList(NULL),
		m_size(0),
		m_allocated(0)
	{
	}

	inline void release()
	{
		delete [] m_innerList;
		m_innerList = NULL;
		m_size = m_allocated = 0;
	}

	inline ~PixelData()
	{
		release();
	}

	inline size_t size() const
	{
		return m_size;
	}

	inline bool empty() const
	{
		return m_size == 0;
	}

	inline void push_back(const T & elem)
	{
		if ( m_size == m_allocated )
			grow( std::max(m_allocated,(size_t)10) << 1 );
		m_innerList[m_size++] = elem;
	}

	inline void push_back()
	{
		if ( m_size == m_allocated )
			grow( std::max(m_allocated,(size_t)10) << 1 );
		++m_size;
	}

	/// Remove element from back
	inline void pop_back()
	{
		--m_size;
	}

	/// Clear array, but don't free memory
	inline void clear()
	{
		m_size = 0;
	}

	/// Access functions
	inline T & operator[](int i)
	{
		assert(m_size > i);
		return m_innerList[i];
	}

	inline const T & operator[](int i) const
	{
		assert(m_size > i);
		return m_innerList[i];
	}

	inline T & front()
	{
		assert(m_size > 0);
		return m_innerList[0];
	}

	inline const T & front() const
	{
		assert(m_size > 0);
		return m_innerList[0];
	}

	inline T & back()
	{
		assert(m_size > 0);
		return m_innerList[m_size - 1];
	}

	inline const T & back() const
	{
		assert(m_size > 0);
		return m_innerList[m_size - 1];
	}


	// Iterators


	class iterator
	{
		friend class PixelData;
		inline iterator(T * ptr) :m_ptr(ptr)
		{
		}
	public:
		inline iterator() :
		  m_ptr(0)
		  {
		  }

		  inline iterator operator++()
		  {
			  ++m_ptr;
			  return *this;
		  }

		  inline iterator operator++(int)
		  {
			  iterator tmp = *this;
			  ++m_ptr;
			  return tmp;
		  }

		  inline iterator operator--()
		  {
			  --m_ptr;
			  return *this;
		  }

		  inline iterator operator--(int)
		  {
			  iterator tmp = *this;
			  --m_ptr;
			  return tmp;
		  }

		  inline const T* operator->() const
		  {
			  return m_ptr;
		  }

		  inline T* operator->()
		  {
			  return m_ptr;
		  }

		  inline const T & operator*() const
		  {
			  return *m_ptr;
		  }

		  inline T & operator*()
		  {
			  return *m_ptr;
		  }

		  inline bool operator==(const iterator &aIt)
		  {
			  return m_ptr == aIt.m_ptr;
		  }

		  inline bool operator!=(const iterator &aIt)
		  {
			  return m_ptr != aIt.m_ptr;
		  }
	private:
		T * m_ptr;
	};

	class const_iterator
	{
		friend class PixelData;
		inline const_iterator(const T * ptr) :m_ptr(ptr)
		{
		}
	public:
		inline const_iterator() :
		  m_ptr(0)
		  {
		  }

		  inline const_iterator operator++()
		  {
			  ++m_ptr;
			  return *this;
		  }

		  inline const_iterator operator++(int)
		  {
			  iterator tmp = *this;
			  ++m_ptr;
			  return tmp;
		  }

		  inline const_iterator operator--()
		  {
			  --m_ptr;
			  return *this;
		  }

		  inline const_iterator operator--(int)
		  {
			  iterator tmp = *this;
			  --m_ptr;
			  return tmp;
		  }

		  inline const T *  operator->() const
		  {
			  return m_ptr;
		  }

		  inline const T & operator*() const
		  {
			  return *m_ptr;
		  }

		  inline bool operator==(const const_iterator &aIt)
		  {
			  return m_ptr == aIt.m_ptr;
		  }

		  inline bool operator!=(const const_iterator &aIt)
		  {
			  return m_ptr != aIt.m_ptr;
		  }
	private:
		const T * m_ptr;
	};

	/// Iterator functions

	inline iterator begin()
	{
		return iterator(m_innerList);
	}

	inline iterator end()
	{
		return iterator(m_innerList + m_size);
	}

	inline const_iterator begin() const
	{
		return const_iterator(m_innerList);
	}

	inline const_iterator end() const
	{
		return const_iterator(m_innerList + m_size);
	}

	inline const_iterator cbegin() const
	{
		return const_iterator(m_innerList);
	}

	inline const_iterator cend() const
	{
		return const_iterator(m_innerList + m_size);
	}

	// Reverse
	inline iterator rbegin()
	{
		return iterator(m_innerList + m_size - 1);
	}

	inline iterator rend()
	{
		return iterator(m_innerList - 1);
	}

	inline const_iterator rbegin() const
	{
		return const_iterator(m_innerList + m_size - 1);
	}

	inline const_iterator rend() const
	{
		return const_iterator(m_innerList - 1);
	}

	inline const_iterator crbegin() const
	{
		return const_iterator(m_innerList + m_size - 1);
	}

	inline const_iterator crend() const
	{
		return const_iterator(m_innerList - 1);
	}
private:

	inline void grow(size_t newSize)
	{
		assert(newSize > m_allocated);
		T * tmp = new T[newSize];
		if (m_innerList != NULL) {
			memcpy(tmp,m_innerList,sizeof(T) * m_size);
			delete [] m_innerList;
		}
		m_innerList = tmp;
		m_allocated = newSize;
	}

	T * m_innerList;
	size_t m_size;
	size_t m_allocated;
};

MTS_NAMESPACE_END

