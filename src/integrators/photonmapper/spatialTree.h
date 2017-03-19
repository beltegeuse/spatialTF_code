#pragma once

#include <mitsuba/mitsuba.h>
#include <vector>

// MTS includes
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>

// STL includes
#include <set>
#include <list>

#include "gpaccel.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////
// Other structures
//////////////////////////////////

class UpdateRequest {
public:
	int NbMutatedPath;
	int NbMutatedAcc;

	UpdateRequest() : NbMutatedPath(0),
	    NbMutatedAcc(0)
	{}
};

class CoreNodeData {
public:
	size_t NbMutatedPath;
	size_t NbMutatedAcc;

	CoreNodeData() {
		NbMutatedPath = 0;
		NbMutatedAcc = 0;
	}

	void reset() {
		NbMutatedPath = 0;
		NbMutatedAcc = 0;
	}

	void add(const CoreNodeData& o) {
		NbMutatedPath += o.NbMutatedPath;
		NbMutatedAcc += o.NbMutatedAcc;
	}

	void update(const UpdateRequest& u) {
		NbMutatedPath += u.NbMutatedPath;
		NbMutatedAcc += u.NbMutatedAcc;
	}

	Float accRate() {
		return NbMutatedAcc / (Float) NbMutatedPath;
	}

	void normalize(size_t nCores) {
	}

};

template <class T>
class SpatialTree : public SerializableObject {
public:
	//SpatialTree(size_t nCores, GPAccel<GatherPoint>::Type& gpAccel)
  SpatialTree(size_t nCores, GatherBlocks & gps, GPAccel<GatherPoint>::Type& gpAccel,
              bool usePointKDTree, int depthSplit = 8)
  {
		// Copy the GP kd tree
		m_nodes.clear();
		int expectedCopy = (int)(pow(2.f, depthSplit)-1.f);

		if(usePointKDTree) {
      GPAccelPointKD<GatherPoint>* gpAccelLocal = new GPAccelPointKD<GatherPoint>(gps);
      gpAccelLocal->copyKDTree(m_nodes, depthSplit);
      delete gpAccelLocal;
		} else {
		  gpAccel.copyKDTree(m_nodes, depthSplit);
		}

		SLog(EInfo, "Copied %i KD-Tree node, Expected: %i", m_nodes.size(), expectedCopy);

		// Reverse and initialize the Core data
		m_coreData.reserve(m_nodes.size());
		m_globalData.reserve(m_nodes.size());
		for(size_t i = 0; i < m_nodes.size(); i++) {
			std::vector<T> coresData;
			coresData.reserve(nCores);
			for(size_t j =0; j < nCores; j++) {
				T data;
				coresData.push_back(data);
			}
			m_coreData.push_back(coresData);
			m_globalData.push_back(T()); //< No need to clean
		}

		// Assign ID for each node of the copy of the Kd-tree
		// to get the relation with the data
		for(size_t i = 0; i < m_nodes.size(); i++) {
			m_nodes[i]->idData = (int)i;
		}

		int nbLevel = nbNodeLevel(depthSplit-1);
		SLog(EInfo, "Number of the Leaf of KD-tree: %i", nbLevel);
	}

	SpatialTree(Stream *stream, InstanceManager *manager) :
		SerializableObject(stream, manager) {
		SLog(EError, "No serialization !");
	}

	virtual void serialize(Stream *stream, InstanceManager *manager) const {
		SLog(EError, "No serialization !");
	}

	template <class R>
	void updateStats(const PhotonSplattingList& list, R& req, int coreID) {
		for(size_t i = 0; i < list.photons.size(); i++) {
			updateStats<R>(list.photons[i].its.p, req, coreID);
		}
	}

	template <class R>
	void updateStats(const Point& p, R& req, int coreID) {
		GPNodeCopy* currNode = m_nodes[0];

		while(true) {
			const Float splitVal = (Float) currNode->split;
			const int axis = currNode->axis;

			// Update the node traversed
			if(coreID == -1) {
				m_globalData[currNode->idData].update(req);
			} else {
				m_coreData[currNode->idData][coreID].update(req);
			}

			if(currNode->isLeaf) {
				return;
			}

			// Choose child node ...
			if (p[axis] <= splitVal) {
				if(currNode->leftChild == -1) {
					return;
				}

				currNode = m_nodes[currNode->leftChild];
			} else {
				if(currNode->rightChild == -1) {
					return;
				}

				currNode = m_nodes[currNode->rightChild];
			}
		}

	}

	void getDataIDLevelLocal(int maxDepth, GPNodeCopy* node, std::vector<int>& ids) {
		if(maxDepth == 0) {
		  //std::cout << "FUCK\n";
			ids.push_back(node->idData);
			return;
		}

		if(node->isLeaf) {
			ids.push_back(node->idData);
			return;
		}

		if(node->leftChild != -1) {
			getDataIDLevelLocal(maxDepth-1, m_nodes[node->leftChild], ids);
		}
		if(node->rightChild != -1) {
			getDataIDLevelLocal(maxDepth-1, m_nodes[node->rightChild], ids);
		}
	}

	std::vector<int> getDataIDLevel(int maxDepth) {
		std::vector<int> res;
		getDataIDLevelLocal(maxDepth, m_nodes[0], res);
		return res;
	}

	int nbNodeLevelLocal(int maxDepth, GPNodeCopy * node) {
		if(maxDepth == 0) {
			return 1;
		}

		if(node->isLeaf) {
			return 1;
		}

		// Count for each child ...
		int childNb = 0;
		if(node->leftChild != -1) {
			childNb += nbNodeLevelLocal(maxDepth-1, m_nodes[node->leftChild]);
		} else {
			//SLog(EInfo, "No left child");
		}

		if(node->rightChild != -1) {
			childNb += nbNodeLevelLocal(maxDepth-1, m_nodes[node->rightChild]);
		} else {
			//SLog(EInfo, "No right child");
		}

		return childNb;
	}

	int nbNodeLevel(int maxDepth) {
		return nbNodeLevelLocal(maxDepth, m_nodes[0]);
	}

	void collectStats(size_t nCores) {
		for(size_t i = 0; i < m_nodes.size(); i++) {
			m_globalData[i].reset();
			for(size_t j =0; j < nCores; j++) {
				m_globalData[i].add(m_coreData[i][j]);
			}
			m_globalData[i].normalize(nCores);

			if(m_globalData[i].NbMutatedPath != 0) {
				Float accRate = m_globalData[i].accRate();
				SLog(EDebug, "Acceptance Rate: %f", accRate);
			} else {
				SLog(EDebug, "No NbMutated Path in this cell");
			}
		}
	}

	int getIDChildData(const Point& p, int maxDepth = -1) {
		GPNodeCopy* currNode = m_nodes[0];
		while(true) {
			const Float splitVal = (Float) currNode->split;
			const int axis = currNode->axis;

			if(currNode->isLeaf) {
				break;
			}

			// Cas d'arret sur le niveau
			if(maxDepth == 0) {
				break;
			}

			// Choose child node ...
			if (p[axis] <= splitVal) {
				if(currNode->leftChild == -1) {
					break;
				}
				currNode = m_nodes[currNode->leftChild];
			} else {
				if(currNode->rightChild == -1) {
					break;
				}
				currNode = m_nodes[currNode->rightChild];
			}

			maxDepth--;
		}
		return currNode->idData;
	}

	Float getAccRate(const Point& p) {
		int idData = getIDChildData(p);
		if(m_globalData[idData].NbMutatedPath != 0) {
			return m_globalData[idData].accRate();
		} else {
			return 0.f; // Nothings !
		}
	}

	GPNodeCopy* getNode(size_t i) {
		return m_nodes[i];
	}

	T& getGlobalData(size_t i) {
		return m_globalData[i];
	}

	template <class R>
	void pruneTree(GatherBlocks & gatherBlocks, Float prunePercent = 0.01) {
		// Reset global stats because it will be
		// use to count all GP
		for(size_t i = 0; i < m_globalData.size(); i++) {
			m_globalData[i].reset(); // FIXME: Attention !
		}

		// Get all gps
		// and add all tree data
		size_t nGP = 0;
		for (int i=0; i<(int) gatherBlocks.size(); ++i) {
			GatherBlock &gps = gatherBlocks[i];
			for(int j = 0; j < (int)gps.size(); j++) {
				GatherPointsList &gpl = gps[j];
				for (GatherPointsList::iterator it = gpl.begin(); it != gpl.end(); ++it) {
					R req;
					req.NbMutatedPath = 1;
					updateStats<R>(it->its.p, req, -1);
					nGP++;
				}
			}
		}

		size_t percentGP = (size_t)(nGP * prunePercent);
		pruneTreeRecursive(m_nodes[0], percentGP, 0);
	}

private:
	bool pruneTreeRecursive(GPNodeCopy * currentNode, size_t percentGP, int depth) {
	  size_t childTotal = m_globalData[currentNode->idData].NbMutatedPath;

	  if(currentNode->isLeaf) {
	    return childTotal == 0;
	  }

	  if(childTotal > percentGP) {
	    bool forceCollapse = false;
	    if(currentNode->leftChild != -1) {
	      forceCollapse |= pruneTreeRecursive(m_nodes[currentNode->leftChild], percentGP, depth + 1);
	    }
	    if(currentNode->rightChild != -1) {
	      forceCollapse |= pruneTreeRecursive(m_nodes[currentNode->rightChild], percentGP, depth + 1);
      }

	    if(forceCollapse) {
	      currentNode->isLeaf = true;
        currentNode->leftChild = -1;
        currentNode->rightChild = -1;
	    }
	    return false;
	  } else {
	    currentNode->isLeaf = true;
	    currentNode->leftChild = -1;
      currentNode->rightChild = -1;
      return (childTotal == 0) || depth >= 4;
	  }
	}

	void pruneTreeAgressive(size_t percentGP) {
    for(size_t i = 0; i < m_globalData.size(); i++) {
      bool needColapse = false;
      if(m_globalData[i].NbMutatedPath == 0) {
//        SLog(EInfo, "Detected empty GP node at : %i", i);
        needColapse = true;
      } else if(m_globalData[i].NbMutatedPath < percentGP) {
//        SLog(EInfo, "Detected less 1 percent GP node at : %i", i);
        needColapse = true;
      }

      if(needColapse) {
        for(size_t j = 0; j < m_nodes.size(); j++) {
          if(m_nodes[j]->idData == i) {
            SLog(EInfo, "Erase tree at the parent: %i",m_nodes[j]->parent);
            m_nodes[m_nodes[j]->parent]->isLeaf = true;
            m_nodes[m_nodes[j]->parent]->leftChild = -1;
            m_nodes[m_nodes[j]->parent]->rightChild = -1;
          }
        }
      }
    }
	}

public:
	std::vector< GPNodeCopy * > m_nodes;
	std::vector< std::vector<T> > m_coreData;
	std::vector<T> m_globalData;
};

typedef SpatialTree<CoreNodeData> SpatialTreeAcc;

MTS_NAMESPACE_END

