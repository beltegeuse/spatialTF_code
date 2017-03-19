#pragma once

#include "node.h"

MTS_NAMESPACE_BEGIN

class Node2D : public Node {
 protected:
  // position and size
  // of the current cluster
  Point2i m_posAA;  //< up left point
  Point2i m_posBB;  //< down right
  Vector2i m_size;
 public:
  Node2D(SplittingStrat* s, const Point2i& posAA, const Point2i& posBB)
      : Node(s), // nbCores
        m_posAA(posAA),
        m_posBB(posBB) {
    m_size = m_posBB - m_posAA;
  }

  virtual void save(FileStream* f) {

    // Write all data
    f->writeInt(m_posAA.x);
    f->writeInt(m_posAA.y);
    f->writeInt(m_posBB.x);
    f->writeInt(m_posBB.y);
    saveData(f);

    // Go deeper
    f->writeInt(m_child.size());
    for(int i = 0; i < (int)m_child.size(); i++) {
      m_child[i]->save(f);
    }
  }

  static Node* load(FileStream* f) {

    // Load all data
    Point2i posAA;
    Point2i posBB;
    posAA.x = f->readInt();
    posAA.y = f->readInt();
    posBB.x = f->readInt();
    posBB.y = f->readInt();

    Node* n = new Node2D(new SplittingStratEmpty, posAA, posBB);
    n->loadData(f);

    // Go deeper
    int nbChilds = f->readInt();
    n->m_child.resize(nbChilds);
    for(int i = 0; i < nbChilds; i++) {
      n->m_child[i] = Node2D::load(f);
    }

    return n;
  }

  void buildHierachy() {
    if (m_size.x >= 2 || m_size.y >= 2) {
      bool fourChilds = (m_size.x >= 2 && m_size.y >= 2);
      if (fourChilds) {
        m_child.reserve(4);
        Point2i m11 = Point2i(m_posAA.x + (m_size.x / 2),
                              m_posAA.y + (m_size.y / 2));

        // === Instance all childs
        m_child.push_back(createNode(m_strat->getChildStrat(4), m_posAA, m11));
        m_child.push_back(createNode(m_strat->getChildStrat(4), m11, m_posBB));
        m_child.push_back(
            createNode(m_strat->getChildStrat(4), Point2i(m11.x, m_posAA.y),
                       Point2i(m_posBB.x, m11.y)));
        m_child.push_back(
            createNode(m_strat->getChildStrat(4), Point2i(m_posAA.x, m11.y),
                       Point2i(m11.x, m_posBB.y)));

        // === Build hierachy
        for (size_t i = 0; i < m_child.size(); i++)
          ((Node2D*) m_child[i])->buildHierachy();

      } else {
        // TODO: Check this !
      }
    }
  }

  // FIXME: Change the instant of the hierachie for depricated
  virtual Node* createNode(SplittingStrat* s, const Point2i& posAA,
                           const Point2i& posBB) {
    return new Node2D(s, posAA, posBB);
  }

  virtual size_t nodeID(const GatherPoint& p) const {
    for (size_t i = 0; i < m_child.size(); i++) {
      if (((Node2D*) m_child[i])->isIn(p.points->pos))
        return i;
    }

    SLog(EError, "Not found the child for %s", p.points->pos.toString().c_str());
    return m_child.size();  // Not valid value
  }

 private:
  bool isIn(const Point2i& pos) {
    return (pos.x >= m_posAA.x) && (pos.y >= m_posAA.y) && (pos.x < m_posBB.x)
        && (pos.y < m_posBB.y);
  }
};

class Node3D : public Node {
 private:
  GPNodeCopy* m_node;
  int m_id;
 public:
  Node3D(SplittingStrat* s, std::vector<GPNodeCopy*>& currNodes, int id)
      : Node(s), // nbCores
        m_node(currNodes[id]),
        m_id(id) {
  }

  virtual void save(FileStream* f) {

    // Write all data
    f->writeInt(m_id);
    m_node->save(f);
    saveData(f);

    // Go deeper
    f->writeInt(m_child.size());
    for(int i = 0; i < (int)m_child.size(); i++) {
      m_child[i]->save(f);
    }
  }

  static Node* load(FileStream* f) {

    // Load all data
    int id = f->readInt();
    std::vector<GPNodeCopy*> nodeOne(1);
    nodeOne[0] = new GPNodeCopy;
    nodeOne[0]->load(f);

    Node* n = new Node3D(new SplittingStratEmpty, nodeOne, 0);
    static_cast<Node3D*>(n)->m_id = id;
    n->loadData(f);

    // Go deeper
    int nbChilds = f->readInt();
    n->m_child.resize(nbChilds);
    for(int i = 0; i < nbChilds; i++) {
      n->m_child[i] = Node3D::load(f);
    }

    return n;
  }

  virtual ~Node3D() {
    delete m_node;
  }

  // FIXME: Change the instant of the hierachie for depricated
  virtual Node* createNode(SplittingStrat* s, int nbCores,
                           std::vector<GPNodeCopy*>& currNodes, int id) {
    return new Node3D(s, currNodes, id);
  }

  void buildHierachy(std::vector<GPNodeCopy*>& currNodes, int nbCores) {
    if (!m_node->isLeaf) {
      // Count the number of childs
      //TODO: Is it usefull ?
      int nbChilds = 0;
      if (m_node->leftChild != -1)
        nbChilds += 1;
      if (m_node->rightChild != -1)
        nbChilds += 1;

      if (nbChilds != 2) {
        // We don't want to support the other case
        // for now, to be safe
        // if the hypothesis of two child is false
        // we will not continue to create child
        // raise an warning
        if (nbChilds == 0) {
          SLog(EDebug, "No childs but not a leaf!");
          return;
        } else {
          SLog(EInfo, "Found empty KD-tree optimization");
        }
        // === Assign the new child
        // --- associated ID
        if (m_node->leftChild != -1)
          m_id = m_node->leftChild;
        if (m_node->rightChild != -1)
          m_id = m_node->rightChild;
        m_node = currNodes[m_id];

        // And call build hierachy on the selected node
        buildHierachy(currNodes, nbCores);  // Because we want to continue to build the hierachy

      } else {
        SLog(EDebug, "Create child !!!");

        // Allocate children
        m_child.reserve(nbChilds);
        if (m_node->leftChild != -1) {
          m_child.push_back(
              createNode(m_strat->getChildStrat(nbChilds), nbCores,
                         currNodes, m_node->leftChild));
        }
        if (m_node->rightChild != -1) {
          m_child.push_back(
              createNode(m_strat->getChildStrat(nbChilds), nbCores,
                         currNodes, m_node->rightChild));
        }

        // Continue in the hierachy
        for (size_t i = 0; i < m_child.size(); i++)
          ((Node3D*) m_child[i])->buildHierachy(currNodes, nbCores);
      }
    }
  }

  int nbLeaf() {
    if (m_node->isLeaf || m_child.size() == 0) {
      return 1;
    }

    int nbL = 0;
    for (size_t i = 0; i < m_child.size(); i++) {
      nbL += ((Node3D*) m_child[i])->nbLeaf();
    }
    return nbL;
  }

  virtual size_t nodeID(const GatherPoint& p) const {
    return internalNodeID(p.its.p);
  }


 private:
  inline size_t internalNodeID(const Point& p) const {
    if (m_node->isLeaf || m_child.size() == 0) {
      SLog(EError, "Reach a leaf and ask the child node");
      return 2;  // Invalid value
    }

    const Float splitVal = (Float) m_node->split;
    const int axis = m_node->axis;

    if (p[axis] <= splitVal) {
      if (m_node->leftChild == -1) {  // Unnessary test, just to be safe
        SLog(EError, "Reach an empty left child");
        return 2;  // Invalid value
      }
      return 0;
    } else {
      if (m_node->rightChild == -1) {  // Unnessary test, just to be safe
        SLog(EError, "Reach an empty right child");
        return 2;  // Invalid value
      }
      return 1;
    }
  }


};

MTS_NAMESPACE_END
