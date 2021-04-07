#ifndef TYPES_H
#define TYPES_H

/** 
 * Directed edge type.
 * Contains a source node and a destination node.
 */
template <typename TNode> 
class edge_t {
    TNode from; // From node.
    TNode to;   // To node.

    edge_t(TNode from_, TNode to_) 
        : from(from_)
        , to(to_) 
    {}    
};

/**
 * Directed weighted edge type.
 * In addition to a source and destination node, it contains the edge weight.
 */
template <typename TNode, typename TWeight = TNode>
class weighted_edge_t : edge_t<TNode> {
    TWeight weight;

    weighted_edge_t(TNode from_, TNode to_, TWeight weight_)
        : edge_t<TNode>(from_, to_)
        , weight(weight_)
    {}
};

#endif // TYPES_H
