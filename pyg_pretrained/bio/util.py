import random
import torch
import numpy as np
import networkx as nx
from loader import BioDataset, graph_data_obj_to_nx, nx_to_graph_data_obj

def combine_dataset(dataset1, dataset2):
    data_list = [data for data in dataset1]
    data_list.extend([data for data in dataset2])
    root_supervised = 'dataset/supervised'
    dataset = BioDataset(root_supervised, data_type='supervised', empty = True)

    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset

class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0,i].cpu().item()) + "," + str(data.edge_index[1,i].cpu().item()) for i in range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2,5*num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5*num_edges):
            node1 = redandunt_sample[0,i].cpu().item()
            node2 = redandunt_sample[1,i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges/2:
                break

        data.negative_edge_index = redandunt_sample[:,sampled_ind]
        
        return data

class MaskEdge:
    def __init__(self, mask_rate):
        """
        Assume edge_attr is of the form:
        [w1, w2, w3, w4, w5, w6, w7, self_loop, mask]
        :param mask_rate: % of edges to be masked
        """
        self.mask_rate = mask_rate

    def __call__(self, data, masked_edge_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_edge_indices: If None, then randomly sample num_edges * mask_rate + 1
        number of edge indices. Otherwise should correspond to the 1st
        direction of an edge pair. ie all indices should be an even number
        :return: None, creates new attributes in the original data object:
        data.mask_edge_idx: indices of masked edges
        data.mask_edge_labels: corresponding ground truth edge feature for
        each masked edge
        data.edge_attr: modified in place: the edge features (
        both directions) that correspond to the masked edges have the masked
        edge feature
        """
        if masked_edge_indices == None:
            # sample x distinct edges to be masked, based on mask rate. But
            # will sample at least 1 edge
            num_edges = int(data.edge_index.size()[1] / 2)  # num unique edges
            sample_size = int(num_edges * self.mask_rate + 1)
            # during sampling, we only pick the 1st direction of a particular
            # edge pair
            masked_edge_indices = [2 * i for i in random.sample(range(
                num_edges), sample_size)]

        data.masked_edge_idx = torch.tensor(np.array(masked_edge_indices))

        # create ground truth edge features for the edges that correspond to
        # the masked indices
        mask_edge_labels_list = []
        for idx in masked_edge_indices:
            mask_edge_labels_list.append(data.edge_attr[idx].view(1, -1))
        data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)

        # created new masked edge_attr, where both directions of the masked
        # edges have masked edge type. For message passing in gcn

        # append the 2nd direction of the masked edges
        all_masked_edge_indices = masked_edge_indices + [i + 1 for i in
                                                         masked_edge_indices]
        for idx in all_masked_edge_indices:
            data.edge_attr[idx] = torch.tensor(np.array([0, 0, 0, 0, 0,
                                                             0, 0, 0, 1]),
                                                      dtype=torch.float)

        return data
        # # debugging
        # print(masked_edge_indices)
        # print(all_masked_edge_indices)


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class ExtractSubstructureContextPair:
    def __init__(self, l1, center=True):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds the whole graph, and the
        context substructures that corresponds to
        the subgraph that is between l1 and the edge of the graph. If
        center=True, then will select the center node as the root node.
        :param center: True, will select a center node as root node, otherwise
        randomly selects a node
        :param l1:
        """
        self.center = center
        self.l1 = l1

    
        if self.l1 == 0:
            self.l1 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: Usually None. Otherwise directly sets node idx of
        root (
        for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        G = graph_data_obj_to_nx(data)

        if root_idx == None:
            if self.center == True:
                root_idx = data.center_node_idx.item()
            else:
                root_idx = random.sample(range(num_atoms), 1)[0]

        # in the PPI case, the subgraph is the entire PPI graph
        data.x_substruct = data.x
        data.edge_attr_substruct = data.edge_attr
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = data.center_node_idx


        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_atoms)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)   # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

    def __repr__(self):
        return '{}(l1={}, center={})'.format(self.__class__.__name__,
                                              self.l1, self.center)

if __name__ == "__main__":
    root_supervised = 'dataset/supervised'
    thresholds = [266, 1, 777, 652, 300, 900, 670]
    d_supervised = BioDataset(root_supervised, thresholds,
                              max_search_depth=2, max_n_neighbors_sampled=10,
                              n_subgraphs=1e12, data_type='supervised')
    # test ExtractSubstructureContextPair for PPI networks
    data = d_supervised[0]
    sub_context_transform = ExtractSubstructureContextPair(1, center=True)
    sub_context_transform(data)
