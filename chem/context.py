import torch
from torch_geometric.data import Batch
import random
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from rdkit import Chem

from data import graph_to_nx, nx_to_graph


class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)
        
class BatchSubstructContext(Batch):
    """
    用于子结构上下文预训练任务的批处理类
    """
    
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        # 自定义拼接：为 *_substruct 与 *_context 字段按各自节点数进行偏移，
        # 并构造训练所需的 batch_overlapped_context 与 overlapped_context_size。
        batch = cls()

        # 累加器
        x_sub_list, ea_sub_list, ei_sub_list, center_idx_list = [], [], [], []
        x_ctx_list, ea_ctx_list, ei_ctx_list, overlap_idx_list = [], [], [], []
        batch_overlap_list, overlap_size_list = [], []

        sub_offset = 0
        ctx_offset = 0

        for i, data in enumerate(data_list):
            # -------- substruct --------
            x_sub = getattr(data, 'x_substruct', torch.empty((0, getattr(data.x, 'size')(1) if hasattr(data, 'x') and data.x.dim()==2 else 2), dtype=getattr(data.x, 'dtype', torch.long)))
            ei_sub = getattr(data, 'edge_index_substruct', torch.empty((2, 0), dtype=torch.long))
            ea_sub = getattr(data, 'edge_attr_substruct', torch.empty((0, 2), dtype=torch.long))
            center_idx = getattr(data, 'center_substruct_idx', torch.tensor([0], dtype=torch.long))

            n_sub = int(x_sub.size(0)) if isinstance(x_sub, torch.Tensor) and x_sub.dim() == 2 else 0

            if ei_sub.numel() > 0:
                ei_sub = ei_sub + sub_offset
            center_idx = center_idx + sub_offset if n_sub > 0 else center_idx

            x_sub_list.append(x_sub)
            ei_sub_list.append(ei_sub)
            ea_sub_list.append(ea_sub)
            center_idx_list.append(center_idx)

            # -------- context --------
            x_ctx = getattr(data, 'x_context', torch.empty((0, getattr(data.x, 'size')(1) if hasattr(data, 'x') and data.x.dim()==2 else 2), dtype=getattr(data.x, 'dtype', torch.long)))
            ei_ctx = getattr(data, 'edge_index_context', torch.empty((2, 0), dtype=torch.long))
            ea_ctx = getattr(data, 'edge_attr_context', torch.empty((0, 2), dtype=torch.long))
            overlap_idx = getattr(data, 'overlap_context_substruct_idx', torch.empty((0,), dtype=torch.long))

            n_ctx = int(x_ctx.size(0)) if isinstance(x_ctx, torch.Tensor) and x_ctx.dim() == 2 else 0

            if ei_ctx.numel() > 0:
                ei_ctx = ei_ctx + ctx_offset
            if overlap_idx.numel() > 0:
                overlap_idx = overlap_idx + ctx_offset

            x_ctx_list.append(x_ctx)
            ei_ctx_list.append(ei_ctx)
            ea_ctx_list.append(ea_ctx)
            overlap_idx_list.append(overlap_idx)

            # 构造 overlapped 的 batch 向量与每图的重叠节点数
            overlap_size = int(overlap_idx.numel())
            overlap_size_list.append(overlap_size)
            if overlap_size > 0:
                batch_overlap_list.append(torch.full((overlap_size,), i, dtype=torch.long))

            # 更新偏移
            sub_offset += n_sub
            ctx_offset += n_ctx

        # 合并为大批次张量
        def _cat(tensors, dim):
            tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
            if len(tensors) == 0:
                return None
            # 处理全为空的情况
            valid = [t for t in tensors if t.numel() > 0]
            if len(valid) == 0:
                return tensors[0]
            return torch.cat(valid, dim=dim)

        x_sub_all = _cat(x_sub_list, dim=0)
        batch.x_substruct = x_sub_all if x_sub_all is not None else torch.empty((0, 2), dtype=torch.long)

        ei_sub_all = _cat(ei_sub_list, dim=1)
        batch.edge_index_substruct = ei_sub_all if ei_sub_all is not None else torch.empty((2, 0), dtype=torch.long)

        ea_sub_all = _cat(ea_sub_list, dim=0)
        batch.edge_attr_substruct = ea_sub_all if ea_sub_all is not None else torch.empty((0, 2), dtype=torch.long)

        center_idx_all = _cat(center_idx_list, dim=0)
        batch.center_substruct_idx = center_idx_all if center_idx_all is not None else torch.empty((0,), dtype=torch.long)

        x_ctx_all = _cat(x_ctx_list, dim=0)
        batch.x_context = x_ctx_all if x_ctx_all is not None else torch.empty((0, 2), dtype=torch.long)

        ei_ctx_all = _cat(ei_ctx_list, dim=1)
        batch.edge_index_context = ei_ctx_all if ei_ctx_all is not None else torch.empty((2, 0), dtype=torch.long)

        ea_ctx_all = _cat(ea_ctx_list, dim=0)
        batch.edge_attr_context = ea_ctx_all if ea_ctx_all is not None else torch.empty((0, 2), dtype=torch.long)

        overlap_idx_all = _cat(overlap_idx_list, dim=0)
        batch.overlap_context_substruct_idx = overlap_idx_all if overlap_idx_all is not None else torch.empty((0,), dtype=torch.long)

        batch_overlap_all = _cat(batch_overlap_list, dim=0)
        batch.batch_overlapped_context = batch_overlap_all if batch_overlap_all is not None else torch.empty((0,), dtype=torch.long)

        batch.overlapped_context_size = torch.tensor(overlap_size_list, dtype=torch.long)

        batch.num_graphs = len(data_list)
        return batch
    
class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
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
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_to_nx(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index
        else:
            # 为了让批处理时各样本拥有一致的键，填充空上下文
            feat_dim = int(data.x.size(1)) if hasattr(data.x, 'size') and data.x.dim() == 2 else 2
            data.x_context = torch.empty((0, feat_dim), dtype=data.x.dtype, device=data.x.device)
            # 边特征维度为 2（bond type, bond direction），与 mol_to_graph 对齐
            data.edge_attr_context = torch.empty((0, 2), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
            data.edge_index_context = torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)
            data.overlap_context_substruct_idx = torch.empty((0,), dtype=torch.long, device=data.x.device)

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)
        elif not hasattr(data, 'overlap_context_substruct_idx'):
            # 若上面 context 为空或无重叠，则确保键存在
            data.overlap_context_substruct_idx = torch.empty((0,), dtype=torch.long, device=data.x.device)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


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


# TODO(Bowen): more unittests