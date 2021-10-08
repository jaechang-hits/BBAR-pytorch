import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MPNN']

class Pair_Message(nn.Module) :
    """
    See Eq of Pair Message of Gilmer, J. et al. 2017

    message from node w to v
    v: target node
    e: edge
    w: neighbor node
    
    m_wv = f(e, w, v)
    m_wv = m_vw

    Input Size
    nodes: N, V, Fv
    edge: N, V, V, Fe
    """
    def __init__(self, node_size, edge_size, message_size):
        super(Pair_Message, self).__init__()
        self.linear1 = nn.Linear(node_size, node_size)
        self.linear2 = nn.Linear(2 * node_size + edge_size, message_size)

    def forward(self, nodes, edges, adj) :
        batch, n_node, node_size = nodes.size()
        hv = self.linear1(nodes)
        hv = hv.unsqueeze(2).repeat(1, 1, n_node, 1)                        # [N, V, V, Fv]
        hw = hv.transpose(1, 2)                                             # [N, V, V, Fv]
        h = torch.cat([hv, hw, edges], dim=-1)                              # [N, V, V, 2Fv+Fe]
        message = self.linear2(h)                                           # [N, V, V, Fm]
        message = message.masked_fill((adj==0).unsqueeze(-1), 0)
        message = torch.sum(message, 2)                                     # [N, V, Fm]
        message = F.relu(message)

        return message
                 
class Update(nn.Module) :
  def __init__(self, node_hidden_size, message_hidden_size) :
    super(Update, self).__init__()
    self.grucell = nn.GRUCell(message_hidden_size, node_hidden_size)

  def forward(self, nodes, message) :
    batch, n_node, _ = nodes.size()
    nodes = nodes.view(batch*n_node, -1)
    message = message.view(batch*n_node, -1)
    upd_nodes = self.grucell(message, nodes)
    upd_nodes = upd_nodes.view(batch, n_node, -1)
    return upd_nodes

class MPNNLayer(nn.Module) :
    def __init__(self, node_size, edge_size, hidden_size, n_head, dropout) :
        super(MPNNLayer, self).__init__()
        self.message = nn.ModuleList([Pair_Message(node_size, edge_size, hidden_size) for _ in range(n_head)])
        self.update = nn.ModuleList([Update(node_size, hidden_size) for _ in range(n_head)])
        self.linear = nn.Linear(node_size*n_head, node_size)
        self.dropout = nn.Dropout(p=dropout)
        self.n_head = n_head
    
    def forward(self, nodes, edges, adj) :
        upd_nodes_total = []
        for head in range(self.n_head) :
            _nodes = self.dropout(nodes)
            msg = self.message[head](_nodes, edges, adj)
            upd_nodes_total.append(self.update[head](_nodes, msg))
        upd_nodes = torch.cat(upd_nodes_total, dim=-1)
        upd_nodes = F.relu(upd_nodes)
        upd_nodes = self.linear(upd_nodes)
        return (nodes + upd_nodes)

class MPNN(nn.Module) :
    """
    Input
    nodes: batch, num_nodes, node_hidden
    edge: batch, num_nodes, num_nodes, edge_hidden
    adj: batch, num_nodes, num_nodes

    Output
    X(node state): batch, num_nodes, node_hidden
    """
    def __init__(self, node_size, edge_size, hidden_size, n_head, n_layer, dropout) :
        super(MPNN, self).__init__()
        self.layers = nn.ModuleList([MPNNLayer(node_size, edge_size, hidden_size, n_head, dropout) for _ in range(n_layer)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nodes, edges, adj) :
        upd_nodes = nodes
        for mod in self.layers :
            upd_nodes = mod(upd_nodes, edges, adj)

        return upd_nodes 
