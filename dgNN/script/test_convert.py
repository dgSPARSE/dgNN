import dgl
import torch
import scipy.sparse as sp
from torch.utils.cpp_extension import load

import format_conversion

data = dgl.data.PubmedGraphDataset()
g = data[0]
col,row=g.edges(order='srcdst')

adj_csr = sp.csr_matrix((torch.ones(row.shape), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
csr_ptr=torch.from_numpy(adj_csr.indptr)
csr_ind=torch.from_numpy(adj_csr.indices)
m=csr_ptr.shape[0]-1

adj_coo=adj_csr.tocoo()
coo_row_ind=torch.from_numpy(adj_coo.row)
coo_col_ind=torch.from_numpy(adj_coo.col)
print(coo_row_ind,coo_col_ind)

adj_csc=adj_csr.tocsc()
csc_ptr=torch.from_numpy(adj_csc.indptr)
csc_ind=torch.from_numpy(adj_csc.indices)

csr_ptr_our=format_conversion.coo2csr(coo_row_ind.cuda(),m)
csc_ptr_our,_,_=format_conversion.csr2csc(csr_ptr_our.cuda(),csr_ind.cuda(),csr_ind.cuda().float())

print(torch.allclose(csr_ptr,csr_ptr_our.cpu(),1e-2))
print(torch.allclose(csc_ptr,csc_ptr_our.cpu(),1e-2))

