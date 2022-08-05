from dgNN.layers.gatconv_layer import GATConv
import argparse
import time
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
import GPUtil
import scipy.sparse as sp

class Net(nn.Module):
    def __init__(self,
                 row_ptr,col_idx,col_ptr,row_idx,permute,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=None,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=None):
        super(Net, self).__init__()
        self.row_ptr=row_ptr
        self.col_idx=col_idx
        self.col_ptr=col_ptr
        self.row_idx=row_idx
        self.permute=permute
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],feat_drop,attn_drop,
            negative_slope,residual,bias=False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],feat_drop,attn_drop,
                negative_slope,residual))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],feat_drop,attn_drop,
           negative_slope))

    def forward(self,inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.row_ptr,self.col_idx,self.col_ptr,self.row_idx,self.permute,h).flatten(1) # h.shape[-1] = num_heads*out_feats
        # output projection
        logits = self.gat_layers[-1](self.row_ptr,self.col_idx,self.col_ptr,self.row_idx,self.permute,h).mean(1)
        return logits

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def load_dataset(args):
    if args.dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    
    col,row=g.edges(order='srcdst')
    edge_idx=torch.stack((col,row))
    numlist = torch.arange(col.size(0), dtype=torch.int32)

    adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    
    row_ptr=torch.from_numpy(adj_csr.indptr)
    col_ind=torch.from_numpy(adj_csr.indices)

    adj_csc=adj_csr.tocsc()

    col_ptr=torch.from_numpy(adj_csc.indptr)
    row_ind=torch.from_numpy(adj_csc.indices)

    adj_csr_new=sp.csr_matrix((numlist.numpy(),col_ind.cpu().numpy(),row_ptr.cpu().numpy()))
    adj_csc_new=adj_csr_new.tocsc()
    permute=torch.from_numpy(adj_csc_new.data)
    print(permute)
    features=g.ndata['feat']
    labels=g.ndata['label']
    n_feats=features.shape[1]
    n_classes=data.num_labels
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']

    return row_ptr,col_ind,col_ptr,row_ind,edge_idx,features,labels,n_feats,n_classes,train_mask,test_mask,permute

def main(args):
    #load dataset
    row_ptr,col_idx,col_ptr,row_idx,edge_idx,features,labels,n_feats,n_classes,train_mask,test_mask,permute=load_dataset(args)

    row_ptr=row_ptr.to(args.gpu).int()
    col_idx=col_idx.to(args.gpu).int()
    col_ptr=col_ptr.to(args.gpu).int()
    row_idx=row_idx.to(args.gpu).int()
    features=features.to(args.gpu)
    labels=labels.to(args.gpu)
    train_mask=train_mask.to(args.gpu)
    test_mask=train_mask.to(args.gpu)
    permute=permute.to(args.gpu)

    heads = ([args.n_heads] * args.n_layers) + [1]
    model=Net(row_ptr,col_idx,col_ptr,row_idx,permute,args.n_layers,n_feats,args.n_hidden,n_classes,heads,attn_drop=args.attn_drop,feat_drop=args.dropout,negative_slope=args.negative_slope).to(args.gpu)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print('warm up')
    maxMemory = 0
    for _ in range(10):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        GPUs = GPUtil.getGPUs()
        maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory) 
  
    print('profile training')
    model.train()
    torch.cuda.synchronize()
    start=time.time()
    for _ in range(args.n_epochs):
        logits=model(features)
        loss=loss_fcn(logits[train_mask],labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        print(loss.item())
    torch.cuda.synchronize()
    end=time.time()
    train_time=(end-start)/args.n_epochs

    print('profile inference')
    model.eval()
    torch.cuda.synchronize()
    start=time.time()
    for epoch in range(args.n_epochs):  
        with torch.no_grad():
            logits=model(features)
    torch.cuda.synchronize()
    end=time.time()
    inference_time=(end-start)/args.n_epochs
    
    logits = logits[test_mask]  
    acc=accuracy(logits, labels[test_mask])
    print("Test Accuracy {:.4f}".format(acc))
    print(f'max memory:{maxMemory}MB')
    print("train time:",train_time)
    print("inference time:",inference_time)

    if args.output!=None:
        with open("{}".format(args.output),'a') as f:
            print("train_GAT_dgnn,{} heads={} hidden_dim={},{:f}s,{:f}s,{}MB,{}".format(args.dataset,args.n_heads,args.n_hidden,train_time,inference_time,maxMemory,acc),file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--attn-drop',type=float,default=0.,
                        help="drop out rate for attention weights")
    parser.add_argument('--output',type=str,default=None,
                        help="output file")
                    
    args = parser.parse_args()
    print(args)
    main(args)

