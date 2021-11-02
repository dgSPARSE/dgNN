import argparse
import time
import torch
import torch.nn.functional as F
import dgl
# import dgl.data

import torch.nn as nn
# import sys
# sys.path.append('../..')
# from util.indicator import *
from dgNN.layers.gatconv_layer import GATConv
# from torch.autograd.profiler import profile


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            negative_slope))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                negative_slope))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
           negative_slope))

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](row_ptr,col_ind,col_ptr,row_ind,h).flatten(1) # h.shape[-1] = num_heads*out_feats
        # output projection
        logits = self.gat_layers[-1](row_ptr,col_ind,col_ptr,row_ind,h).mean(1)
        return logits


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)
import scipy.sparse as sp
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
    adj_csr = sp.csr_matrix((torch.ones(row.shape), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    
    row_ptr=torch.from_numpy(adj_csr.indptr)
    col_ind=torch.from_numpy(adj_csr.indices)

    adj_coo=adj_csr.tocoo()
    row_ind=torch.from_numpy(adj_coo.row)
    print('dataset verified:',torch.equal(col_ind,torch.from_numpy(adj_coo.col)))

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    n_classes=data.num_labels
    num_feats = features.shape[1]
    return row_ind,row_ptr,col_ind,features,labels,train_mask,val_mask,test_mask,n_classes,num_feats

def preprocess_csr2csc(rowptr,colind,args):
    # numlist = torch.arange(colind.size(0), device=args.gpu, dtype=torch.int32)
    numlist=torch.arange(colind.size(0))

    adj_csr=sp.csr_matrix((numlist.numpy(),colind.cpu().numpy(),rowptr.cpu().numpy()))
    adj_csc=adj_csr.tocsc()
    # permute=adj_csc.data
    # print(permute)
    # print(torch.max(torch.from_numpy(permute)))
    colptr=adj_csc.indptr
    rowind=adj_csc.indices
    # print(colptr.shape)
    # colptr, rowind, permute = spmm.csr2csc(rowptr, colind, numlist.float())
    # permute = permute.int()
    return torch.from_numpy(colptr).to(args.gpu),torch.from_numpy(rowind).to(args.gpu)

def main(args):
    #load dataset
    row_ind,row_ptr,col_ind,features,labels,train_mask,val_mask,test_mask,n_classes,num_feats=load_dataset(args)
    n_edges = row_ind.shape[0]

    # row_ind=row_ind.to(args.gpu).int()
    row_ptr=row_ptr.to(args.gpu).int()
    col_ind=col_ind.to(args.gpu).int()
    features=features.to(args.gpu).float()
    labels=labels.to(args.gpu)
    train_mask=train_mask.to(args.gpu)
    val_mask=val_mask.to(args.gpu)
    test_mask=test_mask.to(args.gpu)

    col_ptr,row_ind=preprocess_csr2csc(row_ptr,col_ind,args)
    

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d
      #Input features %d
     """ %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item(),features.shape[1]))
    
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                torch.nn.functional,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    model.to(args.gpu)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(10):
        logits = model(row_ptr,col_ind,col_ptr,row_ind,features)

    print(args)
    print('profile training')
    torch.cuda.synchronize()
    start=time.time()
    for epoch in range(args.epochs):
        # print(epoch)
        model.train()
        # if(args.profileio):
        #     profile_start()
        logits = model(row_ptr,col_ind,col_ptr,row_ind,features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        # if(args.profileio):
        #     profile_end()
        #     break
        print("loss",loss.item())     
    torch.cuda.synchronize()
    end=time.time()
    train_time=(end-start)/args.epochs

    # acc = evaluate(model, features, labels, test_mask)
    model.eval()
    with torch.no_grad():
        logits = model(row_ptr,col_ind,col_ptr,row_ind,features)
        logits = logits[test_mask]
        
    acc=accuracy(logits, labels[test_mask])
    print("Test Accuracy {:.4f}".format(acc))

    print("train time:",train_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--profileio", type=int, default=0,
                    help="1 for profile io")
    args = parser.parse_args()

    main(args)


