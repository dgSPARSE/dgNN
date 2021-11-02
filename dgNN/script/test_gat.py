import argparse
import time
import torch
from dgNN.layers import GATConv


def main(args):


    # row_ind=row_ind.to(args.gpu).int()
    row_ptr=torch.tensor([0,1,2,3,4])
    col_ind=torch.tensor([0,1,2,3,4])
    row_ptr=row_ptr.to(args.gpu).int()
    col_ind=col_ind.to(args.gpu).int()

    col_ptr=row_ptr
    row_ind=col_ind
    
    model=GATConv(args.in_feats,args.out_feats,args.num_heads).to(args.gpu)
    features=torch.rand(row_ptr.shape[0]-1,args.in_feats,device=args.gpu)
    
    # if args.profileio:
    #     # profile_start()
    #     model(row_ptr,col_ind,col_ptr,row_ind,features,True)
    #     # profile_end()
    #     exit()

    maxMemory = 0
    for _ in range(5):
        model(row_ptr,col_ind,col_ptr,row_ind,features)
        # GPUs = GPUtil.getGPUs()
        # maxMemory = max(GPUs[args.gpu].memoryUsed, maxMemory)    

    torch.cuda.synchronize()
    start=time.time()
    
    for epoch in range(args.epochs):
        model(row_ptr,col_ind,col_ptr,row_ind,features)   
        
    torch.cuda.synchronize()
    end=time.time()
    # print(maxMemory)
    print("gatconv_our forward time:",(end-start)/args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--in_feats",type=int,default=16)
    parser.add_argument("--out_feats",type=int,default=6)
    parser.add_argument("--dataset",type=str,default="cora")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument('--profileio',type=int,default=0)
                    
    args = parser.parse_args()

    main(args)


