from torch.utils.cpp_extension import load
import torch
# import dgNN
import fused_gatconv as fused_gat

def GATConvFuse(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,permute,negative_slope,in_feat,attn_drop):
    return FusedGATFunction.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,permute,negative_slope,in_feat,attn_drop)

def GATConvFuse_inference(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat):
    return fused_gat.gat_inference(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)

class FusedGATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,permute,negative_slope,in_feat,attn_drop):
    
        out_feat,edge_max,edge_sum,edge_mask=fused_gat.gat_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat,attn_drop)
        # print(edge_mask)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,permute,edge_max,edge_sum,edge_mask,in_feat,attn_row,attn_col)
        ctx.negative_slope=negative_slope
        ctx.attn_drop=attn_drop      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,permute,edge_max,edge_sum,edge_mask,in_feat,attn_row,attn_col=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_gat.gat_backward(
            ctx.negative_slope,ctx.attn_drop,row_ptr,col_ind,col_ptr,row_ind,permute,edge_max,edge_sum,edge_mask,in_feat,attn_row,attn_col,grad_out)
        # print('end backward')
        # print(torch.isnan(grad_feat).sum())
        # print(torch.isnan(grad_attn_row).sum())
        # print(torch.isnan(grad_attn_col).sum())
        return grad_attn_row,grad_attn_col,None,None,None,None,None,None,grad_feat,None

