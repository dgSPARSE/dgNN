import torch

def TileBalanceScheduler(row_ptr):
    tile_scheduler=[]
    for rid in range(row_ptr.shape[0]-1):
        lb=row_ptr[rid]
        hb=row_ptr[rid+1]
        for _ in range((lb-hb+31)/32):
            tile_scheduler.append[rid]

    return torch.tensor(tile_scheduler).int()

    