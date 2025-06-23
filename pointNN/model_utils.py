import torch
import torch.nn as nn
import torch.nn.functional as F



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B, N, C = points.shape
    
    # Clamp indices to valid range to prevent illegal memory access
    idx = torch.clamp(idx, 0, N - 1)

    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, S, C = new_xyz.shape
    _, N, _ = xyz.shape
    
    # Calculate batch size based on available memory
    # Process in chunks to avoid OOM with large matrices like 22255 * 44510
    chunk_size = min(1000, S)  # Process 1000 query points at a time
    
    group_idx_list = []
    
    for i in range(0, S, chunk_size):
        end_idx = min(i + chunk_size, S)
        chunk_new_xyz = new_xyz[:, i:end_idx, :]
        
        try:
            sqrdists = square_distance(chunk_new_xyz, xyz)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA out of memory error: {e}")
                raise e
            else:
                raise e
        
        # Clamp to avoid NaN/inf values that could cause illegal memory access
        sqrdists = torch.clamp(sqrdists, min=0.0, max=1e6)
        
        if not sqrdists.is_contiguous():
            sqrdists = sqrdists.contiguous()
        
        _, chunk_group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        group_idx_list.append(chunk_group_idx)
        
        # Clear intermediate results to free memory
        del sqrdists
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all chunks
    group_idx = torch.cat(group_idx_list, dim=1)
    
    return group_idx