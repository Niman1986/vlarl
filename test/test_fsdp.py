"""
Usage:
    NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test/test_torch.py
    CUDA_VISIBLE_DEVICES=4,5 NCCL_DEBUG=TRACE torchrun --nproc-per-node=2 test/test_torch.py
"""
