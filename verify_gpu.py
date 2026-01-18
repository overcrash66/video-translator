
import torch
import warnings

# Suppress the specific warning to see if it runs otherwise
warnings.filterwarnings("ignore", message=".*NVIDIA GeForce RTX 5060 Ti.*")

try:
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Try to allocate a tensor and do a math op
        x = torch.randn(100, 100).to(device)
        y = torch.matmul(x, x)
        print("Basic Tensor Operation: Success")
        
        # Try to load mmpose (which uses mmcv generic ops)
        import mmpose
        from mmcv.ops import RoIAlign
        print("MMCV/MMPose Import: Success")
        
        import mmdet
        print("MMDetection Import: Success")
        
        # Try a CUDA-specific MMCV op if possible?
        # RoIAlign is a compiled op.
        roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0)
        print("MMCV Op Instantiation: Success")
        
    else:
        print("CUDA not available!")

except Exception as e:
    print(f"FATAL ERROR: {e}")
