
import tensorrt as trt
import os
import sys
import argparse
import ctypes

def build_engine(onnx_file_path, engine_file_path, use_fp16=True, plugin_path=None):
    """
    Builds a TensorRT engine from an ONNX file.
    """
    logger = trt.Logger(trt.Logger.INFO)
    
    # Load Plugin if provided
    if plugin_path:
        if not os.path.exists(plugin_path):
            print(f"Error: Plugin not found at {plugin_path}")
            return False
        print(f"Loading plugin library: {plugin_path}")
        ctypes.CDLL(plugin_path)
        trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    
    # Create Network
    # EXPLICIT_BATCH is required for ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, logger)
    
    config = builder.create_builder_config()
    
    # Memory Pool (replacing max_workspace_size)
    # 2GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 extraction...")
        config.set_flag(trt.BuilderFlag.FP16)

    # Parse ONNX
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found: {onnx_file_path}")
        return False
        
    print(f"Parsing ONNX: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    print("Building serialized network...")
    # Build
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        # Fallback for older TRT versions if build_serialized_network doesn't exist (it should in TRT 10)
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize()

    if serialized_engine is None:
        print("Build failed.")
        return False
        
    print(f"Saving engine to: {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
        
    print("Success.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT Engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX file")
    parser.add_argument("--saveEngine", required=True, help="Path to save output .engine file")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--plugins", help="Path to plugin DLL (optonal)")
    
    args = parser.parse_args()
    
    success = build_engine(args.onnx, args.saveEngine, args.fp16, args.plugins)
    if not success:
        sys.exit(1)
