"""
Patches ONNX models to use the GridSample3D TensorRT plugin.

TensorRT's built-in GridSample only supports 4D inputs, but LivePortrait's
warping_spade model uses 5D GridSample. This script renames 5D GridSample
nodes to GridSample3D so TensorRT routes them to the custom plugin.
"""
import onnx
import argparse
import os

def patch_onnx_for_gridsample3d(input_path: str, output_path: str) -> int:
    """
    Patches all GridSample nodes in the ONNX model to use GridSample3D.
    
    Returns the number of nodes patched.
    """
    print(f"Loading ONNX model: {input_path}")
    model = onnx.load(input_path)
    
    patched_count = 0
    for node in model.graph.node:
        if node.op_type == "GridSample":
            print(f"  Patching node: {node.name} (GridSample -> GridSample3D)")
            node.op_type = "GridSample3D"
            patched_count += 1
    
    if patched_count > 0:
        print(f"Saving patched model to: {output_path}")
        onnx.save(model, output_path)
        print(f"Done. Patched {patched_count} node(s).")
    else:
        print("No GridSample nodes found. Model unchanged.")
        # Still save to output if different path, to maintain workflow
        if input_path != output_path:
            onnx.save(model, output_path)
    
    return patched_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch ONNX GridSample to GridSample3D")
    parser.add_argument("--input", required=True, help="Path to input ONNX file")
    parser.add_argument("--output", required=True, help="Path to output ONNX file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
        
    patch_onnx_for_gridsample3d(args.input, args.output)
