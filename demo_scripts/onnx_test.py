import onnx
import onnxruntime as ort

# Check if model is valid
try:
    model = onnx.load("/home/smr/Model/boxing1_his4.onnx")
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid")

    # Test ONNX runtime session creation
    session = ort.InferenceSession("/home/smr/Model/boxing1_his4.onnx")
    print("✅ ONNX runtime session created successfully")

except Exception as e:
    print(f"❌ Model validation failed: {e}")
