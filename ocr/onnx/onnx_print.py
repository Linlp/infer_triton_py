import onnxruntime

session = onnxruntime.InferenceSession("./ppv4_rec.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape

print(f"Input Name: {input_name}, Shape: {input_shape}")
print(f"Output Name: {output_name}, Shape: {output_shape}")

