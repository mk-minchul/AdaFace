import torch
import onnx
import onnxruntime
from torch.autograd import Variable
from inference import load_pretrained_model  # Add this line

def export_onnx(model, input_shape, output_path):
    # Move the model to GPU if available
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape[1:], device=device)

    # Export the model
    # Export the model with dynamic_axes specified
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    # Specify dynamic_axes parameter
    torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)

model_arch = 'ir_101'
# Load AdaFace model
model = load_pretrained_model(model_arch)

# Define dynamic input shape
dynamic_input_shape = (1, 3, 112, 112)  # You can adjust the batch size as needed

# Export the model to ONNX
onnx_output_path = f'./pretrained/adaface_{model_arch}_webface12m_dynamic_model.onnx'
export_onnx(model, dynamic_input_shape, onnx_output_path)

# Validate the exported ONNX model
onnx_model = onnx.load(onnx_output_path)
onnx.checker.check_model(onnx_model)

# Create ONNX Runtime session
ort_session = onnxruntime.InferenceSession(onnx_output_path)

# Validate the dynamic input shape with a sample input
sample_input = torch.randn(dynamic_input_shape).cpu().numpy()
ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
ort_outputs = ort_session.run(None, ort_inputs)

print("ONNX model conversion and validation successful.")
