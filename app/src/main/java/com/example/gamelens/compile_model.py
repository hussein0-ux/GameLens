import qai_hub as hub

# Initialize client connection (Requires you to log in to aihub.qualcomm.com to get your API Token)
client = hub.Client()

# Point to your custom N=8 batch-optimized SESR-M5 ONNX file
model_file = "models/sesr_m5_batch8.onnx"

print("Submitting optimized compilation job to Qualcomm AI Hub...")
compile_job = client.submit_compile_job(
    model=model_file,
    device=hub.Device("Samsung Galaxy S24 (Family)"), # Targets Snapdragon 8 Gen 3 HTP
    options="--target_runtime qnn_context_binary --quantization_type int8",
    input_specs=dict(input_name=(8, 128, 128, 3)) # Explicitly forces the static N=8 batch layout
)

# Block until the cloud cluster completes generation
compile_job.wait()

# Download your deployment target assets directly to Termux
print("Downloading production binary (.so / .bin)...")
compile_job.download_artifacts_for_type(
    artifact_type=hub.ArtifactType.QNN_CONTEXT_BINARY, 
    output_dir="./build_output"
)
print("Success! Move the downloaded lib to your application's native directory.")
