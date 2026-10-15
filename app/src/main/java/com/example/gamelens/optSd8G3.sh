# 1. Convert the static N=8 ONNX model graph to QNN layout format
qnn-onnx-converter \
    --input_network models/sesr_m5_batch8.onnx \
    --input_dim input_name 8,128,128,3 \
    --output_path build/sesr_m5_qnn.cpp

# 2. Compile the model using W8A8 (INT8) vector metrics targeted for Hexagon V75
qnn-model-lib-generator \
    --model_cpp build/sesr_m5_qnn.cpp \
    --model_bin build/sesr_m5_qnn.bin \
    --output_dir build/libs \
    --target_arch aarch64-android \
    --lib_name libsesr_htp_v75

# 3. Optimize execution pathways specifically for high performance bursts
# This writes a layout library (.so) that gets packaged straight into your app's nativeLibDir
