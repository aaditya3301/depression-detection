"""
One-time script: convert your .keras model to .onnx format.
Run this locally (where tensorflow is installed), then push the .onnx file.

Uses SavedModel intermediate format to avoid tf2onnx from_keras bugs.
"""
import tensorflow as tf
import subprocess
import os
import sys

MODEL_PATH = "models/best_depression_model.keras"
SAVED_MODEL_DIR = "models/temp_saved_model"
ONNX_PATH = "models/depression_model.onnx"

# Step 1: Load Keras model
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# Step 2: Export as SavedModel format
print(f"\nExporting to SavedModel format at: {SAVED_MODEL_DIR}")
model.export(SAVED_MODEL_DIR)

# Step 3: Convert SavedModel to ONNX using command line
print("\nConverting SavedModel to ONNX...")
result = subprocess.run(
    [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", SAVED_MODEL_DIR,
        "--output", ONNX_PATH,
        "--opset", "13",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    raise RuntimeError("ONNX conversion failed")

print(f"\nSaved ONNX model to: {ONNX_PATH}")
print(f"Size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")

# Cleanup temp saved model
import shutil
shutil.rmtree(SAVED_MODEL_DIR, ignore_errors=True)
print("Cleaned up temp SavedModel directory.")
