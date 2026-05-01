from piper import PiperVoice
import inspect, sys

# Load any onnx you have
import glob, os
onnx = glob.glob("voices/**/*.onnx", recursive=True)
if not onnx:
    print("No .onnx found"); sys.exit()

v = PiperVoice.load(onnx[0])
sig = inspect.signature(v.synthesize)
print(f"piper version: {PiperVoice.__module__}")
print(f"synthesize signature: {sig}")

# Try calling it with a short text
result = v.synthesize("test")
print(f"synthesize() return type: {type(result)}")

# Is it a generator?
import types
if isinstance(result, types.GeneratorType):
    chunk = next(result)
    print(f"First chunk type: {type(chunk)}, shape/len: {getattr(chunk,'shape', len(chunk))}")
else:
    print(f"Direct return, len={len(result) if hasattr(result,'__len__') else 'N/A'}")
