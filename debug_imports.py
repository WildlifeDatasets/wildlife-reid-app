import sys
import os

def test_import(module_name):
    print(f"Testing import of {module_name}...")
    try:
        __import__(module_name)
        print(f"SUCCESS: {module_name} imported.")
    except Exception as e:
        print(f"FAILED: {module_name} failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Path: {sys.executable}")
    
    test_import("torch")
    test_import("faiss")
    test_import("PyQt6")
    test_import("pandas")
    test_import("torchvision")
