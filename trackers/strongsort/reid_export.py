"""
Minimal ReID export functionality for StrongSORT
"""

def export_formats():
    """Return supported export formats for ReID models"""
    from collections import namedtuple
    import pandas as pd
    
    data = [
        ['.pt', 'PyTorch'],
        ['.torchscript', 'TorchScript'],
        ['.onnx', 'ONNX'],
        ['.xml', 'OpenVINO'], 
        ['.engine', 'TensorRT'],
        ['.tflite', 'TensorFlow Lite'],
    ]
    
    df = pd.DataFrame(data, columns=['Suffix', 'Description'])
    return df