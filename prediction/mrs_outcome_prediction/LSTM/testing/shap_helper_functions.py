# Shap values require very specific versions
# verify versions: TensorFlow 1.14, Python 3.7, Protobuf 3.20, h5py 2.10
import tensorflow as tf
import sys
import google.protobuf
import h5py


def check_shap_version_compatibility():
    error_message = "Shap values require very specific versions: TensorFlow 1.14, Python 3.7, Protobuf 3.20, h5py 2.10"

    assert tf.__version__ == '1.14.0', error_message
    assert sys.version_info[0] == 3 and sys.version_info[1] == 7, error_message
    assert google.protobuf.__version__ == '3.20.0', error_message
    assert h5py.__version__ == '2.10.0', error_message