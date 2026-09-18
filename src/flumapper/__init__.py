"""FluMAPPER public Python API.

FluMAPPER: An AI-based Molecular Analysis, Prediction and Profiling platform
for influenza viruses.

Author: Dipro Sinha
"""

from flumapper.constants import PACKAGE_VERSION
from flumapper.predict import predict_fasta, predict_sequence, predict_sequences

__all__ = [
    "predict_fasta",
    "predict_sequence",
    "predict_sequences",
]
__version__ = PACKAGE_VERSION
__author__ = "Dipro Sinha"
