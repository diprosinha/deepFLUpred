"""deepFLUpred public Python API.

deepFLUpred: A Deep Learning-Based Framework for Genomic Characterization,
Subtyping, and Pathogenicity Prediction of Avian Influenza Viruses.

Authors: Dipro Sinha, Naveen Duhan, Jagathiswaran Radhakrisnan, Sunil Mor
"""

from deepflupred.constants import PACKAGE_VERSION
from deepflupred.predict import predict_fasta, predict_sequence, predict_sequences

__all__ = [
    "predict_fasta",
    "predict_sequence",
    "predict_sequences",
]
__version__ = PACKAGE_VERSION
__author__ = "Dipro Sinha"
