"""Reusable leakage detectors extracted into a small package.

The original notebooks remain the research record. This package provides a
shared importable API for the four datasets represented there.
"""

from .detectors.base import DetectorConfig, LeakageDetector
from .detectors.target import TargetLeakageConfig, TargetLeakageDetector
from .detectors.contamination import ContaminationConfig, ContaminationLeakageDetector
from .detectors.temporal import TemporalConfig, TemporalLeakageDetector
from .detectors.preprocessing import PreprocessingConfig, PreprocessingLeakageDetector
from .main import generate_json_report, main

__all__ = [
    "LeakageDetector",
    "DetectorConfig",
    "TargetLeakageConfig",
    "TargetLeakageDetector",
    "ContaminationConfig",
    "ContaminationLeakageDetector",
    "TemporalConfig",
    "TemporalLeakageDetector",
    "PreprocessingConfig",
    "PreprocessingLeakageDetector",
    "main",
    "generate_json_report",
]
