"""
Pipeline Logger Utility

Provides a centralized logging mechanism for the entire DPVT data generation pipeline.
Each dataset gets its own log file that tracks all phases of processing.
"""

import os
from datetime import datetime
from pathlib import Path


class PipelineLogger:
    """Logger for tracking the DPVT pipeline execution.

    Creates a single log file per dataset that appends entries from all phases
    of the pipeline (preprocessing, dataset preparation, training data generation).
    """

    def __init__(self, log_file_path, dataset_name=None):
        """Initialize the logger.

        Args:
            log_file_path: Path to the log file
            dataset_name: Optional dataset name for context
        """
        self.log_file = Path(log_file_path)
        self.dataset_name = dataset_name

        # Ensure the directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Write header if this is a new log file
        if not self.log_file.exists():
            self._write_header()

    def _write_header(self):
        """Write the log file header."""
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DPVT Pipeline Execution Log\n")
            if self.dataset_name:
                f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Log created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

    def log(self, phase, message, level="INFO"):
        """Log a message with timestamp and phase information.

        Args:
            phase: Pipeline phase (e.g., "PREPROCESSING", "AGGREGATION")
            message: The message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{phase}] [{level}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        # Also print to console for real-time feedback
        print(log_entry.rstrip())

    def log_section(self, phase, title):
        """Log a section header for better readability.

        Args:
            phase: Pipeline phase
            title: Section title
        """
        with open(self.log_file, 'a') as f:
            f.write("\n" + "-"*80 + "\n")
            f.write(f"{phase}: {title}\n")
            f.write("-"*80 + "\n")

        print(f"\n{'='*80}\n{phase}: {title}\n{'='*80}")



def get_logger(data_dir, dataset_name=None):
    """Get or create a logger for a dataset.

    Args:
        data_dir: Directory where data is being processed
        dataset_name: Name of the dataset (will be inferred from data_dir if not provided)

    Returns:
        PipelineLogger instance
    """
    if dataset_name is None:
        dataset_name = Path(data_dir).name

    log_file = Path(data_dir) / f"{dataset_name}_pipeline.log"
    return PipelineLogger(log_file, dataset_name)