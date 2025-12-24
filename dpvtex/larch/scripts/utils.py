"""Shared constants and utilities for the DPVT larch pipeline.

This module provides a single source of truth for edge distribution suffix mappings
and other shared utilities used across the pipeline.
"""

import os

# Suffix mapping for edge distributions
# Note: "constant" -> "_spr" and "random_subtree" -> "_subtree" for historical reasons
EDGE_DIST_TO_SUFFIX = {
    "constant": "_spr",
    "uniform": "_uniform",
    "treesearch_mimic": "_treesearch_mimic",
    "random_subtree": "_subtree",
}

SUFFIX_TO_EDGE_DIST = {
    suffix: dist_id for dist_id, suffix in EDGE_DIST_TO_SUFFIX.items()
}


def get_dup_sites_suffix(remove_site_patterns):
    """Get the duplicate sites suffix based on the config setting.

    Args:
        remove_site_patterns: Config value for remove_duplicate_site_patterns.
            Can be bool or string.

    Returns:
        "_no_dup_sites" if patterns should be removed, empty string otherwise.
    """
    if remove_site_patterns in [True, "True", "true"]:
        return "_no_dup_sites"
    return ""


def determine_file_format(filename):
    """Determine alignment file format from extension.

    Args:
        filename: Path to the alignment file.

    Returns:
        "nexus" for .nex/.nexus files, "fasta" otherwise.
    """
    if str(filename).endswith((".nex", ".nexus")):
        return "nexus"
    return "fasta"


def get_alignment_name_from_path(filepath):
    """Extract alignment name from a file path.

    Assumes the alignment name is the parent directory of the file.
    For example: /data/alignment_1/input.fasta -> "alignment_1"

    Args:
        filepath: Path to a file within an alignment directory.

    Returns:
        The name of the parent directory.
    """
    return os.path.basename(os.path.dirname(str(filepath)))
