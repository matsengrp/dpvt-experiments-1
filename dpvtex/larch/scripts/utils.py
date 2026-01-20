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


def get_spr_param_suffix(spr_radius, spr_target_proportion):
    """Generate suffix like '_r2_t0.1' for SPR parameters.

    Args:
        spr_radius: SPR radius parameter. If None, 'None' is used in suffix.
        spr_target_proportion: Target non-MP proportion for SPR methods.

    Returns:
        Parameter suffix string (e.g., '_r2_t0.1' or '_rNone_t0.1').
    """
    radius_str = "None" if spr_radius is None else str(spr_radius)
    return f"_r{radius_str}_t{spr_target_proportion}"


def get_subtree_param_suffix(subtree_target_proportion):
    """Generate suffix like '_t0.167' for subtree parameters.

    Args:
        subtree_target_proportion: Target non-MP proportion for subtree method.

    Returns:
        Parameter suffix string (e.g., '_t0.167').
    """
    return f"_t{subtree_target_proportion}"


def get_full_edge_suffix(
    edge_distribution,
    spr_radius=None,
    spr_target_proportion=None,
    subtree_target_proportion=None,
):
    """Generate full suffix including params for each edge distribution method.

    Combines the base edge distribution suffix with method-specific parameter
    suffixes to create unique identifiers for output files.

    Args:
        edge_distribution: Name of the edge distribution method (e.g., 'constant',
            'uniform', 'treesearch_mimic', 'random_subtree').
        spr_radius: SPR radius parameter for SPR-based methods. If None, no radius
            suffix is added.
        spr_target_proportion: Target non-MP proportion for SPR-based methods.
        subtree_target_proportion: Target non-MP proportion for random_subtree method.

    Returns:
        A string suffix combining the method abbreviation and relevant parameters.
        For example: '_C_r2_t0.1' for constant with radius 2 and target 0.1.
    """
    base_suffix = EDGE_DIST_TO_SUFFIX.get(edge_distribution, "")

    if edge_distribution in ("constant", "uniform", "treesearch_mimic"):
        if spr_radius is not None or spr_target_proportion is not None:
            return base_suffix + get_spr_param_suffix(spr_radius, spr_target_proportion)
    elif edge_distribution == "random_subtree":
        if subtree_target_proportion is not None:
            return base_suffix + get_subtree_param_suffix(subtree_target_proportion)

    return base_suffix
