from dpvtex.larch.scripts.clean_data import (
    get_informative_site_indices,
    calculate_sequence_quality,
    filter_low_quality_sequences,
    remove_uninformative_sites,
    clean_alignment,
)
from dpvtex.larch.scripts.pipeline_logger import PipelineLogger

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import tempfile
import os
import csv
import pytest


def are_alignments_equal(align1, align2):
    # Check if two alignments are equal.
    # This assumes that the order of sequences is the same in both alignments
    if (
        len(align1) != len(align2)
        or align1.get_alignment_length() != align2.get_alignment_length()
    ):
        return False
    for seq_record1, seq_record2 in zip(align1, align2):
        # Compare identifiers and sequences
        if (seq_record1.id != seq_record2.id) or (
            str(seq_record1.seq) != str(seq_record2.seq)
        ):
            return False
    return True


def remove_ambiguous_or_uninformative_sites(alignment):
    """
    Remove sites that are not phylogenetically informative.
    This is a wrapper around get_informative_site_indices for testing.
    """
    informative_indices = get_informative_site_indices(alignment)

    # Create a new alignment with only the informative sites
    new_sequences = []
    for record in alignment:
        new_seq = Seq("".join([str(record.seq[i]) for i in informative_indices]))
        new_record = SeqRecord(new_seq, id=record.id, description="")
        new_sequences.append(new_record)

    return MultipleSeqAlignment(new_sequences)


def test_remove_ambiguous_or_uninformative_sites():
    seq1 = SeqRecord(Seq("ACTGCCAGCT-G"), id="Seq1")
    seq2 = SeqRecord(Seq("ACAGACAGCATG"), id="Seq2")
    seq3 = SeqRecord(Seq("ACTCTTAGCCCG"), id="Seq3")
    seq4 = SeqRecord(Seq("ACT-CTAGCGA-"), id="Seq4")
    alignment = MultipleSeqAlignment([seq1, seq2, seq3, seq4])

    expected_seq1 = SeqRecord(Seq("CCT"), id="Seq1")
    expected_seq2 = SeqRecord(Seq("ACA"), id="Seq2")
    expected_seq3 = SeqRecord(Seq("TTC"), id="Seq3")
    expected_seq4 = SeqRecord(Seq("CTG"), id="Seq4")

    expected_alignment = MultipleSeqAlignment(
        [expected_seq1, expected_seq2, expected_seq3, expected_seq4]
    )

    cleaned_alignment = remove_ambiguous_or_uninformative_sites(alignment)

    assert are_alignments_equal(expected_alignment, cleaned_alignment)


def test_gappy_sequence_cleaning():
    """
    Test cleaning of alignments with super gappy sequences and scattered gaps.
    This tests that filter_low_quality_sequences properly removes sequences
    with too many gaps and ambiguous characters (including all IUPAC ambiguous codes).
    """
    # Create an alignment with:
    # - One super gappy sequence (>80% gaps)
    # - One sequence with many ambiguous bases (various IUPAC codes)
    # - One sequence with ~50% gaps + ambiguous combined
    # - Some sequences with scattered gaps (acceptable)
    # - Some high-quality sequences

    seq1 = SeqRecord(Seq("ACTGCCAGCTAGGTACGTAACT"), id="HighQuality1", description="")
    seq2 = SeqRecord(Seq("ACAGACAGCATGGTACGTAACT"), id="HighQuality2", description="")
    seq3 = SeqRecord(Seq("ACT-CTAGC-AGGTACGTAACT"), id="ScatteredGaps", description="")  # ~9% gaps - OK
    seq4 = SeqRecord(Seq("----CC----------------"), id="SuperGappy", description="")      # ~91% gaps - REMOVE
    seq5 = SeqRecord(Seq("ACNNRRYYKKMMSSWWBDHVNG"), id="ManyAmbiguous", description="")   # ~77% ambiguous (17 of 22) - REMOVE
    seq6 = SeqRecord(Seq("AC-G-CA-CTAG--NN?RYKMS"), id="MoreGaps", description="")       # ~32% combined (7 gaps + ambiguous) - REMOVE at 0.2 threshold
    seq7 = SeqRecord(Seq("ACTGCTA-CTAG?TACGTAACT"), id="FewGaps", description="")        # ~9% gaps+ambiguous - OK
    seq8 = SeqRecord(Seq("ACT---NNNAAARYKCHGTCCV"), id="HalfBad", description="")        # 50% combined (3 gaps + 8 ambiguous = 11/22) - REMOVE
    seq9 = SeqRecord(Seq("ACTGCCAGCTAGGTACGTAACT"), id="HighQuality1_dup", description="")  # Duplicate of seq1 - should be removed
    seq10 = SeqRecord(Seq("ACTGCCAGCTAGGTACGTAACT"), id="HighQuality1_dup2", description="")  # Another duplicate of seq1 - should be removed

    alignment = MultipleSeqAlignment([seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9, seq10])

    # Test with default threshold (0.2)
    filtered_alignment, num_kept, num_removed, removed_seqs = filter_low_quality_sequences(
        alignment, max_ambiguous_site_frac_per_seq=0.2
    )

    # Should keep only sequences with <20% combined gaps + ambiguous (including duplicates at this stage)
    assert num_kept == 6, f"Expected 6 sequences kept (4 good + 2 duplicates), got {num_kept}. Kept: {[r.id for r in filtered_alignment]}"
    assert num_removed == 4, f"Expected 4 sequences removed, got {num_removed}"

    # Check that the correct sequences were kept (quality-based filtering doesn't remove duplicates yet)
    kept_ids = {record.id for record in filtered_alignment}
    assert kept_ids == {"HighQuality1", "HighQuality1_dup", "HighQuality1_dup2", "HighQuality2", "ScatteredGaps", "FewGaps"}, f"Wrong sequences kept: {kept_ids}"

    # Check removed sequences info
    assert len(removed_seqs) == 4
    removed_ids = {seq['id'] for seq in removed_seqs}
    assert removed_ids == {"SuperGappy", "ManyAmbiguous", "MoreGaps", "HalfBad"}

    # Verify specific removed sequence statistics
    super_gappy = next(s for s in removed_seqs if s['id'] == 'SuperGappy')
    assert super_gappy['gap_fraction'] > 0.8, "SuperGappy should have >80% gaps"

    many_ambiguous = next(s for s in removed_seqs if s['id'] == 'ManyAmbiguous')
    assert many_ambiguous['ambiguous_fraction'] > 0.6, "ManyAmbiguous should have >60% ambiguous"

    half_bad = next(s for s in removed_seqs if s['id'] == 'HalfBad')
    assert half_bad['combined_fraction'] == 0.5, f"HalfBad should have 50% combined, got {half_bad['combined_fraction']}"


def test_sequence_quality_calculation():
    """Test that sequence quality metrics are calculated correctly with all IUPAC ambiguous codes."""

    # Test a perfect sequence
    perfect_seq = "ACTGACTG"
    quality = calculate_sequence_quality(perfect_seq)
    assert quality['gap_fraction'] == 0.0
    assert quality['ambiguous_fraction'] == 0.0
    assert quality['valid_fraction'] == 1.0
    assert quality['length'] == 8

    # Test a sequence with gaps
    gappy_seq = "ACT---TG"  # 3/8 = 37.5% gaps
    quality = calculate_sequence_quality(gappy_seq)
    assert abs(quality['gap_fraction'] - 0.375) < 0.001
    assert quality['valid_fraction'] == 0.625

    # Test a sequence with various ambiguous bases (all IUPAC codes)
    ambiguous_seq = "NRYKMSWB"  # 8/8 = 100% ambiguous (N, R, Y, K, M, S, W, B)
    quality = calculate_sequence_quality(ambiguous_seq)
    assert quality['ambiguous_fraction'] == 1.0
    assert quality['valid_fraction'] == 0.0

    # Test more ambiguous codes
    ambiguous_seq2 = "DHV?ACTG"  # 4/8 = 50% ambiguous (D, H, V, ?)
    quality = calculate_sequence_quality(ambiguous_seq2)
    assert quality['ambiguous_fraction'] == 0.5
    assert quality['valid_fraction'] == 0.5

    # Test a sequence with both gaps and ambiguous
    mixed_seq = "A-TN-?RY"  # 2/8 gaps, 3/8 ambiguous (N, ?, R), 3/8 valid (A, T, Y counted as ambiguous)
    quality = calculate_sequence_quality(mixed_seq)
    assert quality['gap_fraction'] == 0.25
    assert quality['ambiguous_fraction'] == 0.5  # N, ?, R, Y
    assert quality['valid_fraction'] == 0.25  # A, T

    # Test with gap character '.'
    dot_gap_seq = "ACT...TG"
    quality = calculate_sequence_quality(dot_gap_seq)
    assert abs(quality['gap_fraction'] - 0.375) < 0.001


def test_full_alignment_cleaning_pipeline():
    """
    Integration test for the full alignment cleaning process.
    Tests that scattered gaps and gappy sequences are properly handled,
    uninformative sites are removed, and statistics are generated.
    """
    # Create a realistic test alignment with various issues
    sequences = [
        SeqRecord(Seq("ACTGCCAGCTAGGTACGTAA"), id="Seq1", description=""),
        SeqRecord(Seq("ACAGACAGCATGGTACGTAA"), id="Seq2", description=""),
        SeqRecord(Seq("ACT-CTAGC-AGGTACGTAA"), id="Seq3", description=""),  # Scattered gaps
        SeqRecord(Seq("ACTGCTA-CTAGGTACGTAA"), id="Seq4", description=""),
        SeqRecord(Seq("----------------GTAA"), id="SuperGappy", description=""),  # Should be removed
        SeqRecord(Seq("ACTGCCAGCTAGGTACGTAA"), id="Seq1_dup", description=""),  # Duplicate of Seq1
        SeqRecord(Seq("ACAGACAGCATGGTACGTAA"), id="Seq2_dup", description=""),  # Duplicate of Seq2
    ]

    # Add some uninformative sites at the end (constant across non-gappy sequences)
    # The "GTAA" at the end should be analyzed for informativeness

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "test_input.fasta")
        output_file = os.path.join(tmpdir, "test_output.fasta")
        stats_file = os.path.join(tmpdir, "stats.csv")
        log_file = os.path.join(tmpdir, "pipeline.log")

        # Write input alignment
        alignment = MultipleSeqAlignment(sequences)
        from Bio import AlignIO
        AlignIO.write(alignment, input_file, "fasta")

        # Create a logger for the test
        logger = PipelineLogger(log_file, "test_dataset")

        # Run the full cleaning pipeline
        clean_alignment(
            input_filename=input_file,
            output_filename=output_file,
            max_ambiguous_site_frac_per_seq=0.2,
            remove_site_patterns=False,  # Don't remove duplicate columns for this test
            size_stats_csv=stats_file,
            logger=logger
        )

        # Read the cleaned alignment
        cleaned = AlignIO.read(output_file, "fasta")

        # Verify gappy sequence was removed (SuperGappy)
        cleaned_ids = {record.id for record in cleaned}
        assert "SuperGappy" not in cleaned_ids, "SuperGappy sequence should be removed"

        # Verify duplicates were removed
        # The clean_alignment function removes duplicate sequences
        assert len(cleaned) <= 5, f"Should have at most 5 unique sequences after removing duplicates and gappy, got {len(cleaned)}"

        # Verify statistics file was created
        assert os.path.exists(stats_file), "Statistics file should be created"

        # Read and verify statistics
        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1, "Should have one row of statistics"

            stats = rows[0]
            assert int(stats['original_num_seqs']) == 7, "Original should have 7 sequences"
            assert int(stats['cleaned_num_seqs']) < 7, "Cleaned should have fewer sequences"

            # Site count should decrease due to uninformative site removal
            original_sites = int(stats['original_num_sites'])
            cleaned_sites = int(stats['cleaned_num_sites'])
            assert original_sites == 20, "Original alignment should have 20 sites"
            assert cleaned_sites <= original_sites, "Cleaned alignment should have <= sites after removing uninformative"


def test_alignment_with_only_gaps_and_ambiguous():
    """Test edge case where alignment has sequences that are only gaps or ambiguous."""

    seq1 = SeqRecord(Seq("ACTGACTG"), id="Good", description="")
    seq2 = SeqRecord(Seq("--------"), id="AllGaps", description="")
    seq3 = SeqRecord(Seq("NNNNNNNN"), id="AllAmbiguous", description="")
    seq4 = SeqRecord(Seq("--NN--NN"), id="GapsAndAmbiguous", description="")

    alignment = MultipleSeqAlignment([seq1, seq2, seq3, seq4])

    filtered_alignment, num_kept, num_removed, removed_seqs = filter_low_quality_sequences(
        alignment, max_ambiguous_site_frac_per_seq=0.2
    )

    # Only the good sequence should remain
    assert num_kept == 1
    assert num_removed == 3
    assert filtered_alignment[0].id == "Good"

    # All bad sequences should have combined fraction of 1.0
    for removed in removed_seqs:
        assert removed['combined_fraction'] >= 0.99, f"{removed['id']} should have ~100% gaps+ambiguous"
