import os
import sys
import subprocess
import tempfile
import shutil
import pytest
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


def run_command(command):
    """Run a shell command and return its output"""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


# Get the scripts directory (one level up from tests directory)
def get_scripts_dir():
    """Get the path to the scripts directory"""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(tests_dir), "scripts")


class TestCleanData:
    @pytest.fixture
    def test_environment(self):
        """Set up test environment with test files"""
        scripts_dir = get_scripts_dir()
        clean_data_script = os.path.join(scripts_dir, "clean_data.py")

        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_fasta = os.path.join(tmp_dir, "test.fasta")
            output_fasta = os.path.join(tmp_dir, "output.fasta")
            stats_file = os.path.join(tmp_dir, "stats.txt")

            # Create a test alignment with:
            # - 5 sequences (with 2 duplicates)
            # - 10 sites (with some uninformative)
            sequences = [
                SeqRecord(Seq("AACGTACGTT"), id="seq1"),
                SeqRecord(Seq("AACGTACGTT"), id="seq2"),  # Duplicate of seq1
                SeqRecord(Seq("GGCGTACGCC"), id="seq3"),
                SeqRecord(Seq("TTCGTACGAA"), id="seq4"),
                SeqRecord(Seq("TTCGTACGAA"), id="seq5"),  # Duplicate of seq4
            ]

            alignment = MultipleSeqAlignment(sequences)
            # Write test alignment to FASTA
            AlignIO.write(alignment, test_fasta, "fasta")

            yield {
                "script": clean_data_script,
                "test_fasta": test_fasta,
                "output_fasta": output_fasta,
                "stats_file": stats_file,
                "tmp_dir": tmp_dir,
            }

    def test_basic_cleaning(self, test_environment):
        """Test that the script removes duplicates and uninformative sites"""
        env = test_environment

        # Run without specifying target dimensions
        cmd = f"python {env['script']} {env['test_fasta']} {env['output_fasta']} {env['stats_file']}"
        returncode, stdout, stderr = run_command(cmd)

        # Check command success
        assert returncode == 0, f"Command failed with: {stderr}"

        # Check files exist
        assert os.path.exists(env["output_fasta"]), "Output FASTA not created"
        assert os.path.exists(env["stats_file"]), "Stats file not created"

        # Read the cleaned alignment
        cleaned_alignment = AlignIO.read(env["output_fasta"], "fasta")

        # Verify duplicate sequences are removed
        assert (
            len(cleaned_alignment) == 3
        ), f"Expected 3 sequences, got {len(cleaned_alignment)}"

        # Read the stats file
        with open(env["stats_file"], "r") as f:
            length_str, seqs_str = f.read().strip().split(",")
            cleaned_length = int(length_str)
            cleaned_seqs = int(seqs_str)

        # Verify stats file matches alignment
        assert cleaned_length == cleaned_alignment.get_alignment_length()
        assert cleaned_seqs == len(cleaned_alignment)

    def test_target_dimensions(self, test_environment):
        """Test that the script trims to specified target dimensions"""
        env = test_environment

        # Target dimensions
        target_length = 4
        target_seqs = 2

        # Run with target dimensions
        cmd = f"python {env['script']} {env['test_fasta']} {env['output_fasta']} {env['stats_file']} {target_length} {target_seqs}"
        returncode, stdout, stderr = run_command(cmd)

        # Check command success
        assert returncode == 0, f"Command failed with: {stderr}"

        # Read the cleaned alignment
        cleaned_alignment = AlignIO.read(env["output_fasta"], "fasta")

        # Verify dimensions match targets
        assert (
            cleaned_alignment.get_alignment_length() <= target_length
        ), f"Alignment length {cleaned_alignment.get_alignment_length()} exceeds target {target_length}"
        assert (
            len(cleaned_alignment) <= target_seqs
        ), f"Sequence count {len(cleaned_alignment)} exceeds target {target_seqs}"

        # Read the stats file
        with open(env["stats_file"], "r") as f:
            length_str, seqs_str = f.read().strip().split(",")
            cleaned_length = int(length_str)
            cleaned_seqs = int(seqs_str)

        # Verify stats file matches alignment
        assert cleaned_length == cleaned_alignment.get_alignment_length()
        assert cleaned_seqs == len(cleaned_alignment)


@pytest.mark.slow  # Mark this test as slow so it can be skipped with -m "not slow"
class TestCreateAlignments:
    @pytest.fixture
    def test_environment(self):
        """Create a temporary test environment with all necessary files and directories"""
        scripts_dir = get_scripts_dir()
        project_root = os.path.dirname(os.path.dirname(scripts_dir))

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create necessary directory structure in temp dir
            temp_data_dir = os.path.join(temp_dir, "data", "simulated_alignments")
            os.makedirs(temp_data_dir, exist_ok=True)

            # Create a temp scripts directory and copy necessary scripts
            temp_scripts_dir = os.path.join(temp_dir, "scripts")
            os.makedirs(temp_scripts_dir, exist_ok=True)

            # Copy clean_data.py to temp scripts dir
            clean_data_path = os.path.join(scripts_dir, "clean_data.py")
            temp_clean_data_path = os.path.join(temp_scripts_dir, "clean_data.py")
            shutil.copy2(clean_data_path, temp_clean_data_path)

            # Create a simplified version of generate_configs.py
            temp_generate_configs_path = os.path.join(
                temp_scripts_dir, "generate_configs.py"
            )
            with open(temp_generate_configs_path, "w") as f:
                f.write(
                    """#!/usr/bin/env python3
# A simplified version of generate_configs.py for testing
import sys
# Just print the arguments for testing
print(f"generate_configs called with: {sys.argv}")
"""
                )

            # Make it executable
            os.chmod(temp_generate_configs_path, 0o755)

            # Create a modified version of the alignment creation script
            create_script = os.path.join(scripts_dir, "create_alisim_alignments.sh")
            temp_create_script = os.path.join(temp_dir, "create_alisim_alignments.sh")

            with (
                open(create_script, "r") as f_in,
                open(temp_create_script, "w") as f_out,
            ):
                content = f_in.read()

                # Modify parameters for a quick test
                content = content.replace("num_alignments=500", "num_alignments=1")
                content = content.replace(
                    "num_sequences_list=(10 20 30 50)", "num_sequences_list=(5)"
                )
                content = content.replace(
                    "alignment_length_list=(100)", "alignment_length_list=(20)"
                )

                # Set script_dir to the temp scripts dir
                content = content.replace(
                    'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                    f'script_dir="{temp_scripts_dir}"',
                )

                # Simplify the base_directory assignment to point directly to the temp dir
                content = content.replace(
                    'base_directory="$(cd "${script_dir}/../../data/simulated_alignments" && pwd)/clean_alignment_',
                    f'base_directory="{temp_data_dir}/clean_alignment_',
                )

                f_out.write(content)

            # Make the script executable
            os.chmod(temp_create_script, 0o755)

            yield {
                "temp_dir": temp_dir,
                "scripts_dir": temp_scripts_dir,
                "data_dir": temp_data_dir,
                "create_script": temp_create_script,
            }

    def test_alignment_generation(self, test_environment):
        """Test that the script generates alignments with the correct dimensions"""
        env = test_environment

        # Run the test script
        cmd = f"bash {env['create_script']}"
        returncode, stdout, stderr = run_command(cmd)

        # Print output for debugging
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

        # Check command success
        assert returncode == 0, f"Command failed with: {stderr}"

        # Check if the output directory was created
        output_base = os.path.join(
            env["data_dir"], "clean_alignment_5_seq_20_sites_1_algnmnts"
        )
        assert os.path.exists(
            output_base
        ), f"Output directory {output_base} was not created"

        # Check alignment file
        alignment_path = os.path.join(output_base, "alignment_1/alignment_1.fasta")
        assert os.path.exists(
            alignment_path
        ), f"Alignment file {alignment_path} was not created"

        # Verify alignment dimensions
        alignment = AlignIO.read(alignment_path, "fasta")
        assert (
            alignment.get_alignment_length() == 20
        ), f"Alignment has length {alignment.get_alignment_length()}, expected 20"
        assert (
            len(alignment) == 5
        ), f"Alignment has {len(alignment)} sequences, expected 5"


if __name__ == "__main__":
    # This allows running with python directly or with pytest
    pytest.main(["-v", __file__])
