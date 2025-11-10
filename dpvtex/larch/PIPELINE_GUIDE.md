# DPVT Training Data Generation Workflow

This workflow processes empirical alignments in three separate phases to generate training data for DPVT.

**Key Feature**: Phases 1 & 2 share a **unified config file**, while Phase 3 uses simple split-specific configs.

## Overview

```
Phase 1: Preprocessing          Phase 2: Dataset Preparation       Phase 3: Training Data Generation
(preprocess_alignments)         (prepare_datasets)                 (generate_dpvt_input)
─────────────────────           ────────────────────────           ─────────────────────────────
Raw alignments                  Filtered datasets with             Larch processing
↓                               symlinks                           ↓
Check unequal lengths           ↓                                  Run larch-usher
↓                               Filter by site_ratio               ↓
Clean sequences                 ↓                                  Extract training data
↓                               Split into train/test              ↓
Generate stats                  ↓                                  Aggregate results
↓                               Symlink to originals
alignment_size_stats.csv                                           Training dataset ready
```

## Two-Stage Filtering Strategy

This workflow uses **two independent quality filters**:

1. **Phase 1 - Sequence-level filtering** (`max_ambiguous_site_frac_per_seq`):
   - Removes **individual sequences** from each alignment if they have too many gaps/ambiguous characters
   - Example: With `max_ambiguous_site_frac_per_seq: 0.2`, sequences with >20% gaps/ambiguous chars are removed
   - The alignment continues to Phase 2 with fewer sequences

2. **Phase 2 - Alignment-level filtering** (`min_frac_sites_retained`):
   - Excludes **entire alignments** that lost too many sites during preprocessing
   - Example: With `min_frac_sites_retained: 0.8`, alignments that retained <80% of their original sites are excluded
   - `fraction_sites_retained = cleaned_sites / original_sites`

**These parameters are independent**: An alignment can lose sequences (Phase 1) but still be included if it retained enough sites (Phase 2).

## Configuration

**Unified Config** (`configs/{dataset_name}_prepare.yaml`) - Used for Phases 1 & 2:
```yaml
dataset_name: "{dataset_name}"
input_data: "path/to/your/alignments"

# Phase 1 settings - Sequence quality filtering
max_ambiguous_site_frac_per_seq: 0.2     # Remove sequences with >20% gaps/ambiguous chars
remove_duplicate_site_patterns: false

# Phase 2 settings - Alignment quality filtering
output_datasets: "../../data"
min_frac_sites_retained: 0.8             # Exclude alignments that lost >20% of sites
create_train_test_split: true
test_fraction: 0.2

# Phase 3 settings (reference only)
larch_build: "../../../larch/build"
larch_output: "../../data"
edge_distribution: "constant"
num_cores: 8
```

**Phase 3 Configs** (`configs/{dataset_name}_train.yaml` and `configs/{dataset_name}_test.yaml`) - Simple split-specific configs

**Example**: For the simulated alignments included in this repo, see `configs/simulated_15seq_20sites_100algnmnts_prepare.yaml` and the corresponding train/test configs.

## Phase 1: Preprocessing

**Purpose**: Clean raw alignments and generate quality statistics

**What it does**:
- **Removes low-quality sequences** from each alignment (sequences with too many gaps/ambiguous characters)
- Optionally removes duplicate site patterns
- Generates quality statistics for each alignment

**Input**: Raw alignment directories with `.fasta` or `.nex` files

**Output**:
- Cleaned alignments (`input.fasta`) - same alignments, but with some sequences removed
- Quality statistics (`size_stats.csv`, `alignment_size_stats.csv`)
- Exclusion list (`unequal_length_alignments.txt`)

**Run**:
```bash
cd dpvtex/larch
snakemake --snakefile preprocess_alignments.snakefile \
  --configfile configs/{dataset_name}_prepare.yaml \
  --cores 4
```

**Example** (using included simulated data):
```bash
snakemake --snakefile preprocess_alignments.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_prepare.yaml \
  --cores 4
```

**Key outputs to review**:
- `alignment_size_stats.csv` - Review to decide `min_frac_sites_retained` threshold
- Check `site_ratio` column: ratio of cleaned_sites / original_sites

**Manual Checkpoint**: Review the statistics before proceeding to Phase 2!

## Phase 2: Dataset Preparation

**Purpose**: Filter alignments by quality and split into train/test sets

**What it does**:
- **Excludes entire alignments** that lost too many sites during Phase 1 preprocessing
- Splits remaining alignments into train/test sets
- Creates symlinks (no data duplication)

**Input**: Preprocessed data from Phase 1

**Output**:
- Filtered dataset directory with symlinks
- Train/test split directories with symlinks
- Manifest files documenting dataset composition

**Configuration**: Uses the same unified config as Phase 1. Adjust `min_site_ratio` based on Phase 1 review.

**Run**:
```bash
cd dpvtex/larch
snakemake --snakefile prepare_datasets.snakefile \
  --configfile configs/{dataset_name}_prepare.yaml \
  --cores 1
```

**Example** (using included simulated data):
```bash
snakemake --snakefile prepare_datasets.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_prepare.yaml \
  --cores 1
```

**Output structure**:
```
output_directory/
├── {dataset_name}_filtered_{threshold}/
│   ├── manifest.txt
│   ├── alignment_1/  → symlink to original
│   └── ...
├── {dataset_name}_train_{threshold}/
│   ├── manifest.txt
│   └── ... (80% of alignments)
└── {dataset_name}_test_{threshold}/
    ├── manifest.txt
    └── ... (20% of alignments)
```

**Manual checkpoint**: Review the manifests and filtered alignment count before proceeding.

## Phase 3: Training Data Generation

**Purpose**: Run larch and generate DPVT training data

**Input**: Prepared dataset from Phase 2 (train or test directory)

**Output**: Aggregated training data pickle file

**Configuration**: Uses simple split-specific configs (`configs/{dataset_name}_train.yaml` or `configs/{dataset_name}_test.yaml`). These configs specify which prepared dataset to process.

**Run for train set**:
```bash
cd dpvtex/larch
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/{dataset_name}_train.yaml \
  --cores 8
```

**Run for test set**:
```bash
cd dpvtex/larch
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/{dataset_name}_test.yaml \
  --cores 8
```

**Example** (using included simulated data):
```bash
# Train set
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_train.yaml \
  --cores 8

# Test set
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_test.yaml \
  --cores 8
```

**Output**:
- `larch_output/larch_{dataset_name}_train_{threshold}_{edge_distribution}.p` - Training data
- `larch_output/larch_{dataset_name}_test_{threshold}_{edge_distribution}.p` - Test data
- Dataset properties CSV files

## Complete Example Workflow

```bash
# Phase 1: Preprocess simulated dataset
snakemake --snakefile preprocess_alignments.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_prepare.yaml \
  --cores 4

# MANUAL CHECKPOINT: Review alignment_size_stats.csv
# Decide on min_frac_sites_retained threshold, update config if needed

# Phase 2: Filter and split dataset
snakemake --snakefile prepare_datasets.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_prepare.yaml \
  --cores 1

# MANUAL CHECKPOINT: Review manifests in data/
# Check that filtering results make sense

# Phase 3: Generate training data for train set
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_train.yaml \
  --cores 8

# Phase 3: Generate training data for test set
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/simulated_15seq_20sites_100algnmnts_test.yaml \
  --cores 8
```

## Creating Datasets with Different Thresholds

You can easily create multiple filtered versions by creating configs with different `min_frac_sites_retained` values:

```bash
# Create configs with different thresholds using the generate_configs.py script
for ratio in 0.7 0.8 0.9; do
  # Edit the generated config to set min_frac_sites_retained: $ratio
  # Then run Phase 2 with each config
  snakemake --snakefile prepare_datasets.snakefile \
    --configfile configs/my_dataset_${ratio}_prepare.yaml \
    --cores 1
done

# All three coexist as separate directories:
# - data/my_dataset_filtered_0.7/
# - data/my_dataset_filtered_0.8/
# - data/my_dataset_filtered_0.9/
```

## Benefits of This Workflow

1. **Manual checkpoints**: Review quality stats and filter results before expensive processing
2. **Efficiency**: Symlinks avoid data duplication
3. **Flexibility**: Easy to experiment with different filter thresholds
4. **Clarity**: Each phase has clear inputs, outputs, and purpose
5. **Independent splits**: Train/test splits use system randomness, generating different splits each time
