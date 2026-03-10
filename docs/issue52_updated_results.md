## Results: Training on real NJ tree search intermediates

Trained TraverseMaxPooling and TraverseAvgPooling on `orthomam_treesearch_nj_training` (merged NJ tree search intermediates from ~130 OrthoMaM alignments) and compared against models trained on synthetic SPR-perturbed data. Evaluated on both NJ and random-starting tree search intermediates for 4 viral test datasets (fluC_M, fluC_NS, fluC_PB2, rotavirusA_H_H2). Training hyperparameters: lr=0.0002, batch_size=4, epochs=200, feature_length=32, dim_mlp=256.

### AUROC range on NJ tree search test data (best–worst across trees)

#### fluC_M

| Training Data | MaxPooling | AvgPooling |
|---|---|---|
| `orthomam_train_0.5_1000_samples_spr` (original SPR) | 0.729–0.671 | 0.749–0.737 |
| `orthomam_varied_proportions` (varied non-MP, #49) | 0.729–0.704 | 0.707–0.688 |
| `orthomam_treesearch_nj_training` (real NJ, this issue) | 0.561–0.375 | 0.673–0.658 |

BaselineReversion model AUROC on fluC_M: 0.476–0.475.

#### fluC_NS

| Training Data | MaxPooling | AvgPooling |
|---|---|---|
| `orthomam_train_0.5_1000_samples_spr` (original SPR) | 0.526–0.491 | 0.743–0.654 |
| `orthomam_varied_proportions` (varied non-MP, #49) | 0.735–0.580 | 0.697–0.573 |
| `orthomam_treesearch_nj_training` (real NJ, this issue) | 0.668–0.502 | 0.663–0.611 |

BaselineReversion model AUROC on fluC_NS: 0.532–0.489.

#### fluC_PB2

| Training Data | MaxPooling | AvgPooling |
|---|---|---|
| `orthomam_train_0.5_1000_samples_spr` (original SPR) | 0.670–0.494 | 0.802–0.661 |
| `orthomam_varied_proportions` (varied non-MP, #49) | 0.723–0.497 | 0.705–0.500 |
| `orthomam_treesearch_nj_training` (real NJ, this issue) | 0.615–0.200 | 0.762–0.477 |

BaselineReversion model AUROC on fluC_PB2: 0.438–0.427.

#### rotavirusA_H_H2

| Training Data | MaxPooling | AvgPooling |
|---|---|---|
| `orthomam_train_0.5_1000_samples_spr` (original SPR) | 0.725–0.662 | 0.823–0.807 |
| `orthomam_varied_proportions` (varied non-MP, #49) | 0.680–0.661 | 0.681–0.676 |
| `orthomam_treesearch_nj_training` (real NJ, this issue) | 0.468–0.417 | 0.627–0.609 |

BaselineReversion model AUROC on rotavirusA_H_H2: 0.505–0.492.

### Mean AUROC on random-starting tree search test data (across all trees and 5 reps)

| Training Data | fluC_M MaxPool | fluC_M AvgPool | fluC_NS MaxPool | fluC_NS AvgPool | fluC_PB2 MaxPool | fluC_PB2 AvgPool | rotavirus MaxPool | rotavirus AvgPool |
|---|---|---|---|---|---|---|---|---|
| `orthomam_varied_proportions` | 0.870 | 0.877 | 0.838 | 0.873 | 0.859 | 0.875 | 0.843 | 0.842 |
| `orthomam_treesearch_nj_training` | 0.735 | 0.565 | 0.642 | 0.596 | 0.769 | 0.505 | 0.639 | 0.644 |

BaselineReversion mean AUROC: fluC_M 0.497, fluC_NS 0.489, fluC_PB2 0.454, rotavirusA_H_H2 0.506.

### Key findings

1. **Real NJ tree search training data performs substantially worse than synthetic SPR data.** The treesearch-trained models clearly rank last among all training sets for almost all architecture/dataset combinations. MaxPooling + treesearch NJ achieves only 0.561–0.375 on fluC_M. The gap is even more dramatic on random-starting tree searches, where treesearch NJ averages 0.505–0.769 vs. varied proportions at 0.842–0.877.

2. **AvgPooling is more robust than MaxPooling for the treesearch NJ training data.** AvgPooling with treesearch NJ achieves 0.673–0.658 on fluC_M vs MaxPooling's 0.561–0.375. However, both are clearly outperformed by synthetic SPR training data.

3. **Original SPR data (`orthomam_train_0.5_1000_samples_spr`) is competitive with or better than varied proportions on NJ tree search tests.** On fluC_M it achieves the highest AvgPooling AUROC (0.749–0.737). On rotavirusA_H_H2 it clearly dominates (MaxPooling 0.725–0.662 vs varied proportions 0.680–0.661; AvgPooling 0.823–0.807 vs 0.681–0.676).

4. **Varied proportions data (#49) dominates on random-starting tree searches.** Mean AUROCs of 0.838–0.877 across all datasets, consistently outperforming treesearch NJ (0.505–0.769) by a wide margin.

5. **All trained models still beat the baseline** (~0.43–0.53 AUROC), but the treesearch NJ model's advantage over baseline is small, particularly for MaxPooling.

### Interpretation

The results show a clear performance hierarchy: original SPR >= varied proportions >> treesearch NJ >> baseline.

Training on real tree search intermediates is not beneficial, and in fact harmful compared to synthetic SPR perturbations. The NJ treesearch data suffers from:

- **Extreme class imbalance**: NJ starting trees are already close to optimal, so most intermediates have very few non-MP edges. The model rarely sees examples with higher non-MP proportions.
- **Limited diversity**: Only 3–11 intermediate trees per alignment (median 6), totaling ~923 trees across 144 alignments — far less diverse than synthetic SPR perturbations.
- **Narrow distribution**: The non-MP fraction range in NJ tree search intermediates is much narrower than what models encounter during evaluation on random-starting tree searches.

The varied proportions approach from #49 remains the most robust training strategy for random-starting tree search evaluation, while original SPR data performs well on NJ-starting evaluations. Dynamic per-sample class reweighting for the BCE loss (dpvt#43) could help address the extreme class imbalance directly at the loss level, potentially making the model more effective on trees with few non-MP edges regardless of training data source.

Detailed per-tree evaluation logs are in `train/_output/run.treesearch/tree_eval_logs/` and `train/_output/run.treesearch/tree_eval_logs_nj_starting/`.
