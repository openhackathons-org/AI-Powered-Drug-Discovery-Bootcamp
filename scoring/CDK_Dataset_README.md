# CDK Inhibitors Test Dataset

This dataset contains 20 compounds for testing the OpenHackathon evaluation pipeline with known CDK inhibitors and decoy compounds.

## Dataset Files

- **`cdk_inhibitors_dataset.csv`**: Full dataset with annotations
- **`cdk_test_compounds.csv`**: SMILES-only format for direct evaluation

## Dataset Composition

### CDK Inhibitors (10 compounds)

#### FDA-Approved CDK4/6 Inhibitors
1. **Palbociclib** - CDK4/6 inhibitor for breast cancer
2. **Ribociclib** - CDK4/6 inhibitor for breast cancer  
3. **Abemaciclib** - CDK4/6 inhibitor for breast cancer

#### Clinical/Research CDK Inhibitors
4. **SY-1365** - CDK7 inhibitor
5. **LEE011 analog** - CDK4/6 inhibitor analog
6. **Roscovitine analog** - CDK2 inhibitor
7. **Purvalanol analog** - Pan-CDK inhibitor
8. **R-roscovitine derivative** - CDK2 inhibitor
9. **AG-024322 analog** - CDK inhibitor
10. **PD-0332991 analog** - CDK4/6 inhibitor

### Decoy Compounds (10 compounds)

#### Common Pharmaceuticals (Non-CDK)
1. **Aspirin** - Anti-inflammatory
2. **Ibuprofen** - NSAID
3. **Chlorpheniramine** - Antihistamine
4. **Propranolol** - Beta blocker

#### Chemical Decoys
5. **Simple aromatic ketone** - Structural decoy
6. **Palmitic acid** - Fatty acid
7. **Phenylpyrazolone derivative** - Non-kinase active
8. **Quinoline derivative** - Different target class
9. **Simple imidazole** - Small molecule decoy
10. **Mechlorethamine analog** - Alkylating agent

## Expected Results

### CDK Inhibitors (Should Show):
- **High affinity** for CDK4 and CDK6 (low IC50, high pIC50)
- **Variable affinity** for CDK11 (depends on selectivity)
- **Good drug-likeness** (QED > 0.5)
- **Reasonable synthetic accessibility**

### Decoy Compounds (Should Show):
- **Low affinity** for all CDK targets (high IC50, low pIC50)
- **Variable drug-likeness** (some designed to be drug-like)
- **May have good SA scores** (simpler structures)

## Usage Examples

```bash
# Test with the CDK-focused dataset
python evaluate_submission_parallel_no_mock.py cdk_test_compounds.csv CDK_TestSet \
    --endpoints 8000,8001,8002 \
    --max-workers 6 \
    --skip-toxicity --skip-novelty \
    --verbose

# Compare with original demo compounds
python evaluate_submission_parallel_no_mock.py demo_compounds.csv DemoSet \
    --endpoints 8000,8001,8002 \
    --max-workers 6 \
    --skip-toxicity --skip-novelty \
    --verbose
```

## Expected Performance Benchmarks

### CDK Inhibitors (Top 10):
- **pIC50 range**: 6.0 - 9.0 for CDK4/CDK6
- **Selectivity ratio**: Variable (1.0 - 10.0)
- **QED**: 0.4 - 0.8 (drug-like)
- **Composite scores**: Should rank in top 50%

### Decoys (Bottom 10):
- **pIC50 range**: 4.0 - 6.0 for all targets
- **Selectivity ratio**: ~1.0 (no selectivity)
- **QED**: Variable (0.2 - 0.7)
- **Composite scores**: Should rank lower

## Validation Notes

This dataset serves as a **positive/negative control** for the evaluation pipeline:

✅ **Good evaluation system should**:
- Rank known CDK inhibitors higher than decoys
- Show clear binding affinity differences
- Demonstrate selectivity for CDK4/6 vs CDK11

❌ **Poor evaluation system might**:
- Show random rankings
- Similar affinities for all compounds
- No correlation with known activity

Use this dataset to validate that your Boltz2 predictions and scoring system are working correctly! 🎯
