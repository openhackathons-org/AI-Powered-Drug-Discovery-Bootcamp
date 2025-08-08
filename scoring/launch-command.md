# Download and Create ChEMBL Fingerprints Database

```bash
python create_chembl_database_parallel.py --chunk-size 20000 --workers 16
```

# Run Evaluation on a Test Submission 

```bash
python evaluate_submission_parallel_no_mock.py cdk_test_compounds.csv CDK_Validation \
    --endpoints 8000,8001,8002,8003 \
    --max-workers 8 \
    --skip-toxicity --skip-novelty \
    --verbose
```