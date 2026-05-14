# Download and Create ChEMBL Fingerprints Database

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


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