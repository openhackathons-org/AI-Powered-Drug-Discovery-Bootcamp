# Mini Hands-On

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


This folder contains a compact hands-on track for
AI-Powered-Drug-Discovery-Bootcamp. It keeps a shorter guided notebook flow while using the
repository root for shared code, data, scoring utilities, and local NIM endpoint
configuration.

## Notebook Order

1. `00_Introduction.ipynb`
2. `01_NIM_Setup.ipynb`
3. `02_Overview-Designing_CDK4_Inhibitors.ipynb`
4. `03_Hands-On_CDK_Inhibitor_Design.ipynb`
5. `04_Optional_MolMIM_CMA-ES_Controlled_Generation.ipynb`
6. `05_Optional_Boltz2_validation.ipynb`

## Runtime Notes

Start services from the repository root before running the notebooks:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
scripts/openhackathon_services.sh start --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

The notebooks load `.openhackathon-nims.env` automatically when possible. The
hands-on CDK design and Boltz-2 validation notebooks default to demo mode so
they complete in a workshop-friendly amount of time. Set
`OPENHACKATHON_DEMO_MODE=0` for a larger run.

The large scoring cache and ReaSyn MCP server are not duplicated here. The
mini track uses the repository-level `scoring/`, `data/`, and `cdk_oracle/`
directories. The ReaSyn section is optional and skips cleanly unless a ReaSyn
MCP service is available.
