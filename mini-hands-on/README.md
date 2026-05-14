# Mini Hands-On

This folder contains a compact DLI-style hands-on track adapted for the
OpenHackathon bootcamp. It keeps the shorter DLI notebook flow while using the
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

The heavy DLI scoring cache and ReaSyn MCP server are not duplicated here. The
mini track uses the repository-level `scoring/`, `data/`, and `cdk_oracle/`
directories. The ReaSyn section is optional and skips cleanly unless a ReaSyn
MCP service is available.
