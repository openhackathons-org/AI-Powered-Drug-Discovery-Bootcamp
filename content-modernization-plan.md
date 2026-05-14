# Content Modernization Plan

This repository currently teaches MolMIM-driven molecule generation and a CDK4
inhibitor challenge with Boltz-2 validation. The next version should make three
changes explicit:

1. Treat local NIM deployment as part of the learner workflow, not as an
   external prerequisite.
2. Support Apptainer/Singularity as the primary cluster runtime, with Docker as
   a convenience path for local workstations.
3. Align lesson structure and language with the locally cloned DLI source at
   `../x-hx-20-v1`.

## Proposed Lesson Flow

1. **Environment and Services**
   - Explain NGC authentication, cache placement, GPU allocation, and service
     health checks.
   - Provide both Docker and Apptainer commands.
   - Make the expected endpoints visible: MolMIM on `8001`, Boltz-2 on `8000+`.

2. **MolMIM Fundamentals**
   - Keep generation, embedding, clustering, and interpolation as standalone
     notebooks.
   - Add one early "offline fallback" cell that loads saved sample responses so
     learners can keep reading if a NIM is still warming up.

3. **Oracle-Guided Optimization**
   - Show lightweight local property oracles first.
   - Introduce service-backed oracles only after health checks pass.
   - Make CMA-ES settings small by default and call out where to scale them.

4. **Boltz-2 Validation**
   - Use precomputed MSA assets by default.
   - Run a short single-endpoint validation first.
   - Add the multi-endpoint path only after the single endpoint succeeds.

5. **Challenge**
   - Separate learner-facing scoring from organizer-facing final evaluation.
   - Keep real Boltz-2 evaluation in the final path and mock/demo evaluation in
     a clearly marked smoke-test path.

## DLI Comparison Status

Compared against:

```bash
../x-hx-20-v1/task1/task
```

Ported:

- DLI notebook flow into the existing OpenHackathon notebook locations.
- Multi-endpoint Boltz-2 health checking and endpoint balancing.
- RDKit `rdFingerprintGenerator` migration for Morgan fingerprints.
- Novelty-score caching during optimization.
- ReaSyn visualization helper, with notebook cells made optional.
- `known_ligands.csv` used by the updated Boltz-2 validation notebook.
- Boltz-2 Python client requirement updated to `>=0.5.2.post1`.

Adapted for OpenHackathon/HPC:

- DLI Docker service names such as `molmim`, `boltz2-1`, and `boltz2-2` were
  replaced with localhost endpoints.
- Apptainer/Singularity is documented as the primary cluster runtime.
- Multi-endpoint Boltz-2 examples use `8010+` to avoid colliding with MolMIM on
  `8001`.
- ReaSyn is optional because a Singularity/Apptainer deployment path for the MCP
  server is not included yet.

Still worth doing:

- Build or document a local Apptainer/Singularity deployment for ReaSyn MCP.
- Run the updated notebooks end-to-end against live MolMIM and Boltz-2 NIMs.
- Decide whether to preserve the older deep-dive MolMIM tutorial sequence or
  fully converge on the shorter DLI flow.
