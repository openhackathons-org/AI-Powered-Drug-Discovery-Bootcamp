# BioNeMo Bootcamp

## Welcome to the BioNeMo Bootcamp and Hackathon Challenge

NVIDIA BioNeMo is a computational drug discovery platform built for developers and data scientists. BioNeMo provides an open-source machine learning framework for building and training deep learning models, and a set of optimized and easy-to-use NIM microservices for deploying AI workflows at scale.  BioNeMo NIM microservices are constructed as containers with everything needed for efficient, portable deployment and easy API integration into enterprise-grade AI applications.

This bootcamp combines comprehensive tutorials with a cutting-edge hackathon challenge, giving you hands-on experience with NVIDIA BioNeMo's tools for AI-enabled drug discovery. You'll learn to harness the power of MolMIM, a generative model for small-molecule drug discovery.  MolMIM is a probabilistic auto-encoder trained with Mutual Information Machine (MIM) learning designed to learn an informative and clustered latent space.  By leveraging targeted optimization with scoring oracles, you will learn how to explore MolMIM's dense latent-space representation of chemical space to generate and optimize novel molecular structures.

## Repository Structure

At the top level, you'll find detailed instructions for deploying the MolMIM NIM in [`deployment.md`](deployment.md) and all the required dependencies in [`deployment-requirements.txt`](deployment-requirements.txt). Once you have the basics in place, you can follow along with the Tutorials and Challenge.

### 📚 Tutorials
The [`tutorials/`](tutorials/) folder contains everything you need to get started and background on the models and techniques used in the Challenge:
- **00_Container_Setup.ipynb**: Step-by-step deployment guide for setting up MolMIM NIM with NGC API key configuration
- **01_MolMIMGeneration.ipynb**: Fundamental MolMIM operations including unguided sampling and guided optimization using CMA-ES algorithm
- **02_ClusterMolMIMEmbeddings.ipynb**: Clustering molecular embeddings to identify molecular families and functional relationships
- **03_MolMIMInterpolation.ipynb**: Interpolating between molecules by manipulating hidden states to generate novel structures
- **04_MolMIMOracleControlledGeneration.ipynb**: Advanced controlled generation using custom oracle scoring functions with CMA-ES optimization
- **05_Suggested_Tools_for_Scoring_Oracles.ipynb**: Comprehensive guide to tools and resources for building custom scoring oracles
- **06_Boltz2_Validation.ipynb**: Binding affinity prediction validation using Boltz-2 with MSA for CDK inhibitor assessment

### 🏆 Challenge
The [`challenge/`](challenge/) folder contains the hackathon challenge where you'll apply your knowledge to solve real drug discovery problems:
- **01_Challenge_Overview.ipynb**: Complete hackathon challenge introduction with objectives, scoring methods, and evaluation criteria
- **02_The_Challenge-Designing_CDK4_Inhibitors.ipynb**: Detailed challenge specification for designing selective CDK4 inhibitors while avoiding CDK11 binding
- **03_Hands-On_CDK_Inhibitor_Design.ipynb**: End-to-end pipeline for CDK4 inhibitor design including generation, affinity prediction, and composite scoring


## Bootcamp Objectives

By the end of this workshop, participants will:

* Understand NVIDIA BioNeMo architecture and key functionalities
* Gain deep insights into MolMIM for generative small molecule design
* Develop hands-on skills to apply BioNeMo workflows using real-world, complex datasets
* Integrate advanced scoring and optimization methods to refine molecular designs


## Getting Started

1. **Setup Environment**: Follow the deployment guide in [`deployment.md`](deployment.md) or the detailed container setup in [`tutorials/00_Container_Setup.ipynb`](tutorials/00_Container_Setup.ipynb)
3. **Complete Tutorials**: Work through the set of introductory notebooks in the [`tutorials/`](tutorials/) folder
4. **Take the Challenge**: Apply your skills using the examples in the [`challenge/`](challenge/) folder

## Quick Start

```bash
# Clone and navigate to the repository
cd Bootcamp-BioNemo

# Install dependencies
pip install -r deployment_requirements.txt

# Follow deployment guide for MolMIM NIM setup
cat deployment.md

# Start with the overview Start_Here.ipynb notebook
jupyter-lab Start_Here.ipynb
```

## The Challenge

The core of this bootcamp culminates in an exciting challenge: **Accelerating Drug Discovery with NVIDIA MolMIM**. You'll harness cutting-edge AI to revolutionize drug discovery by generating and optimizing novel molecular structures with potential as therapeutic agents.

### What You'll Do:
- Generate diverse molecular structures using MolMIM
- Optimize molecules for drug-like characteristics using custom scoring oracles
- Evaluate drug potential through property prediction and binding assessment

### Key Properties to Explore:
- **Drug-likeness (QED)**: Overall suitability as a drug candidate
- **Synthesizability**: Ease of laboratory synthesis
- **Solubility**: Dissolution characteristics
- **Toxicity**: Safety assessment
- **Tanimoto Similarity**: Chemical similarity to known therapeutics

## Why Participate?

This bootcamp with challenge offers a unique opportunity to:

* Gain hands-on experience with state-of-the-art AI models for molecular design
* Deepen your understanding of computational drug discovery workflows
* Collaborate with fellow innovators and experts in the field
* Contribute to the advancement of medical science by accelerating the identification of new drugs

Prepare to innovate, experiment, and push the boundaries of what's possible in the quest for life-changing medicines!

## Attribution

This material originates from the OpenHackathons Github repository. Check out additional materials [here](https://github.com/openhackathons-org)

Don't forget to check out additional [Open Hackathons Resources](https://www.openhackathons.org/s/technical-resources) and join our [OpenACC and Hackathons Slack Channel](https://www.openacc.org/community#slack) to share your experience and get more help from the community.

## Licensing

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.

