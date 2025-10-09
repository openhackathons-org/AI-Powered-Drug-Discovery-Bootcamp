# BioNeMo Bootcamp

## Welcome to the BioNeMo Bootcamp and Hackathon Challenge

NVIDIA BioNeMo is a computational drug discovery platform built for developers and data scientists. BioNeMo provides an open-source machine learning framework for building and training deep learning models, and a set of optimized and easy-to-use NIM microservices for deploying AI workflows at scale.  BioNeMo NIM microservices are constructed as containers with everything needed for efficient, portable deployment and easy API integration into enterprise-grade AI applications.

This bootcamp combines comprehensive tutorials with a cutting-edge hackathon challenge, giving you hands-on experience with NVIDIA BioNeMo's tools for AI-enabled drug discovery. You'll learn to harness the power of MolMIM, a generative model for small-molecule drug discovery.  MolMIM is a probabilistic auto-encoder trained with Mutual Information Machine (MIM) learning designed to learn an informative and clustered latent space.  By leveraging targeted optimization with scoring oracles, you will learn how to explore MolMIM's dense latent-space representation of chemical space to generate and optimize novel molecular structures.

## Repository Structure

At the top level, you'll find detailed instructions for deploying the MolMIM NIM in [`deployment.md`](deployment.md) and all the required dependencies in [`deployment-requirements.txt`](deployment-requirements.txt). Once you have the basics in place, you can follow along with the Tutorials and Challenge.

### 📚 Tutorials
The [`tutorials/`](tutorials/) folder contains everything you need to get started:
- **Container Setup**: Step-by-step deployment guide (`container_setup.ipynb`)
- **Lab 1**: Fundamental MolMIM operations (clustering, generation, interpolation)
- **Lab 2**: Advanced techniques with custom oracles, optimization, and guided molecule generation

### 🏆 Challenge
The [`challenge/`](challenge/) folder contains the hackathon challenge where you'll apply your knowledge to solve real drug discovery problems.

## Bootcamp Objectives

By the end of this workshop, participants will:

* Understand NVIDIA BioNeMo architecture and key functionalities
* Gain deep insights into MolMIM for generative small molecule design
* Develop hands-on skills to apply BioNeMo workflows using real-world, complex datasets
* Integrate advanced optimization methods (CMA-ES with custom oracles) to refine molecular designs
* Optimize and scale workflows leveraging multi-GPU and HPC resources (optional)

## Getting Started

1. **Setup Environment**: Follow the deployment guide in [`deployment.md`](deployment.md) or the detailed container setup in [`tutorials/container_setup.ipynb`](tutorials/container_setup.ipynb)
3. **Complete Tutorials**: Work through Lab 1 and Lab 2 in the [`tutorials/`](tutorials/) folder
4. **Take the Challenge**: Apply your skills in the [`challenge/`](challenge/) folder

## Quick Start

```bash
# Clone and navigate to the repository
cd Bootcamp-BioNemo

# Install dependencies
pip install -r deployment_requirements.txt

# Follow deployment guide for MolMIM NIM setup
cat deployment.md

# Start with container setup
jupyter notebook tutorials/container_setup.ipynb
```

## The Challenge

The core of this bootcamp culminates in an exciting challenge: **Accelerating Drug Discovery with NVIDIA MolMIM**. You'll harness cutting-edge AI to revolutionize drug discovery by generating and optimizing novel molecular structures with potential as therapeutic agents.

### What You'll Do:
- Design custom scoring oracles combining multiple molecular properties
- Generate diverse molecular structures using MolMIM
- Optimize molecules for drug-like characteristics using CMA-ES
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
