# BioNeMo Bootcamp

## What is NVIDIA BioNeMo?

NVIDIA BioNeMo is a groundbreaking AI platform designed specifically for computational biology and drug discovery. It provides state-of-the-art generative AI models and tools that accelerate the identification and optimization of therapeutic compounds. BioNeMo leverages advanced machine learning techniques, including transformer-based architectures, to understand and generate molecular structures with unprecedented accuracy and efficiency.

Key capabilities of BioNeMo include:
- **Molecular Generation**: Create novel molecular structures with desired properties
- **Property Prediction**: Predict chemical and biological properties of molecules
- **Molecular Optimization**: Optimize existing molecules for improved drug-like characteristics
- **Large-Scale Processing**: Handle massive molecular datasets efficiently

## Welcome to the BioNeMo Bootcamp with Challenge

This bootcamp combines comprehensive tutorials with an exciting hackathon challenge, giving you hands-on experience with NVIDIA's cutting-edge AI tools for drug discovery. You'll learn to harness the power of MolMIM (Molecular Masked Image Modeling) to generate and optimize novel molecular structures.

## Repository Structure

### 📚 Tutorials
The `tutorials/` folder contains everything you need to get started:
- **Lab 1**: Fundamental MolMIM operations (clustering, generation, interpolation)
- **Lab 2**: Advanced techniques with custom oracles, optimization, and guided molecule generation
- **Container Setup**: Step-by-step deployment guide (`container_setup.ipynb`)
- **Requirements**: All necessary dependencies for the environment (`requirements.txt`)

### 🏆 Challenge
The `challenge/` folder contains the hackathon challenge where you'll apply your knowledge to solve real drug discovery problems.

## Bootcamp Objectives

By the end of this workshop, participants will:

* Understand NVIDIA BioNeMo architecture and key functionalities
* Gain deep insights into MolMIM for generative small molecule design
* Develop hands-on skills to apply BioNeMo workflows using real-world, complex datasets
* Integrate advanced optimization methods (CMA-ES with custom oracles) to refine molecular designs
* Optimize and scale workflows leveraging multi-GPU and HPC resources (optional)

## Getting Started

1. **Setup Environment**: Follow the deployment guide in `deployment.md` or the detailed container setup in `tutorials/container_setup.ipynb`
2. **Install Dependencies**: Use `tutorials/requirements.txt` for package installation
3. **Complete Tutorials**: Work through Lab 1 and Lab 2 in the tutorials folder
4. **Take the Challenge**: Apply your skills in the challenge folder

## Quick Start

```bash
# Clone and navigate to the repository
cd Bootcamp-BioNemo

# Install dependencies
pip install -r tutorials/requirements.txt

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
