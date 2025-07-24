# NVIDIA BioNeMo Bootcamp Challenge

## Hackathon Challenge - Accelerating Drug Discovery with NVIDIA MolMIM

Welcome to the NVIDIA MolMIM NIM Hackathon Challenge! In this event, you'll harness the power of cutting-edge AI to revolutionize drug discovery. Your mission is to generate and optimize novel molecular structures, focusing on their potential as therapeutic agents. This challenge will push the boundaries of computational chemistry, combining advanced generative models with robust scoring tools to identify promising drug candidates.

## The Challenge

The core of this challenge lies in the intelligent design and evaluation of molecules. You will leverage the NVIDIA MolMIM NIM to generate a diverse array of molecular structures. Once generated, these molecules will undergo rigorous assessment of critical chemical properties such as:

* **Drug-likeness (QED)**: Assessing the overall suitability of a molecule as a drug candidate.
* **Synthesizability (Synthetic Accessibility)**: Determining the ease with which a molecule can be synthesized in a lab.
* **Solubility**: Predicting how well a molecule will dissolve in a solvent.
* **Toxicity**: Identifying potential harmful effects of a molecule.
* **Tanimoto Similarity**: Chemical similarity to known therapeutics
* **Other Chemical Properties**: Exploring additional relevant characteristics for drug development.

By combining these scoring techniques with MolMIM's latent-space representation of candidate molecules, you can leverage numerical optimization methods like [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) to steer the search of latent space towards an ideal set of chemical properties.

This is where your creativity comes into play! Which properties are most important? How should they be weighted? Which properties should be maximized or minimized (QED vs. toxicity)? Ultimately you will define some combination of scoring oracles that result in one floating point score for each generated molecule (or its hidden state vector representation), and use this score to guide the search in latent space. How you define this is entirely up to you!

The ultimate test of your generated molecules will be their predicted ability to bind to a specific protein target structure. The combination of high-scoring chemical properties and ability to bind to a target protein structure determine a candidate's viability as a potential therapeutic.

## Getting Started

1. Complete the tutorials in the `../tutorials/` folder to familiarize yourself with MolMIM
2. Set up your environment using the container setup guide
3. Explore the example notebooks to understand the workflow
4. Design your custom scoring oracle
5. Generate and optimize molecular candidates
6. Evaluate your results

## Challenge Objectives

Your challenge is to:

1. **Design a Custom Scoring Oracle**: Create a scoring function that combines multiple molecular properties
2. **Generate Novel Molecules**: Use MolMIM to generate diverse molecular structures
3. **Optimize Properties**: Use CMA-ES or other optimization methods to improve molecular properties
4. **Evaluate Drug Potential**: Assess the drug-like properties and binding potential of your generated molecules

## Why Participate?

This challenge offers a unique opportunity to:

* Gain hands-on experience with state-of-the-art AI models for molecular design
* Deepen your understanding of computational drug discovery workflows
* Collaborate with fellow innovators and experts in the field
* Contribute to the advancement of medical science by accelerating the identification of new drugs

Prepare to innovate, experiment, and push the boundaries of what's possible in the quest for life-changing medicines!

## Resources

- **Tutorials**: Complete guided examples in the tutorials folder
- **Documentation**: MolMIM NIM documentation and examples
- **Support**: Technical support and community forums

Good luck with your challenge! 