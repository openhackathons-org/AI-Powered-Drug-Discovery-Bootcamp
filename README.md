# BioNeMo Bootcamp

## Guided Small Molecule Generation with MolMIM and Custom Scoring Oracles

### Bootcamp Objectives

By the end of this workshop, participants will:

* Understand NVIDIA BioNeMo architecture and key functionalities.

* Gain deep insights into MolMIM for generative small molecule design.

* Develop hands-on skills to apply BioNeMo workflows using real-world, complex datasets.

* Integrate advanced optimization methods (CMA-ES with custom oracles) to refine molecular designs.

* Optimize and scale workflows leveraging multi-GPU and HPC resources.  


## Hackathon Challenge \- Accelerating Drug Discovery with NVIDIA MolMIM

Welcome to the NVIDIA MolMIM NIM Hackathon Challenge\! In this event, you'll harness the power of cutting-edge AI to revolutionize drug discovery. Your mission is to generate and optimize novel molecular structures, focusing on their potential as therapeutic agents. This challenge will push the boundaries of computational chemistry, combining advanced generative models with robust scoring tools to identify promising drug candidates.

### The Challenge

The core of this hackathon lies in the intelligent design and evaluation of molecules. You will leverage the NVIDIA MolMIM NIM to generate a diverse array of molecular structures. Once generated, these molecules will undergo rigorous assessment of critical chemical properties such as:

* **Drug-likeness (QED)**: Assessing the overall suitability of a molecule as a drug candidate.

* **Synthesizability (Synthetic Accessibility)**: Determining the ease with which a molecule can be synthesized in a lab.

* **Solubility**: Predicting how well a molecule will dissolve in a solvent.

* **Toxicity**: Identifying potential harmful effects of a molecule.

* **Tanimoto Similarity**: Chemical similarity to known therapeutics

* **Other Chemical Properties**: Exploring additional relevant characteristics for drug development.

By combining these scoring techniques with MolMIM's latent-space representation of candidate molecules, you can leverage numerical optimization methods like [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) to steer the search of latent space towards an ideal set of chemical properties.

This is where your creativity comes into play\!  Which properties are most important? How should they be weighted?  Which properties should be maximized or minimized (QED vs. toxicity)?  Ultimately you will define some combination of scoring oracles that result in one floating point score for each generated molecule (or its hidden state vector representation), and use this score to guide the search in latent space.  How you define this is entirely up to you\!

The ultimate test of your generated molecules will be their predicted ability to bind to a specific protein target structure. The combination of high-scoring chemical properties and ability to bind to a target protein structure determine a candidate's viability as a potential therapeutic.

### Why Participate?

This hackathon offers a unique opportunity to:

* Gain hands-on experience with state-of-the-art AI models for molecular design.

* Deepen your understanding of computational drug discovery workflows.

* Collaborate with fellow innovators and experts in the field.

* Contribute to the advancement of medical science by accelerating the identification of new drugs.

Prepare to innovate, experiment, and push the boundaries of what's possible in the quest for life-changing medicines.