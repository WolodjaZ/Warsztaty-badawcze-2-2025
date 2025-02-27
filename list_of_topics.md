# List of Potential Topics:

## Easiest Topics

### Project 1: Training SSL Model and Evaluating its Performance Against Various Representations
This project involves training a masked self-supervised learning (SSL) model on a simple dataset and comparing its performance with traditional representation learning methods. SSL models learn useful representations from unlabeled data by predicting masked or hidden parts of the input. Students will evaluate how these self-supervised approaches compare to classical dimensionality reduction techniques and supervised learning. The project culminates in training multiple SSL architectures and comparing their relative strengths.

**Project Goals**
- Choose, understand and train a SSL model
- Evaluate the model's performance on downstream tasks
- Compare the results with other representation learning methods
- Investigate how different representation learning approaches impact performance on varied datasets

<span style="color:green">Basic Requirements</span>
- Choose and understand the fundamentals of masked SSL (e.g., BERT for text or MAE for images)
- Train a masked SSL model on the CIFAR100 dataset
- Evaluate the model's performance using linear probing and k-Nearest Neighbors (kNN) on CIFAR100 and CIFAR10 datasets
- Compare the results with traditional representation learning methods:
  - Principal Component Analysis (PCA)
  - t-distributed Stochastic Neighbor Embedding (t-SNE)
  - Uniform Manifold Approximation and Projection (UMAP)

<span style="color:purple">Intermediate Requirements</span>
- Train a masked SSL model on a more complex dataset such as ImageNet-1k
- Train a contrastive SSL model (e.g., SimCLR, MoCo) on the CIFAR100 and ImageNet-1k datasets
- Train a supervised classifier on CIFAR100 dataset as a baseline
- Compare results across all approaches on both CIFAR100 and CIFAR10 datasets

<span style="color:red">Advanced Requirements</span>
- Train a self-distillation SSL model (e.g., DINO, BYOL) on the CIFAR100 and ImageNet-1k datasets
- Compare all results comprehensively, analyzing strengths and weaknesses of each representation learning approach

**Computation Requirements:** *medium*

**References**
- [Library for Self-Supervised Learning](https://github.com/lightly-ai/lightly)
- [Datasets](https://pytorch.org/vision/main/datasets.html)

## Project 2: Analysis of Logit Lens on Visual Models
**Project Overview**
This project explores the logit lens technique for analyzing internal representations in visual neural networks. Logit lens is a method that projects intermediate layer representations to the output space, revealing how information flows through the network. Students will implement this analysis technique across different vision models to understand how feature representations evolve through model layers.

**Project Goals**
- Implement logit lens analysis for vision models
- Analyze representation patterns across network layers
- Compare effectiveness across different model architectures
- Evaluate alignment between intermediate and final representations

<span style="color:green">Basic Requirements</span>
- Understand logit lens fundamentals and theory
- Implement a complete analysis pipeline that:
  - Extracts intermediate layer representations from models
  - Applies the logit lens technique to each layer
  - Calculates distance metrics between lens representation and final layer output
- Select and prepare two different vision models of CLIP (e.g., ResNet, ViT)
- Test the implementation on ImageNet-1k dataset

<span style="color:purple">Intermediate Requirements</span>
- Conduct comprehensive analysis:
  - Compare representation quality across different model architectures
  - Analyze layer-by-layer progression patterns
  - Calculate and interpret alignment metrics
- Test the logit lens on a different SSL model (e.g., MAE, DINO)
- Document findings and critically evaluate: Is logit lens an effective tool for understanding visual models?

<span style="color:red">Advanced Requirements</span>
- Implement and evaluate newer logit lens variations (e.g., tuned logit lens)
  - Compare with the standard approach
  - Analyze improvements in representation interpretability
  - Document effectiveness across different model architectures and depths

**Computation Requirements:** *low*

**References**
- [Logit Lens](https://arxiv.org/abs/2104.14548)
- [Tuned Logit Lens](https://arxiv.org/abs/2303.08112)
- [Logit Lens for Images](https://medium.com/@adjileyeb/unlocking-visual-insights-applying-the-logit-lens-to-image-data-with-vision-transformers-b99cb70dd704)
- [Alignment Metrics](https://arxiv.org/abs/2405.07987)
- [Neural Representation Comparison](https://arxiv.org/abs/2106.07682)
- [ImageNet-1k Dataset](https://huggingface.co/datasets/timm/imagenet-1k-wds)

### Project 3: Sparse Autoencoder (SAE) as a Concept Bottleneck Model (CBM)
**Project Overview**
This project explores using Sparse Autoencoders (SAEs) as concept-based models for interpreting learned representations from pre-trained vision models. SAEs compress input data while enforcing sparsity constraints, potentially revealing interpretable concepts. Students will implement an SAE to analyze how these extracted concepts relate to model predictions and evaluate the effectiveness of using SAE as a concept bottleneck model (CBM).

**Project Goals**
- Train or utilize a Sparse Autoencoder (SAE) model on CLIP representations
- Implement a concept bottleneck model using the SAE's extracted concepts
- Train and evaluate models on concept detection tasks
- Compare concept-based approaches with standard end-to-end models

<span style="color:green">Basic Requirements</span>
- Understand Sparse Autoencoder fundamentals and current state-of-the-art implementations
- Find an implementation of visual model SAE (or train it) and test it on a imagenet-1k dataset
- Design an evaluation pipeline based on the CUB200 dataset (birds classification with attribute annotations)

<span style="color:purple">Intermediate Requirements</span>
- Train concept bottleneck model through SAE with different sparsity levels and hyperparameters
- Train and evaluate a classification head (without concept bottleneck)
- Train and evaluate a concept bottleneck model on CLIP
- Compare results and analyze effectiveness of the concept-based approach
- Test model robustness using the TravelingBirds dataset (out-of-distribution test)

<span style="color:red">Advanced Requirements</span>
- Extend evaluation to the CelebA dataset, demonstrating generalization to human-centric concepts
  - Choose which attributes to use as a target (e.g., age, gender, Attractiveness)
- Analyze which concepts are most predictive for different attributes
- What is the smallest set of concepts needed to achieve a certain accuracy?

**Computation Requirements:** *low/medium*

**References**
- [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
- [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
- [SAE CLIP Implementation](https://github.com/neuroexplicit-saar/discover-then-name)
- [Concept-Based Models](https://arxiv.org/pdf/2007.04612)
- [CUB200 Dataset](https://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [CelebA Dataset](https://pytorch.org/vision/0.17/generated/torchvision.datasets.CelebA.html)

### Project 4: Analyzing LLMs as Automatic Interpretability Tools
**Project Overview**
This project examines the effectiveness of Large Language Models (LLMs) as zero/few-shot learners from neural network representations. Can LLMs "read" what concepts are encoded in a representation without specialized training? Students will compare LLMs against purpose-built supervised predictors and analyze their relative performance on concept detection tasks.

**Project Goals**
- Prepare concept datasets that pair model representations (x) with human-interpretable concepts (y)
- Evaluate how well LLMs can predict concepts from representations without specialized training
- Train dedicated concept predictors for comparison
- Assess whether LLMs can serve as general-purpose interpretability tools

<span style="color:green">Basic Requirements</span>
- Prepare concept dataset:
  - Identify suitable datasets for text concept prediction
  - Generate representation-concept pairs (x,y) using a self-supervised learning model
- Set up an interpretation pipeline:
  - Implement LLM-based interpretation using careful prompt engineering
  - Design effective prompting strategies
  - Create evaluation metrics to assess concept prediction accuracy

<span style="color:purple">Intermediate Requirements</span>
- If a suitable dataset is not available, adapt a text classification dataset for this purpose
- Conduct comprehensive comparison experiments with one of the following:
  - Test multiple LLM sizes to analyze scaling properties
  - Experiment with various prompting strategies
- Document and analyze results systematically

<span style="color:red">Advanced Requirements</span>
- Train a specialized concept predictor on the concept dataset
- Compare performance with the LLM-based approach
- Analyze tradeoffs between specialized training and general LLM capabilities

**Computation Requirements:** *high*

**References**
- [LMs Explaining Neurons in LMs](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)
- [Open Source Automated Interpretability](https://blog.eleuther.ai/autointerp/)

## Not So Easy Topics

### Project 5: Finetuning BERT Model with Different CLS Token Objectives*
**Project Overview**
This project explores how BERT was originally trained with two objectives: masked language modeling (MLM) and next sentence prediction (NSP). The special CLS token serves as an aggregate representation of the input text. Students will fine-tune pre-trained BERT models with different CLS token objectives beyond the original NSP task and evaluate their performance on various downstream tasks.

**Project Goals**
- Implement different training objectives for the CLS token representation
- Compare performance across multiple downstream tasks
- Analyze the impact of different CLS training strategies on model capabilities

<span style="color:green">Basic Requirements</span>
- Prepare an experimental pipeline:
  - Select at least two datasets for evaluation tasks (e.g., sentiment analysis, text classification)
  - Implement appropriate evaluation metrics
  - Design a method for CLS token representation modification
- Choose a new CLS task and prepare an augmentation model, for example:
  - Sentiment classification
  - Text similarity prediction
- Implement necessary preprocessing pipeline
- Evaluate a baseline BERT model (with original CLS objective) fine-tuning classifier

<span style="color:purple">Intermediate Requirements</span>
- Prepare a training pipeline for CLS objective modification
- Train and evaluate models:
  - Fine-tune BERT with new CLS objectives
  - Analyze performance across multiple downstream tasks
- Compare results with the baseline BERT model
- Document which CLS objectives lead to better generalization

<span style="color:red">Advanced Requirements</span>
- Design a novel CLS objective and implement it
- Compare with baseline approaches
- Analyze what properties of CLS objectives lead to better downstream performance

**Computation Requirements:** *medium*

**References**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [BERT Finetuning Best Practices](https://arxiv.org/abs/1905.05583)
- [BERT pretraining with Hugging Face](https://github.com/stelladk/PretrainingBERT/blob/main/pretrain.py)
- [BERT pretraining Hugging Face Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForPreTraining)
- [BERT from scratch](https://github.com/ChanCheeKean/DataScience/blob/main/13%20-%20NLP/C04%20-%20BERT%20(Pytorch%20Scratch).ipynb)
- [Nearest-Neighbor Contrastive Learning](https://arxiv.org/abs/2104.14548)
- [Sentiment Classification](https://huggingface.co/blog/sentiment-analysis-python)

### Project 6: Concept-Based Sparse Representations for Information Retrieval (IR)
**Project Overview**
This project explores how Sparse Autoencoders (SAEs) can enhance information retrieval tasks compared to dense representations. Dense embeddings (like those from CLIP) encode information in a distributed way, while sparse representations may better isolate distinct concepts. Students will implement and evaluate concept-based sparse representations from CLIP using either text or image inputs, studying how SAE sparse representations can improve information retrieval performance.

**Project Goals**
- Train or utilize a Sparse Autoencoder (SAE) model on CLIP representations
- Compare retrieval performance between dense (original) and sparse representations
- Evaluate both accuracy and computational efficiency
- Assess the practical usefulness of sparse representations for IR tasks

<span style="color:green">Basic Requirements</span>
- Understand Sparse Autoencoder fundamentals and current state-of-the-art approaches
- Find implementation of CLIP SAE (or train it) and test it on the ImageNet-1k or CC3m dataset
- Implement basic retrieval functionality for both dense and sparse representations

<span style="color:purple">Intermediate Requirements</span>
- Design a comprehensive evaluation pipeline for IR tasks, including:
  - Selection of appropriate metrics (precision, recall, MRR, etc.)
  - At least two datasets for evaluation (text-to-image, image-to-text, or text-to-text)
- Evaluate both dense and sparse representations on the IR datasets
- Present a quantitative comparison of both approaches
- Analyze retrieval speed and memory efficiency tradeoffs

<span style="color:red">Advanced Requirements</span>
- Choose one:
  - Develop a hybrid IR pipeline combining dense and sparse representations
  - Train an SAE model with an alternative base representation model
- Provide comparative analysis of your chosen approach against baseline methods

**Computation Requirements:** *low/medium*

**References**
   - [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
   - [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
   - [Implementation of TopK SAE](https://github.com/bartbussmann/BatchTopK/tree/main)
   - [SAE CLIP Repo](https://github.com/neuroexplicit-saar/discover-then-name?tab=readme-ov-file)
   - [CC3m Dataset](https://huggingface.co/datasets/pixparse/cc3m-wds)
   - [IR benchmark paper](https://aclanthology.org/Q14-1006.pdf)
   - [IR text dataset (title and text)](https://huggingface.co/datasets/mteb/scifact)

### Project 7: Sparse Autoencoders (SAE) as an Explainability Method for Downstream Tasks
**Project Overview**
This project explores using Sparse Autoencoders (SAEs) for concept-based explainability in vision neural networks. Students will implement SAE to interpret learned concepts from pre-trained vision representations, adapting the TCAV (Testing with Concept Activation Vectors) framework to work with SAE-discovered concepts. The project analyzes how well these automatically discovered concepts explain model predictions.

**Project Goals**
- Train or utilize a Sparse Autoencoder (SAE) model on CLIP or similar representations
- Develop an explainability method based on SAE-discovered concepts and TCAV methodology
- Evaluate the explainability effectiveness across different datasets and tasks
- Compare SAE-based explanations with other concept-based explainability methods

<span style="color:green">Basic Requirements</span>
- Understand Sparse Autoencoder fundamentals and current state-of-the-art
- Find implementation of visual model SAE (or train it) and test it on a toy dataset
- Study TCAV implementation and understand its approach to concept-based explanations
- Design a new explainability method that combines TCAV principles with SAE-discovered concepts

<span style="color:purple">Intermediate Requirements</span>
- Prepare a comprehensive evaluation pipeline
- Train downstream models on two datasets: CelebA and ImageNet-1k
- Compare SAE-based explanations with standard TCAV explanations
- Analyze the relationship between model predictions and explanations
- Evaluate how SAE's explanations relate to ground truth attributes when available

<span style="color:red">Advanced Requirements</span>
- Prepare a training pipeline for downstream models that deliberately suppresses a chosen concept
- Train a modified downstream model on the CelebA dataset with the suppressed concept
- Evaluate both standard and concept-suppressed models, comparing their results
- Evaluate whether the concept-suppression was effective using your SAE-TCAV method
- Quantify explanations using metrics like robustness, stability, consistency, and understandability

**Computation Requirements:** *low/medium*

**References**
- [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
- [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
- [SAE CLIP Implementation](https://github.com/neuroexplicit-saar/discover-then-name)
- [TCAV](https://arxiv.org/abs/1711.11279)
- [CelebA Dataset](https://pytorch.org/vision/0.17/generated/torchvision.datasets.CelebA.html)

### Project 8: Comparing Representation Learning Models with LLMs for Jailbreak Detection
**Project Overview**
This project compares two approaches for detecting language model jailbreak attempts: 1) fine-tuned representation learning models (BERT-like) and 2) direct LLM detection (e.g., Llama Guard). Jailbreak detection identifies attempts to bypass safety guardrails in LLMs. Students will implement and evaluate both detection methods to determine their effectiveness under different conditions and analyze their relative strengths.

**Project Goals**
- Implement a representation learning model for jailbreak detection
- Evaluate Llama Guard or similar safety-focused LLM for jailbreak detection
- Compare detection effectiveness across different jailbreak scenarios
- Analyze tradeoffs between the two approaches (accuracy, speed, robustness)

<span style="color:green">Basic Requirements</span>
- Prepare datasets for training and evaluation (using available jailbreak datasets)
- Implement appropriate evaluation metrics (accuracy, precision, recall, F1)
- Develop a comprehensive evaluation pipeline
- Set up Llama Guard or similar safety LLM:
  - Deploy model
  - Integrate with the evaluation pipeline
  - Develop effective prompting strategy
- Evaluate Llama Guard's performance on jailbreak detection

<span style="color:purple">Intermediate Requirements</span>
- Fine-tune a BERT-like model for jailbreak detection
- Evaluate the fine-tuned model on test datasets
- Compare results with Llama Guard
- Analyze effectiveness across:
  - Different jailbreak categories (prompt injection, role-playing, etc.)
  - Various model sizes (for both Llama Guard and BERT-like models)

<span style="color:red">Advanced Requirements</span>
Choose one:
- Experiment with:
  - Advanced training strategies for representation models (contrastive learning, etc.)
  - Different prompting approaches for Llama Guard
- Implement LLM fine-tuning with a classification head:
  - Compare with baseline approaches
  - Analyze performance tradeoffs (accuracy vs. computational resources)

**Computation Requirements:** *low*

**References**
   - [Jailbreak Detection via Pretrained Embeddings](https://arxiv.org/pdf/2412.01547)
   - [Meta LLM Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
   - [BERT MLMs as Generative Classifiers](https://arxiv.org/abs/2502.03793)
   - [Turning Language Models into Classifiers](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier)
   - [Jailbreak Dataset](https://huggingface.co/datasets/jackhhao/jailbreak-classification)
   - [Jailbreak evaluation dataset](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)

## Hardest Topics

### Project 9: Analysis of Mixture of Vision Experts (MoVE) for Image Classification
**Project Overview**
This project explores implementing and analyzing Mixture of Vision Experts (MoVE) for image classification. MoVE is an architecture that intelligently combines multiple specialized vision models, similar to how Mixture of Experts (MoE) works in language models. Students will build a simplified MoVE architecture that uses a routing mechanism to combine multiple pre-trained vision models, then analyze how tasks are distributed among experts through systematic experiments.

**Project Goals**
- Implement simplified MoVE architecture for example:
  - Baseline: CLIP
  - Vision Expert models (at least two): DINO, MAE, SimCLR
- Train and evaluate model on image classification tasks
- Compare performance with single expert systems
- Analyze expert contributions through ablation studies

<span style="color:green">Basic Requirements</span>
- Understand Mixture of Experts fundamentals and current state-of-the-art
- Study MoVE implementation details
- Develop simplified MoVE model for image classification

<span style="color:purple">Intermediate Requirements</span>
- Train and evaluate model on imagenet-1k and place365 datasets
- Document results and provide insights on model performance

<span style="color:red">Advanced Requirements</span>
- Analyze:
   - Expert contribution patterns
   - Architecture hyperparameters
 - Exchange baseline model with vision expert model

**Computation Requirements:** *high*

**References**
- [Mixture of Experts Survey](https://arxiv.org/pdf/2407.06204)
- [Mixture of Vision Experts](https://arxiv.org/pdf/2404.13046)
- [ImageNet Dataset](https://huggingface.co/datasets/timm/imagenet-1k-wds)
- [Place365 Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Places365.html)

---

*\* to be verified as we need to make sure that the project is not too hard*