# List of Potential Topics:

## Project 1: Concept-Based Sparse Representations for Information Retrieval (IR)
**Project Overview**
This project explores how Sparse Autoencoders (SAE) can enhance information retrieval tasks compared to dense representations. Students will implement and evaluate concept-based sparse representations using either text (BERT) or image (CLIP) models as a foundation. Students will study how SAE sparse representations can be used for information retrieval tasks and compare the results with dense representations.

**Project Goals**
- Compare retrieval performance between dense (original) and sparse representations
- Evaluate both accuracy and computational efficiency
- Understand and implement state-of-the-art Sparse Autoencoder techniques
- Evaluate usefulness of sparse representations for IR tasks

<span style="color:green">Basic Requirements</span>
- Understand Sparse Autoencoder fundamentals and current state-of-the-art approaches
- Find implementation of CLIP SAE (or train it) and test it on the toy dataset
- Design evaluation pipeline for IR tasks, including:
  - Selection of appropriate metrics
  - Dataset preparation
  - Code for benchmarking

<span style="color:purple">Intermediate Requirements</span>
- Execute comprehensive model evaluation on IR tasks
- Study and document in Jupyter Notebooks differences between dense and sparse representations
- Present quantitative comparison of both approaches

<span style="color:red">Advanced Requirements</span>
- Choose one:
  - Develop hybrid IR pipeline combining dense and sparse representations
  - Train SAE model with alternative base representation model
- Provide comparative analysis of chosen approach

**References**
   - [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
   - [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
   - [SAE CLIP Repo](https://github.com/neuroexplicit-saar/discover-then-name?tab=readme-ov-file)
   - [IR benchmark paper](https://aclanthology.org/Q14-1006.pdf)

## Project 2: SAE as Concept-Based Explainability Tool
**Project Overview**
This project explores using Sparse Autoencoders (SAE) for concept-based explainability in neural networks. Students will implement SAE to interpret learned concepts from pre-trained vision models and analyze how these concepts relate to model predictions.

**Project Goals**
- Implement SAE for concept-based model interpretation on vision models
- Finetune pre-trained model for specific tasks
- Compare original model predictions with SAE concept explanations  
- Evaluate explainability effectiveness across different datasets

<span style="color:green">Basic Requirements</span>
- Understand Sparse Autoencoder fundamentals and current state-of-the-art
- Find implementation of visual model SAE (or train it) and test it on the toy dataset
- Design evaluation pipeline:
   - Select at least two datasets
   - Implement metrics for explainability
   - Visualize results

<span style="color:purple">Intermediate Requirements</span>
- Compare SAE with another concept-based explainability method
- Analyze relationship between model predictions and SAE concepts
- Evaluate how SAE's explanations relate to ground truth

<span style="color:red">Advanced Requirements</span>
Choose one:
- Improve SAE architecture for better explainability
- Quantify explanations using metrics like:
   - Robustness
   - Stability  
   - Consistency
   - Fairness
   - Understandability

**References**
- [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
- [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
- [SAE CLIP Implementation](https://github.com/neuroexplicit-saar/discover-then-name)
- [Concept-Based Models](https://arxiv.org/abs/2303.15632)
- [Explainability Metrics](https://arxiv.org/abs/2303.15632)

## Project 3: Analysis of Mixture of Vision Experts (MoVE) for Image Classification
**Project Overview**
This project explores implementing and analyzing Mixture of Vision Experts (MoVE) for image classification. Students will build a simplified MoVE architecture that distributes tasks among multiple vision models and analyze expert contributions through systematic experiments.

**Project Goals**
- Implement simplified MoVE architecture
- Compare performance with single expert systems
- Analyze expert contributions through ablation studies

<span style="color:green">Basic Requirements</span>
- Understand Mixture of Experts fundamentals and current state-of-the-art
- Study MoVE implementation details
- Develop simplified MoVE model for image classification
- Select dataset for training and train the model

<span style="color:purple">Intermediate Requirements</span>
- Design evaluation pipeline:
   - Select appropriate datasets (at least two for evaluation)
   - Implement performance metrics
   - Compare with single expert baseline
- Analyze:
   - Expert contribution patterns
   - Performance impact
   - Architecture hyperparameters

<span style="color:red">Advanced Requirements</span>
- Choose one:
  - Test various vision experts on multiple datasets
  - Suggest architectural changes to improve performance
- Conduct comprehensive ablation studies and analyze results

**References**
- [Mixture of Experts Survey](https://arxiv.org/pdf/2407.06204)
- [Mixture of Vision Experts](https://arxiv.org/pdf/2404.13046)
- [ImageNet Dataset](https://huggingface.co/datasets/timm/imagenet-1k-wds)

## Project 4: Comparing Representation Learning Models with LLMs for Jailbreak Detection
**Project Overview**
This project compares two approaches for detecting language model jailbreak attempts: finetuned representation learning models and direct LLM detection (e.g., Llama Guard). Students will implement and evaluate both methods to determine their effectiveness under different conditions.

**Project Goals**
- Implement jailbreak detection using both approaches
- Compare detection effectiveness across different scenarios
- Analyze tradeoffs between approaches

<span style="color:green">Basic Requirements</span>
- Prepare dataset pipeline:
   - Collect training/evaluation data
   - Implement evaluation metrics
   - Design testing scenarios
- Set up Llama Guard:
   - Deploy model
   - Integrate with evaluation pipeline
- Develop representation learning training pipeline

<span style="color:purple">Intermediate Requirements</span>
- Train and evaluate both approaches:
   - Finetune representation model
   - Configure Llama Guard
   - Compare performance
- Analyze effectiveness across:
   - Different jailbreak categories
   - Various model sizes

<span style="color:red">Advanced Requirements</span>
Choose one:
- Experiment with:
   - Advanced training strategies for representation models
   - Different prompting approaches for Llama Guard
- Implement LLM finetuning with classification head:
   - Compare with baseline approaches
   - Analyze performance tradeoffs

**References**
   - [Jailbreak Detection via Pretrained Embeddings](https://arxiv.org/pdf/2412.01547)
   - [Meta LLM Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
   - [BERT MLMs as Generative Classifiers](https://arxiv.org/abs/2502.03793)
   - [Turning Language Models into Classifiers](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier)

## Project 5: Finetuning BERT Models with Different CLS Token Objectives
**Project Overview**
This project explores how training the CLS token with different objectives affects BERT's downstream performance. Students will augment existing BERT models by experimenting with various CLS token training approaches and evaluate their effectiveness across tasks.

**Project Goals**
- Implement different CLS token training objectives
- Compare performance across downstream tasks
- Analyze impact of training strategies

<span style="color:green">Basic Requirements</span>
- Prepare experimental pipeline:
   - Select datasets for multiple tasks
   - Implement evaluation metrics
   - Design CLS token substitution method
- Choose augmentation model for training CLS token for example:
   - Sentiment classification
   - Text similarity
- Implement training CLS token pipeline

<span style="color:purple">Intermediate Requirements</span>
- Train and evaluate models:
   - Finetune with new CLS objectives
   - Compare with baseline BERT
   - Analyze performance across tasks
- Document impact of different objectives

<span style="color:red">Advanced Requirements</span>
- Implement multiple training strategies with at least two new CLS objectives
- Compare with baseline approaches

**References**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [BERT Finetuning Best Practices](https://arxiv.org/abs/1905.05583)
- [Nearest-Neighbor Contrastive Learning](https://arxiv.org/abs/2104.14548)
- [Sentiment Classification](https://huggingface.co/blog/sentiment-analysis-python)

## Project 6: Analysis of Logit Lens on Visual Models
**Project Overview**
This project explores the logit lens technique for analyzing internal representations in visual neural networks. Students will implement and test logit lens analysis across different vision models to understand information flow and feature representation through model layers.

**Project Goals**
- Implement logit lens for vision models
- Analyze representation patterns across layers
- Compare effectiveness across model architectures
- Evaluate using alignment metrics

<span style="color:green">Basic Requirements</span>
- Understand logit lens fundamentals
- Implement analysis pipeline:
   - Extract intermediate layer representations
   - Apply logit lens technique
   - Calculate similarity metrics
- Select and prepare:
   - Three different vision models
   - Evaluation datasets

<span style="color:purple">Intermediate Requirements</span>
- Conduct comprehensive analysis:
   - Compare representations across models
   - Analyze layer-by-layer patterns
   - Calculate alignment metrics
- Document findings:
   - Representation evolution
   - Model differences
   - Technique effectiveness
   - Is logit lens good?

<span style="color:red">Advanced Requirements</span>
- Implement and evaluate newer logit lens versions (e.g., tuned logit lens)
   - Compare with standard approach
   - Analyze improvements
   - Document effectiveness across models

**References**
- [Logit Lens](https://arxiv.org/abs/2104.14548)
- [Tuned Logit Lens](https://arxiv.org/abs/2303.08112)
- [Logit Lens for Images](https://medium.com/@adjileyeb/unlocking-visual-insights-applying-the-logit-lens-to-image-data-with-vision-transformers-b99cb70dd704)
- [Alignment Metrics](https://arxiv.org/abs/2405.07987)
- [Neural Representation Comparison](https://arxiv.org/abs/2106.07682)


## Project 7: Reimplementing CLIP Interpretation Results
**Project Overview**
This project focuses on reproducing and extending the analysis techniques from "Interpreting CLIP" paper. Students will reimplement key experiments examining CLIP, then apply these techniques to other visual models.

**Project Goals**
- Reproduce core CLIP interpretation experiments
- Validate original paper findings
- Extend analysis to other visual models
- Document and compare results

<span style="color:green">Basic Requirements</span>
- Prepare data for experiments
- Set up experimental environment:
   - Implement utility functions
   - Prepare data processing pipeline
   - Configure model access

<span style="color:purple">Intermediate Requirements</span>
- Reproduce paper results:
   - Implement key experiments
   - Document findings in Jupyter notebooks
   - Compare with original results
- Prepare framework for other visual models:
   - Adapt interpretation methods
   - Design comparison metrics
   - Plan extension experiments

<span style="color:red">Advanced Requirements</span>
- Apply analysis to new visual model:
   - Execute interpretation experiments
   - Compare with CLIP results
   - Document findings

**References**
   - [CLIP](https://arxiv.org/abs/2103.00020)
   - [Interpreting CLIP](https://arxiv.org/abs/2310.13040)

## Project 8: Analyzing LLMs as Automatic Interpretability Tools
**Project Overview**
This project examines the effectiveness of Large Language Models (LLMs) for automatic model interpretation, particularly for concept neurons. Students will compare LLMs against specialized text encoders in interpreting neural networks and analyze their relative performance.

**Project Goals**
- Evaluate LLMs for automatic interpretation
- Train concept bottleneck models
- Analyze concept neuron detection accuracy against specialized models

<span style="color:green">Basic Requirements</span>
- Set up interpretation pipeline:
   - Implement LLM-based interpretation
   - Design prompting strategies
   - Create evaluation metrics
- Prepare concept dataset:
   - Identify datasets for text concept prediction

<span style="color:purple">Intermediate Requirements</span>
- Conduct comparison experiments:
   - Test different LLM sizes
   - Vary prompting strategies
   - Document results
- Generate synthetic data (if needed):
   - Use LLMs for data creation
   - Validate data quality
   - Ensure concept coverage

<span style="color:red">Advanced Requirements</span>
- Implement concept bottleneck model:
- Train specialized concept predictor
- Compare with LLM approach

**References**
- [LMs Explaining Neurons in LMs](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)
- [Open Source Automated Interpretability](https://blog.eleuther.ai/autointerp/)
- [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612)