# List of Potential Topics:

## Project 1: Concept Based Sparse Representations for IR
The goal is to explore how Sparse Autoencoders (SAE) can improve information retrieval tasks compared to traditional dense representations. Students will first select either a text model (like BERT) or an image model (like CLIP) as their base representation. Then, they will implement a Sparse Autoencoder that learns concept-based sparse representations from this base model's output.
For evaluation, students will set up an information retrieval pipeline with standard benchmarks to compare how well both representations (original dense vs. new sparse) perform at retrieving relevant documents/images. This comparison should consider both retrieval accuracy and computational efficiency.

References:
   - [CLIP: Contrastive Language-Image Pre-training](https://arxiv.org/abs/2103.00020)
   - [Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1)
   - [Flickr30k similarity metrics](https://aclanthology.org/Q14-1006.pdf)

## Project 2: Analysis of Mixture of Vision Experts (MoVE) for Image Classification
This project focuses on implementing and analyzing the Mixture of Vision Experts (MoVE) approach to image classification. Students will first implement a simplified version of the MoVE architecture, which divides the task among different vision models. They will select appropriate datasets for benchmarking and train their implementation.
The key analysis will involve ablation studies - systematically removing or modifying different experts to understand their contributions and the importance of experts for specific tasks. Students will also experiment with architectural variations to see how changes affect performance.

References:
   - [Mixture of Experts](https://arxiv.org/pdf/2407.06204) 
   - [Mixture of Vision Experts](https://arxiv.org/pdf/2404.13046)
   - [ImageNet](https://huggingface.co/datasets/timm/imagenet-1k-wds)


## Project 3: Comparing Representation Learning Models with LLMs for Jailbreak Detection
This project explores different approaches to detecting jailbreak attempts in language models. Students will compare two main approaches: using traditional finetuned representation learning models versus using LLMs directly. First, they will develop a model using supervised finetuning for detecting jailbreak attempts. Then, they will implement detection using LLM-based approaches like Llama Guard.
The core of the project involves finding appropriate datasets for both training and evaluation. Students will experiment with different prompting strategies when using LLMs and different training approaches for the representation learning models. The goal is to determine which approach is more effective at detecting jailbreak attempts and under what circumstances.

References:
   - [Jailbreak Detection via Pretrained Embeddings](https://arxiv.org/pdf/2412.01547)
   - [Meta LLM Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
   - [Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers](https://arxiv.org/abs/2502.03793)

## Project 4: Finetuning BERT Models on Different CLS Tasks
This project investigates how different training objectives for the CLS token affect BERT's downstream performance. Students will experiment with various ways to train the CLS token augmenting on already created models, such as:
- Sentiment classification  
- CLIP-style text-image similarity
- Contrastive learning between sentences
- Custom tasks they design
The key is to create different versions of BERT, each with its CLS token trained on a different task, then compare how these variants perform on various downstream classification tasks.

References:
- [BERT](https://arxiv.org/abs/1810.04805)
- [Finetuning BERT](https://arxiv.org/abs/1905.05583)
- [Using Nearest-Neighbor to augumenting contrastive learning](https://arxiv.org/abs/2104.14548)

## Project 5: Analysis of Logit Lens on Visual Models
This project explores the logit lens technique - a method for analyzing internal representations in neural networks - specifically for visual models. Students will first understand and implement logit lens analysis for vision models. The goal is to examine how information flows through different layers of the model and what kind of features are represented at each stage.
Students will apply logit lens to various visual models and compare what they discover. They can propose improvements to the logit lens technique or use their findings to provide insights about how good is logit lens using for example alignment metrics.

References:
   - [Logit Lens](https://arxiv.org/abs/2104.14548)
   - [Tuned Logit Lens](https://arxiv.org/abs/2303.08112)
   - [Logit Lens for Image](https://medium.com/@adjileyeb/unlocking-visual-insights-applying-the-logit-lens-to-image-data-with-vision-transformers-b99cb70dd704)
   - [CKA](https://arxiv.org/abs/1905.00414)

## Project 6: Analysis of Representation Alignment Metrics
This project examines different ways to measure and compare neural network representations. First, students will explore various metrics for comparing representations between models. Next, they will apply these metrics to analyze how different SSL vision models learn similar or different representations for the same tasks.
Students are encoredge to follow the [Representation Hypothesis](https://arxiv.org/abs/2405.07987) work for the analysis.

References:
   - [Representation Hypothesis](https://arxiv.org/abs/2405.07987)
   - [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)
   - [CKA](https://arxiv.org/abs/1905.00414)


## Project 7: Reimplementing CLIP Interpretation Results
This project focuses on reproducing and extending the findings from the [Interpreting CLIP](https://arxiv.org/abs/2103.00020) paper. Students will first work to reimplement the key experiments from the paper that show how CLIP represents and processes visual and textual information. After successfully reproducing the original results, students will extend the work by applying similar interpretation techniques to other visual models.

References:
   - [CLIP](https://arxiv.org/abs/2103.00020)
   - [Interpreting CLIP](https://arxiv.org/abs/2103.00020)

## Project 8: Analysing LLMs Automatic Interpretability tools
This project challenges the assumption that Large Language Models are the best tools for automatic model interpretation. Students will:
- Create or find a dataset for text concept prediction
- Train specialized text encoders specifically for concept prediction
- Compare these specialized models against LLMs at interpreting neural networks
The key question is whether LLMs are trurly good at interpreting neuron activation in case of concept neurons. Students will experiment with different approaches to improve the interpretation pipeline, such as:terpretation pipeline, such as:
- Different training strategies for concept prediction
- Various prompting methods for LLMs
- Novel ways to combine specialized models with LLMs

References:
- [Language models can explain neurons](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)
- [Open Source Automated Interpretability](https://blog.eleuther.ai/autointerp/)
