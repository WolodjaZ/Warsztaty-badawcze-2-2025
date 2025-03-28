# Warsztaty-badawcze-2-2025

## 📚 Course Overview
This repository serves as the central hub for groups 2 and 3 from the `Laboratorium` and `Projekt` courses of `Warsztaty-badawcze-2` (Summer Semester 2025). All course materials, assignments, and resources will be maintained here.

## 👥 Teaching Team
- **Vladimir Zaigrajew** (Lead Instructor)
  - Email: vladimir.zaigrajew.dokt@pw.edu.pl
  - Location: Room 316, MINI, PW

- **Tymoteusz Kwieciński** (Co-Instructor)
  - Email: tymoteuszkwiecinski@gmail.com
  - Location: Room 316, MINI, PW

## 🔗 Essential Resources
- [Main Course Repository](https://github.com/mini-pw/2025-warsztaty-badawcze)
- [USOS Course Page](https://usosweb.usos.pw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1120-DS000-ISP-0363)
- [Reference Course Material](https://github.com/HHU-MMBS/RepresentationLearning_SS2023?tab=readme-ov-file)

*Note: The reference course provides extensive material on representation learning and is highly recommended for deeper understanding.*

## 📊 Grading System
The laboratory/project portion accounts for 60% of the final course grade, broken down as follows:

- Paper Workshop Summary: 0-10 points
- Project Work: 0-35 points
- Activity Points: 0-5 points (and optional tasks after each lecture)

Final Grade Scale:
| Points | Grade |
|--------|--------|
| 26-30 | 3.0 |
| 31-35 | 3.5 |
| 36-40 | 4.0 |
| 41-45 | 4.5 |
| 46-50 | 5.0 | 

**if Workshop is excluded:**
| Points | Grade |
|--------|--------|
| 21-24 | 3.0 |
| 25-28 | 3.5 |
| 29-32 | 4.0 |
| 33-36 | 4.5 |
| 37-40 | 5.0 |

## 📅 Course Structure
The plan is to have 6 hours of lectures (I will use both laboratory and project hours) and at 6th hour we will finish discussing the project topic and formulating the groups. Next two hours of laboratory will be dedicated to paper writing workshop and the rest of the time will be for the project work and report. The workshop part can be excluded if majority of students will not be interested in it, but remember the grade will be based only on the project.
The course will be divided into three main parts:

## Part 1: Theoretical Foundations
Here we will cover the basics of representation learning with topics such as self-supervised learning, contrastive learning, masked-based representation and interpretable representation learning. I will focus on foundations and various state-of-the-art methods in the field.

## Week 1: Introduction to Representation Learning
- Group 2: Wednesday, 2025-03-05
- Group 3: Tuesday, 2025-03-04
- [PDF](lectures/Introduction.pdf) || [Markdown](lectures/Introduction.qmd)
- Get to know the course
- Talk about the projects
- Representation learning - a solution for human ML labour
- I understand that some of you are not familiar with Deep Learning, **so please just ask me** if you have any questions. I don't want people to feel lost during the course and be afraid of the projects.
- References:
  - [Y Bengio; Representation learning: A review and new perspectives; 2013](https://ieeexplore.ieee.org/abstract/document/6472238)

**Exercise**:
  - Check [google colab](https://colab.google) here is [tutorial](https://www.geeksforgeeks.org/how-to-use-google-colab/), just try to compile it, etc. Remember you don't need to provide any payment information for a free tier.
  - For people who are not familiar with Deep Learning, I recommend to watch [this](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). For my course if you will understand it, the rest will be easy.
  - For people who are familiar with Deep Learning, I recommend watching lectures from my AI celebrity crush Andrey Karpathy. He served as Director of AI at Tesla, was a researcher at OpenAI, and completed his PhD under Fei-Fei Li at Stanford University, where he created CS231n, a brilliant deep learning for computer vision course that I highly recommend. Here is the link to his [YouTube channel](https://www.youtube.com/@AndrejKarpathy).

## Week 2: Self-supervised Learning
- Group 2: Wednesday, 2025-03-12
- Group 3: Tuesday, 2025-03-11
- [PDF](lectures/Self_supervised_Learning.pdf) || [Markdown](lectures/Self_supervised_Learning.qmd)
- Unlabeled data -> Creating Y from X using pretext task -> Training a model on Y -> Transfer Learning or Downstream task finetuning
- Pretext tasks covered: Colorization, Jigsaw puzzles, Classify corrupted images, Rotation Prediction
- The most popular pretext tasks from 2020:
  - Autoencoders: Creating bottleneck representation of the data from which we can recover the original data (VAE, SAE)
  - Contrastive Learning: Learning representations by contrasting positive and negative pairs (SimCLR, MoCo)
  - Masked-based Representation Learning: Learning representations by masking parts of the data and predicting them (BERT, MAE)
  - Distillation: Learning representations by training a student model to mimic the behavior of a teacher model (DINO, DINOv2)
- Most students know how to create reproducible code with seeding and use the GPU in Google Colab.

**Exercise**:
Let's get familiar with the tools we will be using for the project: [PyTorch](https://pytorch.org/). Pytorch is a free and open-source machine learning library which is now renowned as the best library for Deep Learning so we also will use it. I recommend analyzing the tutorial from the [official website](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) but it is not mandatory. What I want is to take the code from this [tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) and reproduce it on your own in Google Colab. I highly recommend commenting or explaining in markdown each section of the code so you will understand it better, and if you are not sure about something or you don't understand it, I will write a comment on it and explain it to you. The completed and compiled code (notebook) you will send me a link to it on Slack. If everything will be ok, I will give you **1** point for it.


## Week 3: Contrastive Learning and Masked-based Representation Learning
- Group 2: Wednesday, 2025-03-19
- Group 3: Tuesday, 2025-03-18
- [PDF](lecutes/Contrastive_and_mask.pdf) || [Markdown](lectures/Contrastive_and_mask.qmd)
- Inspired by Lab of HHU Dusseldorf [week 4](https://github.com/HHU-MMBS/RepresentationLearning_SS2023/tree/main?tab=readme-ov-file#week-4---contrastive-learning-simclr-and-mutual-information-based-proof) and [week 8](https://github.com/HHU-MMBS/RepresentationLearning_SS2023/tree/main?tab=readme-ov-file#week-8---masked-based-visual-representation-learning-mae-beit-ibot-dinov2)

**Exercise**:

Let's train our first SSL model and compare it against the supervised one. In the provided [notebook](exercise/week_3.ipynb), you will find code for the task. Your goal is to fill in the `CODE HERE` sections and run the code. The task is to train a model on the rotation task in an SSL way and compare it with the supervised model. In this task, we will use the `GTSRB` dataset which already comes with labels, but we will split the dataset and assume that one part doesn't have labels. You will train the model on the unlabeled part with a rotation task and then compare how it performs on the labeled part. You will compare the results with: the supervised model trained on the labeled part only, the SSL model as a feature extractor for a classifier trained on the labeled part, and the SSL model fully fine-tuned on the labeled part. At the end, I expect you to answer the questions in the notebook and do the task it asks you to do. In the previous task, you used Google Colab, so let's use it again with GPU acceleration (don't waste time on CPU). After you finish the task, please send me a link to your notebook on Slack. If everything is satisfactory, I will give you **1** point for it.

<span style="color:red">Additional Exercise </span>

In Week 2's lecture, I introduced you to DINO and DINOv2 SSL methods but didn't explain them in detail (as they're complex and would require more than just 10 minutes). Your task, worth **2** points, is to read the paper [DINO](https://arxiv.org/abs/2104.14294) and explain to me via Slack private chat how it works. You don't need to read the entire paper thoroughly - I just want to verify that you fully understand how the model was trained (pretext task, downstream task, etc.). 

If your explanation is comprehensive, you'll earn the full **2** points. If your explanation lacks certain elements, I'll ask follow-up questions which you should try to answer. In this exercise, I'll take the role of the student and you'll be the teacher. If you teach me how DINO works I will give you **2** points, but if you fail to explain it, I won't give you any points (Don't worry, I'll try to help you get the points).

## Week 4: Exploring Self-supervised Learned Representations 
- Group 2: Wednesday, 2025-03-26
- Group 3: Tuesday, 2025-04-25
- [PDF](lectures/Exploring_Self-supervised_Learned_Representations.pdf) || [Markdown](lectures/Exploring_Self-supervised_Learned_Representations.qmd)
- Limitations of existing vision language models
- Self-supervised VS supervised learned feature representations
- What do vision transformers (ViTs) learn “on their own”?
- Interesting papers discussed on the lecture:
  - [How well do self-supervised models transfer?](https://openaccess.thecvf.com/content/CVPR2021/papers/Ericsson_How_Well_Do_Self-Supervised_Models_Transfer_CVPR_2021_paper.pdf)
  - [An Empirical Study of Training Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
  - [Emerging Properties in Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
  - [What Do Self-Supervised Vision Transformers Learn?](https://arxiv.org/abs/2305.00729)

**Exercise**:

Now as I know that you are running out of the GPU from Colab, this task is easier for you as for the part where you need to train the model, I provide the weights for you. So what is this task about? Previously, you learned and checked how SSL models compare to supervised models. Now you will compare them even more. Using the provided [notebook](exercise/week_4.ipynb), this time you will compare 4 different models: one trained from scratch in a supervised way, second in SSL way, third downloaded from PyTorch pretrained library, and finally, going to the basics, you will have to train PCA.

In the previous task, you learned about linear probing (training only the classifier head). This time you will see two more evaluation methods on representations: visualization and an alternative to linear probing - KNN classification. In the previous exercise, you did everything on one dataset, but when we train representation learning models, we want them to be generalizable and not only be able to solve the task on particular data but be able to easily adapt to new data domains.

So you will get 4 models, and you will evaluate them as feature extractors on 3 different datasets. Your main part of this task is to write a short report (really short one) at the end of the notebook. You have questions that I want you to talk about in the report. The report should be about what you learned from this task basically. I don't require using Google Colab anymore, so as a result, you can just send me the notebook (but now in PDF format). After I validate your report, I will give you **1** point for it.

## Week 5: Interpretable Representation Learning 
- Group 2: Wednesday, 2025-04-02
- Group 3: Tuesday, 2025-04-01
- [PDF]() || [Markdown]()
- TODO  
- References:
  - TODO

**Exercise**:

This is our last exercise before the project. This task aims to combine everything that we've done earlier and add the training and evaluation of [SimCLR model](https://arxiv.org/abs/2002.05709). This task is a combination of exercises from weeks 4 and 5, so some of you may be familiar with the code and the tasks. Basically, in the [notebook](exercise/week_5.ipynb), we have the code to train 2 SSL models (SimCLR and Rotation) and evaluate them on 3 datasets, one on which we will also train it. The task is to fill in the code where required but also to train a model on the rotation task in an SSL way and compare it with the supervised model. In this task, we will use the `GTSRB` dataset which already comes with labels, but we will split the dataset and assume that one part doesn't have labels. You will train the model on the unlabeled part with a rotation task and then compare how it performs on the labeled part. You will compare the results with: the supervised model trained on the labeled part only, the SSL model as a feature extractor for a classifier trained on the labeled part, and the SSL model fully fine-tuned on the labeled part. At the end, I expect you to answer the questions in the notebook in short summary report and do the task it asks you to do. If everything is satisfactory, I will give you **2** points for it.

## Week 6: Group Formation & Project Planning
- Group 2: Wednesday, 2025-04-02
- Group 3: Tuesday, 2025-04-01
- Group formation (2-4 members)
- Project topic selection and discussion
- Project expectations overview

## List of Topics is provided in the [`list_of_topics.md`](list_of_topics.md) file.

## Part 2: Paper Writing Workshop
- Group 2: Wednesday, 2025-04-09 and 2025-04-16
- Group 3: Tuesday, 2025-04-08 and 2025-04-15

In this part, we will focus on the paper writing journey. 
I will show you how the process of writing a paper looks like and what are the most important things to remember. 
I will also provide you with some tips and tricks on how to write a good paper. After this lecture, you will have to choose a paper and write a short summary of it with paraphrasing.

**Deadline**: Group 2 2025-04-30; Group 3 2025-04-29

**Paper Summary Requirements** (Due 2 weeks after this Writing lecture)
- Section-by-section summary (3 points)
- Summary of the review process of the paper (1 point)
- Clear contribution statement and what results prove the contribution (1 point)
- Author background research (who they are, from where, number of citations and main domain of research of each author) (1 point)
- Analysis of papers that have cited the summary paper with focus on why they cited this paper (min. 3 papers, 2 points)
- Overall quality assessment (2 points)

**List of Papers**:
TODO

### Part 3: Project Work
Projects should:
- Focus on representation learning
- Be completed in groups of 2-4
- Use Python as the primary programming language
- Be submitted as a GitHub repository
- Include:
  - Source code or Jupyter notebooks
  - Final presentation
  - Live Demo (bonus points)

The Grading System for the project is as follows:
- <span style="color:green">Basic level done </span>: 0-5 points
- <span style="color:purple">Intermediate level done  </span>: 0-10 points
- <span style="color:red">Advanced level done  </span>: 0-5 points
- Presentation: 0-9 points
  - Structure: 0-2 points
  - Content: 0-2 points
  - Clarity: 0-2 points
  - Visuals: 0-2 points
  - Interaction with audience: 0-1 points
- Code Quality: 0-4 points
  - Readability: 0-2 points
  - Reproducibility: 0-1 points
  - Documentation: 0-1 points
- Live Demo: 0-2 points

Recommended Demo Tools:
- [Gradio](https://github.com/gradio-app/gradio)
- [Mesop](https://github.com/google/mesop)
- [Taipy](https://github.com/Avaiga/taipy)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Dash](https://github.com/plotly/dash)

**Deadlines**: 
Presentation:
- Lecture 2025-06-03
- Group 2 2025-06-04 and 2025-06-11
- Group 3 2025-06-03 and 2025-06-10

Code and Report
- Group 2 2025-06-04
- Group 3 2025-06-03

---
*Lecture materials, paper list, and specific deadlines will be updated as the course progresses.
AI tools may be used for assistance, but understanding of content will be verified.*