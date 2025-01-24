# Can We Generate Images 🌇 with CoT 🧠?

Official repository for the paper "[Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://github.com/ZiyuGuo99/Image-Generation-CoT/blob/main/Can%20We%20Generate%20Images%20with%20CoT%3F.pdf)".

[[📖 Paper](https://arxiv.org/pdf/2501.13926)] [[🤗 HF Checkpoints](https://huggingface.co/ZiyuG/Image-Generation-CoT)] [[🤗 HF Datasets (coming)]()]

## 💥 News
- **[2025.01.23]** We release the code for autoregressive image generation with test-time scaling (ORM, PARM) and DPO 🚀
- **[2025.01.23]** We release the [arXiv paper](https://arxiv.org/pdf/2501.13926) 🚀

## 👀 Reasoning in Image Generation

Chain-of-Thought (CoT) reasoning has been extensively explored by LLMs and LMMs in mathematics. However, it still remains an open question whether such strategies can be applied to **verifying and reinforcing image generation scenarios**. In this project, we provide ***the first*** comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation.

<p align="center">
    <img src="figs/fig1.jpg" width="90%"> <br>
</p>

We focus on three CoT reasoning techniques:
1. ***Scaling Test-time Computation*** for verification (ORM, PRM, and our proposed PARM and PARM++)
2. ***Aligning Model Preferences*** with Direct Preference Optimization (DPO)
3. ***Integrating These Techniques*** for complementary effects

Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve the image generation performance:

<p align="center">
    <img src="figs/fig2.jpg" width="100%"> <br>
</p>
  
Furthermore, given the pivotal role of reward models in our findings, we propose the ***P***otential ***A***ssessment ***R***eward ***M***odel (***PARM***) and ***PARM++***, specialized for autoregressive image generation:

1. ***PARM*** adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models.
2. ***PARM++*** further introduces a reflection mechanism to empower generative models to self-correct the previous unsatisfactory image.

<p align="center">
    <img src="figs/fig3.jpg" width="90%"> <br>
</p>

## 💪 Get Started

### 1. Scaling Test-time Computation 📈

#### 1.1. Zero-shot ORM
#### 1.2. Fine-tuned ORM
#### 1.3. PARM

### 2. Preference Alignment with DPO 🔧

#### 2.1. Initial DPO
#### 2.2. Iterative DPO
#### 2.3. Iterative DPO with PARM Guidance

### 3. Reasoning Strategy Integration 🧩

#### 3.1. Iterative DPO with PARM Guidance + PARM


## 🧠 Related Work

Explore our additional research on **CoT Reasoning** and **3D Vision**:

- **[MathVerse]** [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://mathverse-cuhk.github.io/)
- **[MAVIS]** [MAVIS: Mathematical Visual Instruction Tuning with an Automatic Data Engine](https://arxiv.org/pdf/2407.08739)
- **[SAM2Point]** [SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners](https://sam2point.github.io/)
- **[Point-Bind & Point-LLM]** [Multi-modality 3D Understanding, Generation, and Instruction Following](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)
- **[MMSearch]** [MMSearch: Unveiling the Potential of Large Models as Multi-modal Search Engines](https://mmsearch.github.io/)
