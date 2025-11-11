# Virtual Analogue Modelling of a Compressor

## Most recent papers
### 2025
1. Riccardo Simionato – [Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling](https://arxiv.org/abs/2405.04124)
2. [Unsupervised Estimation of Nonlinear Audio Effects: Comparing Diffusion-Based and Adversarial approaches](https://arxiv.org/abs/2504.04751)
3. Riccardo Simionato – [MODELING TIME-VARIANT RESPONSES OF OPTICALCOMPRESSORS WITH SELECTIVE STATE SPACE MODELS](https://arxiv.org/pdf/2408.12549)
4. [ANTIALIASED BLACK-BOX MODELING OF AUDIO DISTORTION CIRCUITS USING REAL LINEAR RECURRENT UNITS](https://dafx.de/paper-archive/2025/DAFx25_paper_61.pdf)
5. [Diff-SSL-G-COMP: Towards a large-scale and diverse dataset for virtual analog modeling](https://ar5iv.org/html/2504.04589v1)

## Deliverables

1. Gather a subset of the dataset of Diff-SSL for experimenting
2. Extract gain reduction
3. Test models/do small trainings
    - build new
        - predict gain reduction and condition another network with that
    - Used models:
        - TCN
        - GCN
        - LSTM
        - ED
        - S6 ??
    - conditioning: FILM
5. Optimize models
6. Improve Loss function
    - RDC
    - FRAC
    - Spectral Flatness
7. Evaluate on bigger datasets and compare with other models
