# Diffusion-Based Active Perception for Source Seeking

A small research project exploring how generative world models can support active perception and optimization in partially observed environments.

## Motivation

Many robotic systems must make decisions with incomplete information. Rather than exhaustively exploring an environment, an agent should use prior knowledge to infer unobserved regions and guide future measurements.

This project investigates a simple version of that idea:

- Learn a prior over spatial scalar fields using a Denoising Diffusion Probabilistic Model (DDPM).
- Reconstruct an unknown environment from sparse local observations.
- Use the reconstructed world model to guide source-seeking behavior.

Although the current implementation focuses on synthetic Gaussian fields, the framework can be extended to more general spatial phenomena.

## Method

At each iteration:

1. The agent collects a local measurement using a limited sensor footprint.
2. Observed measurements are accumulated into a belief map.
3. A conditional DDPM reconstructs the unobserved portions of the field.
4. A simple multi-scale hill-climbing algorithm searches the reconstructed field for the most promising direction.
5. The agent moves and repeats the process.

              ┌───────────────────────────────┐
              │         Unknown Field         │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      Local Measurements       │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      Partial Belief Map       │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      Conditional DDPM         │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │   Reconstructed World Model   │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │   Multi-Scale Optimization    │
              └───────────────┬───────────────┘
                              ↓
              ┌───────────────────────────────┐
              │   Next Measurement Location   │
              └───────────────────────────────┘

## Current Setup

| Description    | Savings |
| -------- | ------- |
| Environment size  | 28 × 28    |
| Sensor footprint | 5 × 5     |
| Training data    | synthetic Gaussian scalar fields    |
| Reconstruction model | DDPM with U-Net backbone     |
| Planner    | multi-scale hill climbing    |

The project was intentionally designed to be lightweight and executable on modest computing resources.

## Example Research Questions
- Can generative models accelerate source-seeking under partial observability?
- How much prior knowledge is required for accurate field reconstruction?
- How should sensing actions be selected to maximize information gain?
- Can uncertainty estimates from diffusion models improve exploration strategies?

## Future Directions
- General scalar fields beyond Gaussian mixtures
- Information-theoretic action selection
- Uncertainty-aware planning
- Active mapping and environmental monitoring
- Multi-agent source seeking
- Integration with robotic platforms

## Disclaimer
During the preparation of this README, the author utilized ChatGPT to enhance language and readability and to aid in programming the simulation. After using this tool, the author has reviewed and edited the content as needed and take full responsibility for the content of the publication.