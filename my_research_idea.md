
# Evaluation method
# Reinforcement Learning
## Idea 1
## Idea 2
## Idea 2.1
## Idea 2.2
## Idea 2.3
### Key Idea
- Gộp $J_\pi$ và $L_Q$ làm một.
- $\psi = (\theta,\ \phi)$
- $\nabla_\psi J(\psi) = (\nabla_\theta J(\psi),\ \nabla_\phi J(\psi))$
### Main
- $L(\phi) = ( Q_\phi(s_{i}), a_{i}) - r - \gamma * Q_\phi(s_{i+1}, \pi_\theta(s_{i+1}))^2$
- $J(\theta) = Q_{\phi}(s, \pi_{\theta}(s))$
- $J(\psi) = \frac{\sum_{i} J_i(\theta)} {\beta * \sum_{j} L_j(\phi) + 1}$
- $\beta$ is a new hyperparameter.
- $\pi$ and $Q$ have their own minibatch. If PER is used, they also have separate Replay Buffers.
### Consider
1. $J(\psi) = \sum_{i} \frac{ J_i(\theta)} {\beta * L_i(\phi) + 1}$
### Pseudocode
```

```
### Future Improvement (or Issue)
- How to add the entropy idea of SAC.
- How to use target network...
    - for $\pi$ only.
    - for $Q$ only.
    - for both $\pi$ and $Q$.
## Idea 2.3.1
## Idea 3
## Idea 4
## Idea 5
## Idea 6
## Idea 7
## Idea 8
## Idea 9
## Experiment 1
## Experiment 2
# Combinatorial Optimization
# Multi-objective Optimization
# Multi-objective Reinforcement Learning
## Idea 2.3.1
