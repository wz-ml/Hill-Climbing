---
layout: post
title:  "Robust Inverse RL, Explained"
date:   2024-05-10 12:27:49 -0700
categories: post
---
### Tl;dr 
RAMBO-RL is an offline RL algorithm that is based on three primary ideas:
1. There's a small random shift between the experience samples we have and the underlying distribution, so we'd like to train our offline RL models to have good *worst-case* performance instead of good *average-case* performance.
2. Adding an adversary makes the model's worst-case performance robust.
3. We can embed this adversary into the state transition model.

#### Got more time? Read on.

## RARL

RARL (Robust Adversarial Reinforcement Learning) was the precursor paper to RAMBO-RL, and provides a lot of the background intuition that RAMBO-RL builds upon. 

In offline RL, we experience distributional shifts between the training samples we have available and the inference environment. This is due to a variety of reasons:
1. Our experience sample is small, or gathered using a policy that isn't representative of the optimal policy we can learn using the experience sample.
2. We're training an RL agent in silica for deployment in the real world.

In either case, it's not uncommon for there to be environmental differences between training and testing environments - for example, a CartPole agent could be placed in an environment where the weight, angular momentum, or drag of the bar differs from the training environment. We can model these unexpected environmental differences as additional forces that are added on to the environmentally-generated forces during inference time. 

Thus, the intuition goes that we can train a model to be robust to differences in environment by adding an adversary that can perturb the model by applying an additional force at every timestep.

>The results of the paper show that RARL algorithms outperform TRPO. What's more interesting is the fact that the *training distributions become narrower* (performance is more uniformly good over multiple training runs). This is totally expected, and I'll explain why. One of the most important formulations that RARL contributes to RAMBO-RL is the idea of **Conditional Value at Risk (CVaR)**, which is as follows:

>$$\rho_{RC} = \mathbb{E}[\rho|\rho \leq Q_\alpha(\rho)]$$

>Where $\rho$ is the expected reward, and $Q_\alpha(\rho)$ is the $\alpha$-percentile of $\rho$-values. Intuitively, what this is saying is that the Conditional Value at Risk is the average reward over the $\alpha$-% worst possible outcomes. If $\alpha = 5$, CVaR = the mean return over the 5% worst cases.

The adversary forces the model to optimize its CVaR by exposing it's worst-case performance. The paper goes on to establish a link between $\alpha$ and the *strength* of the adversary - how much force it's allowed to apply. Nonetheless, it's unsurprising that the adversarially-trained model has good worst-case training performance as well as being able to generalize across a wide range of environmental hyperparameters.

### Concept: State Vistation
Before we jump into the mathematics of RAMBO-RL, let's take the time to examine a few novel terms (by novel, I mean I wasn't sure what they referred to before I began reading this paper. I hope this segment is interesting to you as well. :P )

#### Terms
- $M$ is our environmental MDP policy, defined by the tuple $M = (S, A, T, R, \mu_0, \gamma)$. 

- $T(s' \vert s, a)$ is the transition function, while $\mu_0$ is the initial state distribution.

- $D$ is the training set of transitions we have. When we say that $(s, a)$ is sampled from the training set, we can write $(s, a) \sim D$.

- We also define $\Pi$ as the space of all Markovian policies over our environment, such that any policy that maps from a state to a distribution over actions $\pi$ has the property $\pi \in \Pi$.

The **improper discounted state visitation distribution** of a policy can be modelled as follows:

$$d^{\pi}_M(s) := \sum_{t=0}^{\infty}\ \gamma^t Pr(s_t = s|\pi, M)$$

The discounted state visitation distribution of a policy is a measure of how frequently each state is visited by an agent following policy $\pi$, taking into account a discount factor $\gamma$. This distribution is important in understanding the behavior of an agent under a given policy.

The discounted state visitation distribution is influenced by both the policy and the environment’s dynamics. For a given policy, states that are visited more frequently or earlier (i.e. in fewer steps from the initial state) will have a higher value in the distribution.

To build intuition, let's imagine that we're in an environment with only one state, S, which we cannot transition out of. The **improper discounted state visitation distribution**, $d^{\pi}_M(s)$, would have the property of $d^{\pi}_M(s=S) = 1 + \gamma + \gamma^2 + \gamma^3... + \gamma^\infty$. 

I hope this also shows why we call this the *improper* discounted state visitation distribution - it doesn't even sum up to one! Note that in the previous case, the sum of the distribution is equal to the taylor series expansion of $\frac{1}{1-\gamma}$.

To normalize the sum of the discounted state visitation distribution to 1, we multiply the improper distribution by $1-\gamma$.

The **discounted state visitation distribution**:

$$(1-\gamma)\cdot d^{\pi}_M(s)$$

Likewise, we can formalize the **improper discounted state-action visitation distribution**, the gamma-weighted frequency of a given state-action pair to be executed under a given policy and environment, to be:

$$d^{\pi}_M(s, a) = \pi(a|s)\cdot d^{\pi}_M(s)$$

And the **discounted state-action visitation distribution** is as follows:

$$d^{\pi}_M(s, a) = (1-\gamma)\cdot d^{\pi}_M(s, a)$$

## (Model) Based
RAMBO-RL is a model-based approach, which entails one (or two) things:
1. We learn a *dynamics model*, $\hat{T}(s' \vert s,a)$, that tries to approximate the transition function, $T(s' \vert s,a)$. It does so to the best of its ability given our data using *maximum likelihood estimation (MLE)*, where we use gradient descent to minimize the *MLE loss*:

$$\mathbb{E}_{(s,a,s') \sim D}\left[-\log \hat{T}(s' \vert s,a)\right]$$

Intuitively, we can think of minimizing the negative logprob as maximizing the positive pre-softmax logits of the model that predict the same state transitions as we've seen in the training data.

2. A model of the reward function that we minimize with MSE loss, which we'll coin $\hat{R}(s, a)$. This is only necessary if the ground truth reward function isn't available to us, and shouldn't be a problem in many offline RL scenarios.

Given our learned transition function $\hat{T}(s' \vert s,a)$, let us construct a new Markov Decision Process (MDP) called $\hat{M}$. We can extract the optimal policy in the learnt model using any other RL algorithm: 

$$\hat{\pi} = \text{arg max}_{\pi \in \Pi} V^{\pi}_{\hat{M}}$$

Where $V^{\pi}_{\hat{M}}$ denotes the expected return of our policy under our approximated environment.

<details><summary>Sidenote: What are some problems that come with this approach?</summary>
<br/>
A historical problem that has plagued model-based offline RL is model exploitation. Naively, we're constructing our new environment, represented by the MDP $\hat{M}$, to start from the same state as the original environment does and then be completely guided by our dynamics model. It's not hard to imagine a model learning to gain insane reward by throwing the dynamics model off-kilter by seeking some super weird state not in the training data.

How do we tackle this problem? Well, what we can do is keep the environment bounded by making sure that the dynamics model is never in control for more than a few timesteps. 

We can sample a transition from our dataset:

$$(s,a,s') \sim D$$

And for a few timesteps:

$$s_{t+1} = \hat{T}(s_{t}, \pi(s_{t}))$$

$r_{t+1} = \hat{R}(s_{t}, \pi(s_{t}))$ or $r_{t+1} = R(s_{t}, \pi(s_{t}))$

Afterwards, we sample another transition. The number of timesteps that this process can be allowed to go on for is a hyperparameter, k. The MBPO paper finds a formula for the optimal k for a given upper bound in dynamics model error. For simplicity, we won't go into that right now - the full proof is beyond my pay grade - but suffice it to say that k is a parameter that we can tune to optimize performance.

</details>
<br/>

### RAMBO-RL Formulation
Finally, the big payoff. 

$$\pi = \underset{\pi \in \Pi}{\text{arg max}}\ \underset{\hat{\pi} \in \hat{\Pi}}{\text{min}}\ V_M^{\pi, \hat{\pi}}$$

What's the hat over $\pi$ doing?

Well, here's the trick. We consider the dynamics model $\hat{T}(s' \vert s,a)$ to be another policy model $\hat{\pi}(s' \vert s, a)$ that plays the game of trying to select the worst s' so that $\pi(a \vert s=s')$ has a hard time. We define $\hat{\Pi}$ to be the space of all dynamics models within a certain divergence bound of the dynamics model trained to optimize accuracy. 

Let's look at our formulation again.
$$\pi = \underset{\pi \in \Pi}{\text{arg max}}\ \underset{\hat{\pi} \in \hat{\Pi}}{\text{min}}\ V_M^{\pi, \hat{\pi}}$$
We're selecting the best policy (maximal return, according to the value function) under the condition that the dynamics model for it is the worst possible one that still accurately predicts state transitions.

Expanding this out:

$$\pi = \underset{\pi \in \Pi}{\text{arg max}}\ \underset{\hat{T} \in \hat{M_D}}{\text{min}}\ V^{\pi}_{\hat{T}}$$

Where

$$M_D = \left\{\hat{T}\ |\ \mathbb{E}_D\left[TV \left(\hat{T}_{MLE}(\cdot|s, a), \hat{T}(\cdot|s, a) \right)^2\right] \leq \xi \right\}$$

This is a lot to take in. 

The first thing to understand is that $\hat{T}_{MLE}$ is the dynamics model that's trained solely using Maximum Likelihood Estimation, which means it's trained only to be as accurate as possible in predicting state transitions.


Next is that $TV(P_1, P_2)$ is the total variation distance between distributions $P_1$ and $P_2$. It's defined as the maximum difference between the probabilities that $P_1$ and $P_2$ assign to any event. 

Taken together, the above means that we're selecting a dynamics model, $\hat{T}$, that minimizes the return of a policy while maintaining the property that that $\left(\hat{T}(s'\vert s, a) - \hat{T}_{MLE}(s'\vert s, a) \right)^2 \leq \xi$ for all possible $(s', s, a) \sim D$.