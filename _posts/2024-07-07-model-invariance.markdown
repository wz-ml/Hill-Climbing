---
layout: post
title:  "Why DNN Loss Landscapes aren't Convex"
date:   2024-07-07 21:04:30 -0600
categories: post
excerpt_separator: <!--more-->
---

## Introduction
I was speaking to a friend recently about model complexity, when I remarked that model loss landscapes weren't convex. The loss landscape has "tracks" that you can smoothly change your model's parameters along without changing its loss:
![](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/b9b31a496c92e3a953fd8f5a7294650502df2aa32aa8f6e3.png)

There's two reasons why:
- Standard neural network architectures allows multiple sets of weights to implement the same "function."Roughly speaking, imagine increasing a weight in a layer and then reducing the corresponding weights in the next layer.
- Model training creates additional directions of travel where different implemented functions perform equally well on the dataset and have the same loss.
<!--more-->

### Example of case 1:

Let's say we have two layers, denoted by (`W1`, `b1`) and (`W2`, `b2`).

Shapes: 
- $x$ (dim_in)
- $w_1$ (dim_in, dim_l1)
- $b_1$ (dim_l1)
- $w_2$ (dim_l1, dim_l2)
- $b_2$ (dim_l2)
- $a$ (dim_l1)
- $y$ (dim_l2)

$a = relu(x W_1 + b_1)$

$y = a W_2 + b_2$

Writing out the sums:

$$a_j = relu(b_{1_j} + \sum_i x_i w_{1_{ij}})$$

$$y_k =  b_{2_k} + \sum_j a_j w_{2_{jk}}$$

For a given $c$, Let's add $\nu$ to $b_{1_c}$.
$$a_c = relu(b_{1_c} +  \left(\sum_i x_i w_{1_{ic}}\right) + \nu)$$
$$y_k = b_{2_k} + \left(\sum_{j \neq c} a_j w_{2_{jk}} \right) + a_c w_{2_{ck}}$$
If $relu(b_{1_j} + \sum_i x_i w_{1_{ij}}) = 0$ and $relu(b_{1_c} +  \left(\sum_i x_i w_{1_{ic}}\right) + \nu) = 0$:
- There's no impact on the function by changing $\nu$, since the relu kills the change.

If $a_c$ increases by $\nu$:
- We can maintain $y_k$ by modifying $
Otherwise, we can maintain $y_k$ by reducing $b_{2k}$ by $\nu$ or modifying $w_{2_{ck}}$.

Things get a little more tricky when part of the $\nu$ increase is cut off by the relu boundary, but you can informally always reduce $b_{2k}$ by the same amount as $a_c$ increases to maintain the same function $y(x)$.