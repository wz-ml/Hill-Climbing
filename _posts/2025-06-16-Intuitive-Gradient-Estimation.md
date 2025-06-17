---
layout: post
title:  "Gradient Estimation, Intuitively Explained"
date:   2025-06-16 12:27:49 -0700
categories: post
excerpt_separator: <!--more-->
toc: true
---

Neural networks work well through the magic of backpropagation, but there are times
when we can't backpropagate through our layers. Let's imagine we have a simple problem:
we want to make our neural network cheaper by routing our computation through one of many possible layers or "experts", like so:

<div class="mermaid" style="width: 70%; height: 70%; margin: 0 auto;">
graph TD
    A[Input] --> B{Router}
    B -->|Route 1| C[Expert 1]
    B -->|Route 2| D[Expert 2]
    B -->|Route 3| E[Expert 3]
    C --> F[Output]
    D --> F
    E --> F
</div>

<!--more-->

Our router just takes in our input vector $x \in \mathbb{R}^d$ and outputs a probability distribution over our experts.

<div id="plot1" style="width: 50%; height: 20%; margin: 0 auto;">
<script>
    document.addEventListener('DOMContentLoaded', function() {
    var x = tf.randomNormal([3]);
    var softmaxed = tf.exp(x).div(tf.sum(tf.exp(x)));
    const softmaxed_cpu = softmaxed.arraySync();
    var data = [{
        type: 'heatmap',
        z: [softmaxed_cpu],
        x: ['Expert 1', 'Expert 2', 'Expert 3'],
        y: [''],
        colorscale: 'Blues',
        reversescale: true,
        colorbar: {
            title: 'Probability',
            len: 1
        },
        zmin: 0,
        zmax: 1
    }];
    var layout = {
        title: 'Router Output',
        width: 400,
        height: 150,
        margin: {
            l: 50,
            r: 50,
            t: 35,
            b: 35
        }
        }
        Plotly.newPlot('plot1', data, layout);
    });
</script>
</div>

Mathematically, we express this as $p = \text{softmax}(W_r x)$, where $W_r \in \mathbb{R}^{d \times k}$ is our routing matrix. We can sample from this distribution to get a single expert to route to ($i \sim p$), before running the rest of our network as usual ($l_{out} = e_i(x)$).

<aside>
<details>

<summary>Defining Terms</summary>

<ul>
<li>$d$ is the dimension of our input vector.</li>
<li>$k$ is the number of experts.</li>
<li>$p$ is the probability distribution over our experts.</li>
<li>$W_r$ is our routing matrix.</li>
<li>$x$ is our input vector.</li>
<li>$e_i(x)$ is the output of expert $i$ on input $x$.</li>
</ul>

</details>

</aside>
<br/>

When it comes to the backwards pass, we have a problem: how do we update our routing layer?

The backpropagation calculus gives us:

$$\underset{Weight\ Gradient}{\frac{\partial L}{\partial W_r}} = \left[\underset{\text{Incoming Gradient}}{\frac{\partial L}{\partial i}} \right] \left[\underset{\text{Local Gradient}}{\frac{\partial p}{\partial W_r}}\right]$$

where $\frac{\partial L}{\partial p}$ is the gradient of the loss with respect to the probability distribution output by our router.