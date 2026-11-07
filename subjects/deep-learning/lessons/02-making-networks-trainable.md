# Module 2 — Making networks trainable

_Week 2 · Core lesson · Authored 2 November 2026 · Revised 7 November 2026_

[Read the formatted HTML lesson](02-making-networks-trainable.html) · [Course contents](../index.html) · [Previous: Module 1](01-from-conventional-ml-to-differentiable-models.html)

## Outcomes and starting point

Module 1 showed how a network calculates a prediction, how backpropagation calculates gradients, and how a parameter update uses those gradients. Here we ask what happens when those operations are mathematically valid but training makes little progress. You should already be able to follow an affine layer, ReLU, the chain rule and a mean batch loss. Familiarity with training and validation splits is assumed.

By the end, you should be able to explain how initialisation and activation functions affect activation and gradient magnitudes. You should also be able to trace why gradients shrink or grow with depth, compare SGD, momentum and Adam, and explain the different roles of normalisation, residual connections, weight decay and dropout. Most importantly, you should be able to turn an observed training failure into a diagnostic hypothesis and a controlled comparison, while distinguishing **trainability** from **generalisation**.

We will use a deeper version of the inspection model from Module 1: numerical image features enter several hidden layers, and a final affine layer predicts a defect-severity score. The numerical examples and diagnostic observations are hypothetical.

## Why a larger network can fit the training data worse

Suppose a defect-severity model with two hidden layers learns to fit the training data, while a model with six hidden layers barely improves. There may be a set of weights that lets the deeper model fit the data well. Whether training can find such weights from its starting point is a separate question.

One reason training can stall is that later layers can dampen changes made by earlier ones. Suppose an early layer’s output increases by 0.1. If each of the next two layers passes on only half the change, the prediction increases by just 0.025. Backpropagation uses those same factors when calculating how much the loss responds to the early layer. If the resulting gradients are small, a fixed learning rate in SGD produces small changes to that layer’s weights. The deeper model may therefore have room to fit the data much better while making little progress with the current training setup.

**Trainability** is how reliably the chosen training setup reduces the training loss within a given budget. **Generalisation** is how well the trained model performs on unseen observations from the population of interest.

Initialisation sets the starting parameter values. Four components then determine how each training step works. **The architecture** specifies the forward computation. **The objective** scores its predictions. **Backpropagation** differentiates that objective through the computation. **The optimiser** uses the gradients and its stored state to update parameters. Training therefore depends on more than whether the forward pass returns an output or the loss has a derivative.

You choose the number and width of the layers, the initialisation rule, and the optimiser settings. These are **hyperparameters**. During training, the optimiser updates the model’s weights and biases to reduce the loss.

For each batch, the order is forward computation, loss, backpropagation, then update. Repeat that cycle with new batches. During inference, the network calculates predictions without gradient-based updates; some components, notably batch normalisation and dropout, also change behaviour between training and inference.

The same loss pattern can have several causes. Use the observations below to decide which explanation to test.

| What you observe | What might explain it | What would distinguish the explanations? |
| --- | --- | --- |
| **Training loss stays high or spikes repeatedly.** | If later layers dampen the effect of changes in earlier layers, early weights can receive tiny gradients and barely move. Large optimiser steps can overshoot regions of lower loss. A model with limited capacity can also settle at a high training loss. | **Compare gradients and weight changes across layers.** Tiny gradients and updates concentrated in early layers support the first explanation. Large updates followed by loss spikes support overshooting, especially if a smaller learning rate reduces the spikes. If training is stable and a wider model reaches a lower training loss under comparable conditions, limited capacity becomes more plausible. Width can also change how easily a model trains. |
| **The model fits the training data well but performs poorly on held-out data.** | The model may have learned patterns specific to the training set. The held-out data may also differ in ways that affect how well those patterns transfer. For example, a gene-expression model might partly rely on a processing-batch signal if cancer status and batch were correlated in the training cohort. | **Look at when and where performance drops.** If held-out performance worsens as training fit improves on comparable data, that supports overfitting. If the drop is concentrated at a particular site or in a particular processing batch, check what differs in that group and whether the model relies on those differences. |
| **Training and validation losses both fall slowly.** | The optimiser may be taking small steps. The loss can also be relatively flat along the directions the updates take, so even substantial weight changes may have little effect. | **Compare the size of the weight changes with the reduction in loss.** If updates are consistently tiny relative to the weights and a modestly higher learning rate reduces training loss faster at the same update count, step size was limiting progress in that comparison. If a higher rate causes spikes while progress stays slow at a stable rate, one possibility is that the loss is much steeper in some directions than in others. |

Compare losses measured in comparable conditions. A training objective that includes a weight penalty differs from a validation data loss without that penalty. Dropout and batch normalisation can also make training-mode and evaluation-mode losses differ. Before interpreting a gap, know which computation each curve records.

An early layer can learn too slowly — or too fast. To see why, we will follow the activations forward and the gradients backward: each layer multiplies them, so they can shrink or grow with depth. Only then can we choose an intervention for a particular failure.

## Depth multiplies the factors that carry a gradient

### Follow one network before examining its derivatives

Each step backward through a single chain multiplies the loss derivative by a connecting weight and the local slope of the activation. Across many layers, those factors can make the derivative shrink or grow.

Take an inspection model with one numerical input, two one-unit ReLU layers and an affine output. Use input $x=1$, first hidden weight $1$, second hidden weight $2$, and output weight $1$. All biases are zero. The first preactivation is $z_1=1$, so the first activation is $h_1=1$. The second preactivation is $z_2=2h_1=2$, so the second activation is $h_2=2$. The output gives $\hat y=h_2=2$. For target $y=1$, the half squared loss $\mathcal L=\tfrac12(\hat y-y)^2$ is $0.5$.

Work backward with the target and all parameters fixed. The loss derivative with respect to the prediction is $\hat y-y=1$. The output weight is $1$ and the second ReLU's local slope is $1$, so the loss derivative with respect to $z_2$ is $1\times1\times1=1$. Moving back to $z_1$ adds two factors. The connection from $h_1$ to $z_2$ has weight $2$, and the first ReLU has local slope $1$. The loss derivative with respect to $z_1$ is therefore $1\times2\times1=2$. If we now vary the first weight alone, its effect on $z_1$ is multiplied by the input $x=1$, giving a loss derivative of $2\times1=2$ for that weight.

ReLU has slope $1$ at positive preactivations and $0$ at negative preactivations. At zero its mathematical derivative is undefined. As in Module 1, backpropagation here uses a local slope of $0$ at zero, so no gradient passes through that activation.

For a network with $L$ hidden layers, let $h_0=x$, where $x$ is now the input feature vector. In hidden layer $\ell$, the weight matrix $W_\ell$ and bias vector $b_\ell$ produce the preactivation vector $z_\ell=W_\ell h_{\ell-1}+b_\ell$. Each row of $W_\ell$ contains one receiving unit’s weights. Its columns correspond to the entries of $h_{\ell-1}$. Applying the chosen activation $\phi_\ell$ entry by entry gives $h_\ell=\phi_\ell(z_\ell)$. The affine output uses the weight row $W_{\mathrm{out}}$ and bias $b_{\mathrm{out}}$ to predict $\hat y=W_{\mathrm{out}}h_L+b_{\mathrm{out}}$.

Let $\delta_\ell$ collect the loss derivatives with respect to the entries of $z_\ell$, and let $d_{\mathrm{out}}=\partial\mathcal L/\partial\hat y$ be the scalar loss derivative at the prediction. The output provides the starting value for the last hidden layer. From there, the same chain rule applies between adjacent hidden layers:

$$
\delta_L=(W_{\mathrm{out}}^T d_{\mathrm{out}})\odot\phi_L'(z_L),
\qquad
\delta_\ell=(W_{\ell+1}^T\delta_{\ell+1})\odot\phi_\ell'(z_\ell)
\quad (\ell<L).
$$

The symbol $\odot$ means entrywise multiplication. Entry $i$ of $W_{\ell+1}^T\delta_{\ell+1}$ adds the contributions from all later units connected to activation $i$. Multiplying by the local slope at preactivation $i$ gives the corresponding entry of $\delta_\ell$. In the two-hidden-layer trace there is only one entry at each step: the output starts $\delta_2$ at $(1\times1)\times1=1$, and the hidden-layer formula gives $\delta_1=(2\times1)\times1=2$.[1](#ref-1)

For one observation, the corresponding weight gradient is

$$
\frac{\partial\mathcal L}{\partial W_\ell}
=\delta_\ell h_{\ell-1}^{T}.
$$

This outer product has the same shape as $W_\ell$. It reveals two reasons a weight gradient can be small: the loss derivatives collected in $\delta_\ell$ can be small, or the incoming activation $h_{\ell-1}$ can be small. For a mean batch loss, average these per-observation contributions. A small gradient is not automatically a fault; it is suspicious when a model with substantial remaining error receives too little useful update to reduce that error.

### A scalar calculation makes the multiplication visible

To isolate the effect of repeated multiplication, choose a single illustrative path with 12 successive local derivatives, each equal to $0.5$. A derivative of 1 at the end becomes $0.5^{12}\approx0.000244$ at the beginning. If each factor is $1.5$, the result is $1.5^{12}\approx129.75$. These are examples of **vanishing gradients** and **exploding gradients**: derivatives repeatedly shrink or grow as they propagate through the computation.

A real network has many paths, matrix products and input-dependent derivatives. Some directions can shrink while others grow. The scalar calculation illustrates multiplication; it does not predict every gradient from the network's depth alone.

The activation function affects those factors. A sigmoid activation $\sigma(z)=1/(1+e^{-z})$ has derivative $\sigma(z)(1-\sigma(z))$, at most $1/4$ and much smaller for large positive or negative preactivations. In those saturated regions, substantial changes in $z$ barely change the activation. That small local slope reduces the loss derivative passed through the activation. ReLU avoids positive-side saturation, but a unit that stays negative for all relevant inputs receives zero gradient through ReLU. Changing a negative unit's incoming weights through that path cannot revive it while those derivatives remain zero.[1](#ref-1),[2](#ref-2)

Large positive ReLU outputs are also unbounded. Replacing sigmoid with ReLU does not prevent large weight products from amplifying activations or gradients. Activation choice and initial weight magnitudes have to work together.

## Initialisation sets the scale at which learning begins

### Different units need different starting weights

If all hidden units in a layer start with identical incoming weights and biases, they calculate identical activations. In an otherwise symmetric network, they can receive identical updates and remain copies of one another. Random weights break that symmetry. Biases can often start at zero because distinct weights already make the units different.

The magnitude of the random weights controls how quickly activations shrink or grow at the start of training. For ReLU, **fan-in He initialisation** sets the weight variance to $2/n$, where the fan-in $n$ is the number of inputs. In the 64-input example below, this gives a standard deviation of $\sqrt{2/64}\approx0.1768$. To see why this value works, consider what one unit does to the average squared activation. Write its weighted sum as $z=\sum_{j=1}^{n}w_jh_j$, choosing a zero bias for this calculation. We choose an initialisation rule that draws independent weights with mean zero and variance $s^2$, independently of the incoming activations. The variance describes the distribution of starting weights, not a constraint on their values after training. Let the incoming activations have a common **second moment** $q=E[h_j^2]$, meaning their average squared magnitude. Averaging over the random initialisation and input distribution gives the approximate scale relation

$$
E[z^2]\approx n s^2 q.
$$

The second moment includes both variability and a nonzero mean. For instance, a constant activation of 2 has variance zero but second moment 4. We use the second moment because ReLU outputs generally have a positive mean. When the preactivation distribution is approximately symmetric about zero, positive and negative values contribute about equally to $E[z^2]$ because $z$ and $-z$ have the same square. ReLU keeps the positive contribution and sets the negative contribution to zero. The resulting second moment is

$$
E[\operatorname{ReLU}(z)^2]\approx\frac12 n s^2q.
$$

The factor $n$ appears because the unit sums $n$ weighted inputs. If the incoming width increases while weight variance stays fixed, that sum tends to grow. Setting the multiplier $ns^2/2$ to $1$ gives $s^2=2/n$, which approximately preserves the forward second moment.[2](#ref-2)

For the same 64-input ReLU unit, compare He initialisation with a smaller weight standard deviation of $0.05$. That gives a second-moment multiplier of $64(0.05)^2/2=0.08$ per layer under this approximation. Across six such layers, the multiplier is $0.08^6=2.62144\times10^{-7}$. Tiny initial weights can therefore make activations rapidly collapse even though the network contains many parameters.

This calculation describes an initialisation model, not a guarantee for trained networks. Learned correlations, changing weight distributions, unequal widths, biases and other components alter the result. Preserving a forward average also does not guarantee that every backward direction has a useful scale.

### Match the rule to the activation and architecture

**Glorot, or Xavier, initialisation** balances forward and backward scale with weight variance approximately $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$ under its assumptions. Here, $n_{\mathrm{in}}$ is fan-in and $n_{\mathrm{out}}$ is fan-out, the number of receiving units. It is a common starting point for roughly linear or suitably scaled tanh layers. He initialisation compensates specifically for the loss of second moment through ReLU. For 64 inputs and 64 receiving units, Glorot gives a standard deviation of $\sqrt{2/(64+64)}=0.125$, compared with $0.1768$ from He. The extra variance in the He rule compensates for the negative preactivations that ReLU zeros.[1](#ref-1),[2](#ref-2)

Use the initialisation prescribed for the actual architecture and activation rather than applying a hidden-layer rule to every tensor. Output layers can have different requirements. Standardising input features using training-set statistics also helps avoid an initial weighted sum being dominated by one feature's units; it does not replace hidden-layer initialisation or justify using held-out data to fit preprocessing.

When a deep model begins with shrinking activation magnitudes and early-layer gradients, inspect initialisation and input scale before adding regularisation. An optimiser cannot reliably compensate for every layer starting in a poorly scaled state.

## The optimiser turns derivatives into parameter changes

Backpropagation answers “how sensitive is this batch loss to each parameter?” It does not decide the step size or how much to trust a noisy batch. Let $\theta_t$ collect the parameters before update $t$, and let $g_t$ be the gradient of the mean loss on that batch. The **learning rate** $\eta_t$ controls the update scale.

You choose the initial learning rate and any rule for changing it. To establish a working rate, compare a few multiplicatively spaced candidates for the chosen optimiser while keeping the data, architecture and update budget fixed. Inspect whether training loss falls stably, then use validation performance to compare useful candidates. There is no single suitable rate for all architectures, loss scales or optimisers.

### SGD and momentum use gradients in different ways

**Stochastic gradient descent (SGD)** uses

$$
\theta_{t+1}=\theta_t-\eta_tg_t.
$$

“Stochastic” refers to using sampled observations or batches rather than the full training dataset for each update. Even a correct batch gradient differs from the full-dataset gradient, so consecutive updates can point in different directions. A very small learning rate makes progress slow; a large one can repeatedly overshoot regions with lower loss or produce numerical divergence.

To see why a large derivative may require a small learning rate, consider the one-parameter loss $\mathcal L(\theta)=a\theta^2/2$, where $a>0$ controls curvature: larger $a$ means a sharper increase away from zero. Its gradient is $a\theta$, so SGD gives $\theta_{t+1}=(1-\eta a)\theta_t$. The magnitude shrinks only when $0<\eta a<2$. A real network has multiple directions with different curvature; a step that works along a gently changing direction can overshoot along a sharply changing one. This quadratic calculation explains a local constraint, not a universal learning-rate bound for networks.

**Momentum** combines the current gradient with an accumulated gradient from earlier updates. Call the accumulated gradient $u_t$. You choose the retention factor $\mu$ between 0 and 1: it multiplies the previous accumulation, so a larger value carries more of that history into the next update. Starting with $u_0=0$, we use the following update:

$$
u_t=\mu u_{t-1}+g_t,
\qquad \theta_{t+1}=\theta_t-\eta_tu_t.
$$

Repeated gradient signs reinforce the accumulated gradient, while rapidly alternating signs tend to cancel. Momentum can therefore make more consistent progress through noisy updates. It can also carry the model past a good region if the learning rate or retained accumulation is too large. Some descriptions instead use an average $u_t=\mu u_{t-1}+(1-\mu)g_t$. For the same sequence of gradients, starting from zero with fixed $\mu$, that average is $1-\mu$ times the accumulation used here, so the same numerical learning rate produces a smaller step. Separate training runs can develop different gradients because their earlier updates differ. Check the equation, not just the name “momentum,” when comparing settings.[10](#ref-10)

### Adam estimates a scale for each parameter

Two parameters can receive gradients of very different magnitudes. **Adam** keeps both an exponentially weighted mean gradient and an exponentially weighted mean squared gradient. Call them $m_t$ and $v_t$; they start at zero. Here $m_t$ is an array of running gradient averages, not the hidden width $m$ from Module 1. Adam's $v_t$ averages **squared** gradients. The momentum accumulator $u_t$ above uses the gradients themselves. You choose retention factors $\beta_1$ and $\beta_2$; larger values retain more of the previous estimates:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

All squares and divisions in Adam are entrywise. Because both estimates start at zero, Adam corrects their initial bias using $\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$. For a constant gradient $g$ in one coordinate, the first average after $t$ updates is $(1-\beta_1)(1+\beta_1+\cdots+\beta_1^{t-1})g=(1-\beta_1^t)g$. Dividing by $1-\beta_1^t$ recovers $g$. The same calculation for the squared gradient gives $(1-\beta_2^t)g^2$ and explains the second correction. At step $t$, $\theta_t$ is the parameter before the step. The moments $m_t$ and $v_t$ already include the gradient at $\theta_t$ and are used to compute $\theta_{t+1}$:

$$
\theta_{t+1}=\theta_t-
\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The small positive constant $\epsilon$ prevents division by zero. For each parameter, a large running average of squared gradients produces a larger denominator, reducing its step relative to the mean gradient in the numerator. Adam adapts the step using that parameter's gradient history; it does not determine which input features matter or guarantee generalisation.[3](#ref-3)

For a numerical comparison, choose a starting parameter $\theta_1=1$, a first batch gradient of 2 and learning rate $0.1$. The round numbers make the update sizes easy to compare. SGD changes the parameter to $0.8$. Momentum also changes it to $0.8$ on its first step because the previous accumulation is zero. For Adam, choose $\beta_1=0.9$ and $\beta_2=0.999$, common starting settings rather than learned values. Then $m_1=0.2$ and $v_1=0.004$; bias correction gives $\hat m_1=2$, $\hat v_1=4$. Ignoring a negligible $\epsilon$, the new parameter is $1-0.1(2/2)=0.9$. The same numerical learning rate therefore does not imply the same parameter change across optimisers. Start with the intended implementation's documented retention settings and compare changes when training behaviour warrants them.[3](#ref-3),[13](#ref-13)

| Method | Useful starting reason | Main caution |
| --- | --- | --- |
| SGD | A transparent baseline with few moving parts | Can be slow or sensitive to a single global learning rate |
| SGD with momentum | Gradient directions contain useful consistency across batches | Accumulation changes the effective update scale and can overshoot |
| Adam | Coordinate scales differ or gradients are noisy or sparse | Still requires a learning rate; adaptation does not fix disconnected gradients or guarantee the best held-out result |

Treat optimiser choice as a comparison to test under a stated budget. Counting epochs alone can hide differences in batch size and number of updates. Compare data exposure, update count, compute and the metric you actually care about.

### Gradient clipping caps the gradient norm before the update

If finite gradient norms spike just before the loss spikes, compare a run with global norm clipping. Let $g$ collect the current batch-loss derivatives with respect to all trainable parameters. For nonzero $\lVert g\rVert_2$, multiply $g$ by $\min(1,c/\lVert g\rVert_2)$, where $c>0$ is the chosen threshold. For example, a norm of $12$ and a threshold of $3$ multiply every entry by $1/4$. A norm of $2$ is unchanged. The scaling preserves the gradient's direction. In plain SGD it also caps the step norm at $\eta_t c$. With momentum or Adam, the optimiser's stored state also determines the actual step.

Choose a threshold by inspecting the model's finite gradient norms and measuring how often clipping occurs and whether loss spikes subside. Check nonfinite gradients separately. A zero gradient remains zero.[12](#ref-12)

### Learning-rate schedules change the update scale over time

A single learning rate need not serve the whole run. A larger early rate may produce useful progress, while a smaller later rate can reduce oscillation and refine the fit. A **schedule** specifies $\eta_t$ over updates: step decay lowers it at chosen milestones, while cosine decay lowers it smoothly toward a chosen minimum. A **warmup** increases it from a small starting value to a target rate during the opening updates.

For example, use linear warmup from $0.001$ at update 1 to $0.01$ at update 101, then cosine decay from $0.01$ to $0.001$ over the next 900 updates. Halfway through the decay, at update 551, the rate is $0.0055$. It reaches $0.001$ at update 1001. The warmup lets the run begin with smaller steps, while the decay reduces step size later without an abrupt drop.

You choose the schedule's duration, milestones and minimum rate; these are not values learned by backpropagation. Compare them within a fixed update budget after establishing a stable initial rate. Warmup can help when early activations or optimiser estimates are still settling, but it cannot reconnect a missing gradient. Decay that happens too soon can leave a poorly fitted model making very small updates. Record whether schedule steps count batches or epochs, since the same stated duration can otherwise describe different runs. For the original cosine schedule with warm restarts, see [11](#ref-11); a restart raises the rate again in a later cycle, whereas warmup describes its initial increase.

## Normalisation controls intermediate values

Initialisation acts once. Hidden activations then change as parameters learn. A normalisation layer repeatedly centres and rescales selected groups of activations, giving later computations a more controlled input scale. The choice of group determines both the computation and the dependencies it introduces.

### Batch normalisation uses other observations in the batch

Take a dense-layer preactivation matrix $Z$ of shape $B\times m$, where $B$ is the batch size and $m$ is the hidden width: rows are observations, columns are hidden units. **Batch normalisation (BatchNorm)** computes a separate mean and variance down each column during training. Thus each unit uses $B$ values, one from each observation. Its learned scale and offset are reused for that unit across all observations. In a convolutional layer, the usual grouping combines batch and spatial positions separately for each channel.[4](#ref-4)

Take the following batch of three observations and three hidden units. The first column contains the preactivations for unit 1.

$$
Z=\begin{bmatrix}
1&2&6\\
3&4&7\\
5&8&9
\end{bmatrix}.
$$

To normalise the first entry with BatchNorm, use the column values $1,3,5$. The mean is 3 and the variance estimate is $(4+0+4)/3=8/3$. Use unit scale and zero offset to isolate the effect of centring and scaling. Ignoring the small denominator stabiliser for this calculation, the first observation's normalised value is $(1-3)/\sqrt{8/3}\approx-1.225$. Change only the third observation's preactivation from 5 to 9. The new mean is $13/3$, the variance is $104/9$, and the first normalised value becomes $-10/\sqrt{104}\approx-0.981$. The value entering BatchNorm for the first observation did not change, but its training-time output did because the batch statistics changed.

For a group of $M$ scalar values $u_1,\ldots,u_M$, define its mean $\bar u$ and variance estimate $s_u^2$ used to normalise a training group by

$$
\bar u=\frac1M\sum_{i=1}^M u_i,
\qquad s_u^2=\frac1M\sum_{i=1}^M(u_i-\bar u)^2.
$$

First centre and rescale each value to obtain $\tilde u_i$. Then apply the learned scale and offset to produce the normalisation layer's output $u_i^{\mathrm{out}}$:

$$
\tilde u_i=\frac{u_i-\bar u}{\sqrt{s_u^2+\epsilon}},
\qquad u_i^{\mathrm{out}}=\gamma\tilde u_i+\beta.
$$

The positive constant $\epsilon$ makes the division defined even for a constant group. The network learns the scale $\gamma$ and offset $\beta$ by backpropagation, just as it learns weights and biases. In contrast, the mean $\bar u$ and variance $s_u^2$ are calculated from the values in the group. The $1/M$ divisor is used for the group variance in this training calculation. With independent samples from the same distribution, $1/(M-1)$ instead gives the usual unbiased sample variance estimate. Because changing one group member also changes the calculated mean and variance, normalisation changes how derivatives connect the group's values.

In usual inference mode, BatchNorm uses stored running statistics accumulated during training, together with the learned scale and offset. It does not estimate the deployed observation's statistics from the current inference batch. Consequently, incorrect mode selection, poorly estimated running statistics or a deployment distribution shift can damage predictions. Small or strongly correlated batches can also make training statistics unreliable; dense BatchNorm with just one observation has no useful within-column variation.

The original paper motivated BatchNorm partly through “internal covariate shift,” meaning changes in intermediate input distributions during training. Later experiments showed that reducing that proposed shift is not a sufficient explanation of its benefits; changes to the smoothness of the optimisation problem provide another account. Use the centring-and-scaling computation to reason about the layer, rather than treating its original motivation as a settled causal explanation.[4](#ref-4),[5](#ref-5)

### Layer normalisation uses features within one observation

For the same $B\times m$ matrix, **layer normalisation (LayerNorm)** computes statistics across the $m$ hidden activations in each row. One observation's statistics therefore do not depend on other observations. Standard LayerNorm has a learned scale and offset for each normalised feature, reused across observations. It calculates each observation's mean and variance from that observation in both training and inference.[6](#ref-6)

Apply LayerNorm to the first row of the same matrix. Its group is $1,2,6$, with mean $3$ and variance $14/3$. With unit scale, zero offset and negligible $\epsilon$, the first entry becomes $(1-3)/\sqrt{14/3}\approx-0.926$. Changing the third observation's first entry from $5$ to $9$ leaves that first row unchanged, so its LayerNorm output stays at $-0.926$. The same change moved the first entry's BatchNorm output from $-1.225$ to $-0.981$ because it changed the first column.

For a sequence tensor with shape $B\times T\times d$, $B$ counts observations, $T$ counts sequence positions and $d$ counts features at each position. A common LayerNorm choice calculates a mean and variance across the $d$ features separately for each observation and position. It does not automatically normalise across time. Always identify the actual axes: “normalise the layer” alone does not specify the operation.

| Property | Dense BatchNorm on $B\times m$ | LayerNorm on $B\times m$ |
| --- | --- | --- |
| Values used for one mean and variance | One feature across $B$ observations | All $m$ features within one observation |
| Other observations affect training output | Yes | No |
| Usual inference statistics | Stored running estimates | Statistics of the current observation |
| Important limitation | Batch composition and train/inference mismatch | Centring and scaling can remove useful within-observation level or magnitude information |

Neither operation replaces training-set input preprocessing, and neither guarantees better performance. LayerNorm's independence from batch size is useful when batches are small, but it changes the representation differently from BatchNorm. The correct choice depends on which variation the task needs to preserve and which grouping the architecture expects.

## Residual connections change the path through the model

A plain stack requires each block to produce the complete next representation. A **residual block** instead computes a correction and adds it to its input. Let $h$ be the incoming activation vector, and let $F(h;\theta_F)$ be a learned branch with parameters $\theta_F$. The block produces

$$
h_{\mathrm{next}}=h+F(h;\theta_F).
$$

The two terms must have the same shape. When they do, a zero residual branch gives the identity map $h_{\mathrm{next}}=h$. This gives the parameterised block a direct way to preserve an existing representation while learning a change. It helps explain why simply adding depth need not require every new block to relearn how to pass useful information onward.[7](#ref-7)

To see the derivative through a residual addition, first consider a single scalar input with an illustrative branch $F(h)=0.2h$. The output is $h_{\mathrm{next}}=h+0.2h=1.2h$, so a change $\Delta h$ produces $1.2\Delta h$ at the output. If the loss derivative there is 3, the input loss derivative is $3(1+0.2)=3+0.6=3.6$. The direct addition contributes 3 and the learned branch contributes 0.6. These contributions add because the same input affects both terms.

For vectors, the following equation collects those contributions. Define $J_F$ as the matrix whose entry $(i,j)$ is $\partial F_i/\partial h_j$: it describes how each branch output changes with each input entry. Let $I$ be the identity matrix. If $g_{\mathrm{next}}$ is the column-vector loss gradient at the block output, the input gradient is

$$
g_h=(I+J_F)^Tg_{\mathrm{next}}
=g_{\mathrm{next}}+J_F^Tg_{\mathrm{next}}.
$$

In the scalar example, $I$ is $1$, $J_F$ is $0.2$, and $g_{\mathrm{next}}$ is $3$. The two terms are $3$ and $0.2\times3=0.6$, which recover $3.6$.

One contribution passes directly through the addition without multiplying by the residual branch's weights. However, the two contributions can reinforce or cancel. In a scalar example, $F(h)=-h$ gives derivative $1+(-1)=0$. A residual connection provides a useful route; it does not guarantee that the total gradient cannot vanish or explode.

If the input vector and residual output have different dimensions, choose a learned linear projection with matrix $P$ that maps the input to the required output dimension. For $n_{\mathrm{in}}$ incoming entries and $n_{\mathrm{out}}$ output entries, $P$ has shape $n_{\mathrm{out}}\times n_{\mathrm{in}}$. The shortcut computes $Ph$, so the sum is $s=Ph+F(h)$. If an elementwise activation $\psi$ follows the sum, first multiply the gradient at its output $g_{\mathrm{out}}$ by the activation slopes to obtain $g_s=g_{\mathrm{out}}\odot\psi'(s)$. The two paths then give $g_h=P^Tg_s+J_F^Tg_s$. These details matter when reading an architecture diagram: the simple identity-route equation applies to the addition exactly as written. Module 4 will examine how residual CNN blocks handle changes in spatial resolution and channel count.

Normalisation and residual connections can complement one another, but they solve different problems. Normalisation changes the scale and dependencies of a representation; a residual addition changes how representations and derivatives can pass through the architecture. Adding either without a diagnostic reason makes its effect harder to interpret.

## Regularisation changes what training is encouraged to learn

Once the network can reduce its training loss, you still need useful predictions on new observations. Weight decay and dropout are tools for that problem, although both also affect optimisation. They are not default repairs for a network whose gradients are disconnected or whose activations have collapsed.

### L2 penalties and weight decay coincide only in particular updates

In this example, let $w$ collect the layer weight matrices flattened into a vector. Biases and normalisation scales are excluded from this penalty. Let $\lambda\geq0$ control its strength. Adding an L2 penalty gives the objective

$$
\mathcal J=\mathcal L_{\mathrm{data}}+\frac\lambda2\lVert w\rVert_2^2.
$$

The penalty's gradient is $\lambda w$. Under plain SGD, let $g_t$ be the data-loss gradient with respect to the selected weights $w_t$. Their update becomes

$$
w_{t+1}=w_t-\eta_t(g_t+\lambda w_t)
=(1-\eta_t\lambda)w_t-\eta_tg_t.
$$

The penalty makes an improvement in data fit worthwhile only if it outweighs the extra cost of larger weights. It can discourage large coefficients that help fit incidental training variation, but strong penalties can also suppress relationships that would generalise. Under this plain SGD rule, the penalty is equivalent to shrinking weights by a common factor on each step. For an illustrative scalar weight, choose $w_t=2$, $g_t=0.5$, $\eta_t=0.1$ and $\lambda=0.01$. Substituting these values gives the next weight:

$$
w_{t+1}=(1-0.1\times0.01)(2)-0.1(0.5)=0.999(2)-0.05=1.948.
$$

With Adam, adding $\lambda w$ to the gradient also feeds that term into the moving averages and coordinate-wise scaling. It no longer produces the same uniform shrinkage. **AdamW** decouples weight decay from Adam's gradient adaptation: Adam's moments use the data gradient, and a separate term $-\eta_t\lambda w_t$ shrinks the selected weights. This distinction is why “Adam with an L2 penalty” and “AdamW” are different update rules.[8](#ref-8)

When configuring weight decay, specify which parameter groups receive it. Strong decay can prevent a useful fit. For scale-invariant normalised layers, shrinking weights can also change effective optimisation behaviour, so smaller weights alone are not a complete explanation of generalisation.

You choose $\lambda$ rather than learning it as a prediction weight. The value $0.01$ above makes the shrinkage visible in the arithmetic. Compare several strengths using validation performance, with the optimiser and loss reduction fixed, because their definitions affect how strongly the penalty acts.

### Dropout changes the training computation

**Dropout** randomly removes selected activations during training, so later units cannot always rely on the same collection of incoming features. You choose the probability $p$ of dropping an activation; backpropagation does not learn $p$ in the setup described here. Let $r_i$ independently equal 1 with probability $1-p$, otherwise 0. In *inverted dropout*, training uses

$$
\tilde h_i=\frac{r_i h_i}{1-p},\qquad 0\leq p<1.
$$

Conditional on the original activation, $E[\tilde h_i\mid h_i]=h_i$. Choose $h_i=4$ and $p=0.5$ to make the two possible outcomes easy to compare: the training output is either 0 or 8, each with probability one half. Its average is 4. At inference, inverted dropout leaves the activation unchanged; it does not randomly discard features or scale them again.[9](#ref-9)

Matching this activation's expectation does not mean a nonlinear network's deterministic prediction equals the exact average of all dropout predictions. Dropout introduces noise and changes the training objective in effect; it can improve robustness to reliance on particular activations, but can also slow fitting or cause underfitting. Its placement near BatchNorm can create a mismatch between noisy training statistics and inference activations. Use the intended architecture's arrangement and measure the effect rather than adding dropout everywhere.

For an application, compare a no-dropout baseline with a few probabilities while holding other settings fixed. Use validation results to judge whether the extra noise helps; if training fit deteriorates without a held-out benefit, the chosen probability or placement is not helping that comparison.

## Worked diagnosis: make a hypothesis distinguishable from alternatives

Return to the six-hidden-layer inspection regressor. For this example it receives 64 precomputed numerical image features, rather than Module 1's two features. We choose width 64 for every hidden layer so the same fan-in calculation applies at all six layers, including the first. Matching width to input count is a teaching convenience, not a design rule. The hidden activations are ReLU, the output is affine, and the loss is mean squared prediction error. Suppose the training loss barely falls, hidden activations become much smaller with depth, and early-layer gradients are tiny. The following is a worked investigation of a hypothetical case.

### Establish that the observations describe a real optimisation problem

First inspect data and the computation. Confirm that inputs and targets align, predictions and targets have matching shapes, the loss is reduced as intended, gradients are cleared between updates, backpropagation reaches the trainable parameters, and the optimiser actually updates those parameters. Check for nonfinite values. These implementation failures can imitate architectural problems.

Next, try fitting a fixed, learnable subset in a separate diagnostic run. For example, select 32 observations and train on that same subset with dropout disabled. If the model uses BatchNorm, record its mode and batch size because both affect the calculation. State separately whether the diagnostic objective includes a weight penalty. A model that cannot substantially reduce error on a learnable small subset warrants implementation, capacity or optimisation investigation before generalisation tuning. Success would show that this setup can fit those observations; it would not establish generalisation or the correctness of every other part of the pipeline.

For the hypothetical case, assume these checks pass. Input features are standardised using training data. The hidden weights were drawn independently with standard deviation $0.05$, and hidden biases are zero. That makes the initialisation-scale explanation concrete enough to test.

### Predict an observable consequence before changing a component

The hypothesis is that the hidden initialisation makes forward activations collapse, contributing to small early-layer gradients and updates. Under the approximation derived earlier, each width-64 ReLU layer multiplies the activation second moment by $0.08$. For the standardised training features in this example, the mean is $0$ and the variance is $1$, so the input second moment is $E[X^2]=\operatorname{Var}(X)+E[X]^2=1$. The predicted progression is:

| Location | Predicted second moment with standard deviation 0.05 | With fan-in He initialisation |
| --- | ---: | ---: |
| Input | 1 | 1 |
| After hidden layer 1 | 0.08 | Approximately 1 |
| After hidden layer 3 | 0.000512 | Approximately 1 |
| After hidden layer 6 | 0.000000262144 | Approximately 1 |

A finite network can depart from these expectations because they average over the initial weights and inputs. The prediction is a strong depth-related decline with the original rule and a substantially flatter initial scale with He initialisation.

### Change one factor and compare the quantities the hypothesis predicts

Compare the original hidden initialisation with fan-in He initialisation, leaving the output-layer rule unchanged. Hold the architecture, dataset split, preprocessing, mean-loss convention, optimiser, learning rate, batch order and update budget fixed. For paired seeds, use the same underlying random draws and rescale the hidden weights according to each rule. Repeat across a small fixed set of seeds so a fortunate draw is less likely to determine the conclusion. The changed factor is the hidden initialisation rule.

Before the first update, record each layer's activation root-mean-square value, which is the square root of its measured mean squared activation. During training, record the data loss, fraction of zero ReLU activations, per-layer gradient norms and actual parameter changes. A gradient norm measures the magnitude of all entries in a layer's gradient; one useful choice is the square root of their summed squares. To interpret update size relative to existing parameters, record

$$
\rho_\ell=
\frac{\lVert\Delta W_\ell\rVert_F}
{\lVert W_\ell\rVert_F+\epsilon_\rho}.
$$

Here $\Delta W_\ell$ is the actual change from one update, the matrix norm means the square root of the sum of squared entries, and the small positive constant $\epsilon_\rho$ protects a zero denominator. For example, if a layer's weight norm is $2$ and one update has norm $0.002$, the ratio is approximately $0.001$, or $0.1\%$. If its next update has norm $0.006$ with a similar weight norm, the ratio rises to about $0.3\%$. Compare these trajectories across layers and between the two conditions alongside the training loss. With Adam, actual updates are especially informative because raw gradient magnitudes do not directly determine step magnitudes.

If the new initialisation flattens the initial activation scales, increases useful early-layer updates and consistently improves training loss at the fixed budget, the results support this hypothesis. They do not prove initialisation was the only cause or establish better held-out performance. If activation scales improve but loss does not, investigate update size, output/loss compatibility, inactive ReLUs or other bottlenecks. If the scales were already stable, the initialisation hypothesis loses support; changing more components at once would hide that result.

### Choose the next intervention from the remaining evidence

| Remaining observation | Plausible next comparison | Reason and limitation |
| --- | --- | --- |
| Loss spikes or diverges; updates are very large | Reduce the learning rate with other settings fixed | Tests overshooting; also check numerical and data errors |
| Finite but very large gradient norms precede spikes | Compare with [global gradient-norm clipping](#gradient-clipping-caps-the-gradient-norm-before-the-update) | Caps the gradient norm before the optimiser step. Check whether actual update and loss spikes subside |
| Scales are reasonable, updates are persistently tiny, loss falls slowly | Test a larger learning rate within a bounded range | Tests conservative steps; a larger rate may become unstable |
| Deeper plain model fits worse after scale and update checks | Compare a suitable residual version | Tests an architectural route to easier fitting; changes architecture, so use a separate comparison |
| Training fit is good but held-out error worsens | Test weight decay or dropout individually; inspect data and validation | Targets generalisation, with a possible cost to training fit |

The central skill is to write down an explanation that could be contradicted by the comparison. A falling loss after changing initialisation, optimiser, normalisation and dropout together would show that the new configuration works better under that run. It would not tell you which explanation was right.

## What to carry into representation learning

Trainability depends on the entire chain from input scale to activation derivatives to actual updates. Initialisation establishes a starting scale; optimisers and schedules determine how parameters change; normalisation and residual paths modify the computation through which values and derivatives travel. Weight decay and dropout change the learning pressure and training behaviour, with potential benefits and costs for held-out performance.

Diagnose in that causal order when the evidence supports it: verify the computation, inspect scales and updates, make one controlled change, then assess both training fit and held-out behaviour. The ability to fit the training data and the usefulness of the learned representation are separate questions. Module 3 examines the latter: what information hidden features retain, what their similarities mean, and how the objective shapes that organisation.

## References

Primary sources checked on 2 November 2026.

1. <span id="ref-1"></span>Glorot, X. & Bengio, Y. (2010). [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html). *AISTATS*, PMLR 9, 249–256. Initialisation, saturation and propagation of activation/gradient scale.
2. <span id="ref-2"></span>He, K., Zhang, X., Ren, S. & Sun, J. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852). *ICCV*. Rectifier-aware initialisation and its assumptions.
3. <span id="ref-3"></span>Kingma, D. P. & Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *ICLR*. Adaptive moment estimates and bias-corrected update.
4. <span id="ref-4"></span>Ioffe, S. & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://proceedings.mlr.press/v37/ioffe15.html). *ICML*, PMLR 37, 448–456. Training statistics, learned affine parameters and inference behaviour.
5. <span id="ref-5"></span>Santurkar, S., Tsipras, D., Ilyas, A. & Madry, A. (2018). [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604). *NeurIPS*. Evidence challenging the original shift-based explanation and analysing optimisation smoothness.
6. <span id="ref-6"></span>Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). Statistics within an observation and independence from batch composition.
7. <span id="ref-7"></span>He, K., Zhang, X., Ren, S. & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *CVPR*. Residual formulation, identity and projection shortcuts.
8. <span id="ref-8"></span>Loshchilov, I. & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). *ICLR*. Distinction between adaptive L2 updates and decoupled weight decay.
9. <span id="ref-9"></span>Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html). *JMLR* 15, 1929–1958. Dropout mechanism; this lesson uses the equivalent inverted-scaling convention.
10. <span id="ref-10"></span>PyTorch documentation, version 2.9. [SGD](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html). Explicit update conventions and implementation differences for momentum.
11. <span id="ref-11"></span>Loshchilov, I. & Hutter, F. (2017). [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). *ICLR*. Cosine schedules and warm restarts.
12. <span id="ref-12"></span>Pascanu, R., Mikolov, T. & Bengio, Y. (2013). [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html). *ICML*, PMLR 28(3), 1310–1318. Gradient-norm clipping; recurrent models themselves are covered in Module 5.
13. <span id="ref-13"></span>PyTorch documentation, version 2.9. [Adam](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html). Gradient-moment updates and documented starting retention settings; consulted for implementation details rather than as a claim about the latest release.
