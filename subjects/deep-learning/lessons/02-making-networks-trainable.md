# Module 2 — Making networks trainable

_Week 2 · Core lesson · Authored 2 November 2026_

[Read the formatted HTML lesson](02-making-networks-trainable.html) · [Course contents](../index.html) · [Previous: Module 1](01-from-conventional-ml-to-differentiable-models.html)

## Outcomes and starting point

Module 1 showed how a network calculates a prediction, how backpropagation calculates gradients, and how a parameter update uses those gradients. Here we ask what happens when those operations are mathematically valid but training makes little progress. You should already be able to follow an affine layer, ReLU, the chain rule and a mean batch loss. Familiarity with training and validation splits is assumed.

By the end, you should be able to explain how initialisation and activation functions affect signal scale; trace why gradients can shrink or grow with depth; compare SGD, momentum and Adam; and explain the different roles of normalisation, residual connections, weight decay and dropout. Most importantly, you should be able to turn an observed training failure into a diagnostic hypothesis and a controlled comparison, while distinguishing **trainability** from **generalisation**.

We will use a deeper version of the inspection model from Module 1: numerical image features enter several hidden layers, and a final affine layer predicts a defect-severity score. All numerical examples and diagnostic observations below are illustrative. They are not measurements from a trained inspection model.

## Why a larger network can fit the training data worse

Suppose a two-hidden-layer inspection model reduces training loss steadily, but a six-hidden-layer model barely improves. More layers give you more adjustable transformations, yet the training algorithm may struggle to find useful parameters. The distinction matters: a flexible model can represent useful functions without your chosen initialisation and update rule being able to reach them efficiently.

The entire training process has four cooperating parts. **The architecture** specifies the forward computation. **The objective** scores its predictions. **Backpropagation** differentiates that objective through the computation. **The optimiser** uses the gradients and its stored state to update parameters. Initialisation chooses where this process begins. Training therefore depends on more than whether the forward pass returns an output or the loss has a derivative.

For each batch, the order is forward computation, loss, backpropagation, then update. Repeat that cycle with new batches. During inference, the network calculates predictions without gradient-based updates; some components, notably batch normalisation and dropout, also change behaviour between training and inference.

A useful first distinction is between two problems you already recognise from conventional machine learning:

| Observation | What it suggests | What it does not establish |
| --- | --- | --- |
| Training loss remains high or becomes unstable | Investigate implementation, objective, signal scale, gradients and updates | It does not establish that the architecture lacks capacity |
| Training loss is low but held-out performance is poor | Investigate generalisation, data leakage, distribution mismatch and objective–metric alignment | A more powerful optimiser alone need not help |
| Training and validation both improve slowly | Several explanations remain: small updates, difficult optimisation, insufficient capacity or a noisy objective | The curves alone cannot identify the cause |

Compare losses measured in comparable conditions. A training objective that includes a weight penalty differs from a validation data loss without that penalty. Dropout and batch normalisation can also make training-mode and evaluation-mode losses differ. Before interpreting a gap, know which computation each curve records.

To understand why an early layer may learn slowly, we will first follow the numbers travelling forward and the gradients travelling backward. Only then can we choose an intervention for a particular failure.

## Depth multiplies the factors that carry a gradient

### Follow one network before examining its derivatives

For one observation, let $x$ be the column vector of input features. The network has $L$ hidden layers, followed by an affine output layer that predicts one score. Let $h_0=x$. Hidden layer $\ell$ receives the previous activation vector $h_{\ell-1}$ and computes

$$
z_\ell=W_\ell h_{\ell-1}+b_\ell,
\qquad h_\ell=\phi(z_\ell).
$$

Here, $W_\ell$ and $b_\ell$ are that layer's weights and biases; $z_\ell$ contains its preactivations; and $\phi$ applies an activation function entry by entry. If layer $\ell-1$ has $n_{\ell-1}$ units and layer $\ell$ has $n_\ell$, then $W_\ell$ has shape $n_\ell\times n_{\ell-1}$ and $h_\ell$ has $n_\ell$ entries. Input values and activations depend on the observation; the same parameters are reused for every observation. The final prediction is $\hat y=W_{\mathrm{out}}h_L+b_{\mathrm{out}}$, and its loss is the scalar $\mathcal L$.

To update an early weight, backpropagation needs to know how changing its layer's output changes the final loss. Define $\delta_\ell=\partial\mathcal L/\partial z_\ell$: entry $i$ is the loss's derivative with respect to preactivation $i$ in layer $\ell$. For adjacent hidden layers, the chain rule gives

$$
\delta_\ell=
\left(W_{\ell+1}^{T}\delta_{\ell+1}\right)
\odot\phi'(z_\ell).
$$

The symbol $\odot$ means entrywise multiplication. The transpose sends the later layer's derivatives back to the units that contributed to it. Multiplication by $\phi'(z_\ell)$ then accounts for how each activation changes with its preactivation. Repeating this relation multiplies many weight-dependent transformations and activation derivatives.[1](#ref-1)

For one observation, the corresponding weight gradient is

$$
\frac{\partial\mathcal L}{\partial W_\ell}
=\delta_\ell h_{\ell-1}^{T}.
$$

This outer product has the same shape as $W_\ell$. It reveals two reasons a weight gradient can be small: the backward derivative $\delta_\ell$ can be small, or the incoming activation $h_{\ell-1}$ can be small. For a mean batch loss, average these per-observation contributions. A small gradient is not automatically a fault; it is suspicious when a model with substantial remaining error receives too little useful update to reduce that error.

### A scalar calculation makes the multiplication visible

Imagine a single path with 12 successive local derivatives, each equal to $0.5$. A derivative of 1 at the end becomes $0.5^{12}=0.000244$ at the beginning. If each factor is $1.5$, the result is $1.5^{12}=129.75$. These are examples of **vanishing gradients** and **exploding gradients**: derivatives repeatedly shrink or grow as they propagate through the computation.

A real network has many paths, matrix products and input-dependent derivatives. Some directions can shrink while others grow. The scalar calculation illustrates multiplication; it does not predict every gradient from the network's depth alone.

The activation function affects those factors. A sigmoid activation $\sigma(z)=1/(1+e^{-z})$ has derivative $\sigma(z)(1-\sigma(z))$, at most $1/4$ and much smaller for large positive or negative preactivations. In those saturated regions, substantial changes in $z$ barely change the activation. The backward derivative therefore shrinks at that operation. ReLU has derivative 1 for positive preactivations and 0 for negative ones. It avoids positive-side saturation, but a unit that stays negative for all relevant inputs receives zero gradient through ReLU. Changing a negative unit's incoming weights through that path cannot revive it while those derivatives remain zero.[1](#ref-1),[2](#ref-2)

Large positive ReLU outputs are also unbounded. Replacing sigmoid with ReLU does not prevent large weight products from amplifying activations or gradients. Activation choice and parameter scale have to work together.

## Initialisation sets the scale at which learning begins

### Different units need different starting weights

If all hidden units in a layer start with identical incoming weights and biases, they calculate identical activations. In an otherwise symmetric network, they can receive identical updates and remain copies of one another. Random weights break that symmetry. Biases can often start at zero because distinct weights already make the units different.

The magnitude of those random weights matters as much as their differences. Consider one unit with $n$ inputs. Write its weighted sum as $z=\sum_{j=1}^{n}w_jh_j$, omitting a zero bias. Suppose at initialisation the weights are independent, have mean zero and variance $s^2$, and are independent of the incoming activations. Suppose the incoming activations have a common second moment $q=E[h_j^2]$. Averaging over the random initialisation and input distribution gives the approximate scale relation

$$
E[z^2]\approx n s^2 q.
$$

The second moment is the average squared magnitude; it includes both variability and a nonzero mean. We use it because ReLU outputs generally have a positive mean. Calling their second moment a variance would discard that distinction. Under an approximately symmetric preactivation distribution, ReLU keeps half the squared magnitude on average, so

$$
E[\operatorname{ReLU}(z)^2]\approx\frac12 n s^2q.
$$

The factor $n$ appears because the unit sums $n$ weighted inputs. If the incoming width increases while weight variance stays fixed, that sum tends to grow. Choosing $s^2=2/n$ approximately preserves the forward second moment for ReLU. This is **He initialisation** with a fan-in convention: *fan-in* means the number of inputs to a unit.[2](#ref-2)

For 64 incoming units, the corresponding weight standard deviation is $\sqrt{2/64}\approx0.1768$. A standard deviation of $0.05$ instead gives a second-moment multiplier of $64(0.05)^2/2=0.08$ per layer under this approximation. Across six such layers, the multiplier is $0.08^6=2.62144\times10^{-7}$. Tiny initial weights can therefore make activations rapidly collapse even though the network contains many parameters.

This calculation describes an initialisation model, not a guarantee for trained networks. Learned correlations, changing weight distributions, unequal widths, biases and other components alter the result. Preserving a forward average also does not guarantee that every backward direction has a useful scale.

### Match the rule to the activation and architecture

**Glorot, or Xavier, initialisation** balances forward and backward scale with weight variance approximately $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$ under its assumptions. Here, $n_{\mathrm{in}}$ is fan-in and $n_{\mathrm{out}}$ is fan-out, the number of receiving units. It is a common starting point for roughly linear or suitably scaled tanh layers. He initialisation compensates specifically for the loss of second moment through ReLU. The two prescriptions differ because they model different transformations.[1](#ref-1),[2](#ref-2)

Use the initialisation prescribed for the actual architecture and activation rather than applying a hidden-layer rule to every tensor. Output layers can have different requirements. Standardising input features using training-set statistics also helps avoid an initial weighted sum being dominated by one feature's units; it does not replace hidden-layer initialisation or justify using held-out data to fit preprocessing.

When a deep model begins with shrinking activation magnitudes and early-layer gradients, inspect initialisation and input scale before adding regularisation. An optimiser cannot reliably compensate for every layer starting in a poorly scaled state.

## The optimiser turns derivatives into parameter changes

Backpropagation answers “how sensitive is this batch loss to each parameter?” It does not decide the step size or how much to trust a noisy batch. Let $\theta_t$ collect the parameters before update $t$, and let $g_t$ be the gradient of the mean loss on that batch. The **learning rate** $\eta_t$ controls the update scale.

### SGD and momentum use gradients in different ways

**Stochastic gradient descent (SGD)** uses

$$
\theta_{t+1}=\theta_t-\eta_tg_t.
$$

“Stochastic” refers to using sampled observations or batches rather than the full training dataset for each update. Even a correct batch gradient differs from the full-dataset gradient, so consecutive updates can point in different directions. A very small learning rate makes progress slow; a large one can repeatedly overshoot regions with lower loss or produce numerical divergence.

To see why a large derivative may require a small learning rate, consider the one-parameter loss $\mathcal L(\theta)=a\theta^2/2$, where $a>0$ controls curvature: larger $a$ means a sharper increase away from zero. Its gradient is $a\theta$, so SGD gives $\theta_{t+1}=(1-\eta a)\theta_t$. The magnitude shrinks only when $0<\eta a<2$. A real network has multiple directions with different curvature; a step that works along a gently changing direction can overshoot along a sharply changing one. This quadratic calculation explains a local constraint, not a universal learning-rate bound for networks.

**Momentum** accumulates a running direction. In the convention used here, $v_t$ is an accumulated gradient, $\mu$ is a retention factor between 0 and 1, and $v_0=0$:

$$
v_t=\mu v_{t-1}+g_t,
\qquad \theta_{t+1}=\theta_t-\eta_tv_t.
$$

Repeated gradient signs reinforce the running direction, while rapidly alternating signs tend to cancel. Momentum can therefore make more consistent progress through noisy updates. It can also carry the model past a good region if the learning rate or retained direction is too large. Some descriptions multiply $g_t$ by $1-\mu$; their accumulated values and equivalent learning rates differ. Keep the update convention fixed when comparing numbers.[10](#ref-10)

### Adam estimates a scale for each parameter

Two parameters can receive gradients of very different magnitudes. **Adam** keeps both an exponentially weighted mean gradient and an exponentially weighted mean squared gradient. Call them $m_t$ and $v_t$; they start at zero. Here $v_t$ has a different role from the momentum variable above. With retention factors $\beta_1$ and $\beta_2$,

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

All squares and divisions in Adam are entrywise. Because both estimates start at zero, Adam corrects their initial bias using $\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$. The update is

$$
\theta_{t+1}=\theta_t-
\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The small positive constant $\epsilon$ prevents division by zero. A coordinate with a large history of squared gradients gets a larger denominator, reducing its step relative to the numerator. This is adaptation to gradient scale, not knowledge of which features matter or an assurance of generalisation.[3](#ref-3)

For a numerical comparison, suppose a single parameter starts at $\theta_1=1$ and the first batch gradient is 2. With learning rate $0.1$, SGD changes it to $0.8$. Momentum also changes it to $0.8$ on its first step under our convention. Adam with $\beta_1=0.9$, $\beta_2=0.999$ has $m_1=0.2$ and $v_1=0.004$; bias correction gives $\hat m_1=2$, $\hat v_1=4$. Ignoring a negligible $\epsilon$, its new value is $1-0.1(2/2)=0.9$. The same numerical learning rate therefore does not imply the same parameter change across optimisers.

| Method | Useful starting reason | Main caution |
| --- | --- | --- |
| SGD | A transparent baseline with few moving parts | Can be slow or sensitive to a single global learning rate |
| SGD with momentum | Gradient directions contain useful consistency across batches | Accumulation changes the effective update scale and can overshoot |
| Adam | Coordinate scales differ or gradients are noisy or sparse | Still requires a learning rate; adaptation does not fix disconnected gradients or guarantee the best held-out result |

Treat optimiser choice as a comparison to test under a stated budget. Counting epochs alone can hide differences in batch size and number of updates. Compare data exposure, update count, compute and the metric you actually care about.

### Learning-rate schedules change the update scale over time

A single learning rate need not serve the whole run. A larger early rate may produce useful progress, while a smaller later rate can reduce oscillation and refine the fit. A **schedule** specifies $\eta_t$ over updates: step decay lowers it at chosen milestones, while cosine decay lowers it smoothly toward a chosen minimum. A **warmup** increases it from a small starting value to a target rate during the opening updates.

Warmup can help when early activations or optimiser estimates are still settling, but it cannot reconnect a missing gradient. Decay that happens too soon can freeze a poorly fitted model. Record whether schedule steps count batches or epochs, since the same stated duration can otherwise describe different runs. Change the learning rate before layering on an elaborate schedule when you have not yet established a stable baseline. For the original cosine schedule with warm restarts, see [11](#ref-11); a restart raises the rate again in a later cycle, whereas warmup describes its initial increase.

## Normalisation controls intermediate values

Initialisation acts once. Hidden activations then change as parameters learn. A normalisation layer repeatedly centres and rescales selected groups of activations, giving later computations a more controlled input scale. The choice of group determines both the computation and the dependencies it introduces.

For a group of $M$ scalar values $u_1,\ldots,u_M$, define its mean $\bar u$ and variance estimate $s_u^2$ by

$$
\bar u=\frac1M\sum_{i=1}^M u_i,
\qquad s_u^2=\frac1M\sum_{i=1}^M(u_i-\bar u)^2.
$$

Each value becomes

$$
\tilde u_i=\frac{u_i-\bar u}{\sqrt{s_u^2+\epsilon}},
\qquad y_i=\gamma\tilde u_i+\beta.
$$

The positive constant $\epsilon$ makes the division defined even for a constant group. The learned scale $\gamma$ and offset $\beta$ allow the network to adjust the normalised outputs. They are model parameters; $\bar u$ and $s_u^2$ are statistics of the chosen group. Their sharing pattern depends on the normalisation layer. Normalisation also changes derivatives because the group's mean and variance depend on its values.

### Batch normalisation uses other observations in the batch

Take a dense-layer preactivation matrix $Z$ of shape $B\times m$: rows are observations, columns are hidden units. **Batch normalisation (BatchNorm)** computes a separate mean and variance down each column during training. Thus each unit uses $B$ values, one from each observation, and has its own learned scale and offset. In a convolutional layer, the usual grouping combines batch and spatial positions separately for each channel.[4](#ref-4)

For a dense feature with batch values 1, 3 and 5, the mean is 3 and the variance estimate is $(4+0+4)/3=8/3$. With $\gamma=1$, $\beta=0$ and negligible $\epsilon$, the first observation's normalised value is $(1-3)/\sqrt{8/3}\approx-1.225$. Change only the third observation from 5 to 9. The new mean is $13/3$, the variance is $104/9$, and the first normalised value becomes $-10/\sqrt{104}\approx-0.981$. The first observation's input did not change, but its training-time output did because the batch statistics changed.

In usual inference mode, BatchNorm uses stored running statistics accumulated during training, together with the learned scale and offset. It does not estimate the deployed observation's statistics from the current inference batch. Consequently, incorrect mode selection, poorly estimated running statistics or a deployment distribution shift can damage predictions. Small or strongly correlated batches can also make training statistics unreliable; dense BatchNorm with just one observation has no useful within-column variation.

The original paper motivated BatchNorm partly through “internal covariate shift,” meaning changes in intermediate input distributions during training. Later experiments showed that reducing that proposed shift is not a sufficient explanation of its benefits; changes to the smoothness of the optimisation problem provide another account. Use the centring-and-scaling computation to reason about the layer, rather than treating its original motivation as a settled causal explanation.[4](#ref-4),[5](#ref-5)

### Layer normalisation uses features within one observation

For the same $B\times m$ matrix, **layer normalisation (LayerNorm)** computes statistics across the $m$ features in each row. One observation's statistics therefore do not depend on other observations. Standard LayerNorm has a learned scale and offset for each normalised feature and uses the observation's own statistics in both training and inference.[6](#ref-6)

For a sequence tensor with shape $B\times T\times d$, where $T$ indexes positions and $d$ indexes features, a common LayerNorm choice normalises the $d$ features separately at each observation and position. It does not automatically normalise across time. Always identify the actual axes: “normalise the layer” alone does not specify the operation.

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

The derivative also has an additive route. Define $J_F$ as the matrix whose entry $(i,j)$ is $\partial F_i/\partial h_j$, and let $I$ be the identity matrix. If $g_{\mathrm{next}}$ is the column-vector loss gradient at the block output, the input gradient is

$$
g_h=(I+J_F)^Tg_{\mathrm{next}}
=g_{\mathrm{next}}+J_F^Tg_{\mathrm{next}}.
$$

One contribution passes directly through the addition without multiplying by the residual branch's weights. However, the two contributions can reinforce or cancel. In a scalar example, $F(h)=-h$ gives derivative $1+(-1)=0$. A residual connection provides a useful route; it does not guarantee that the total gradient cannot vanish or explode.

If the width or resolution changes, a projection shortcut $P h$ can replace $h$, producing $P h+F(h)$. Its backward route contains $P^T$ instead of an identity matrix. An activation after the addition also contributes its own derivative. These details matter when reading an architecture diagram: the simple identity-route equation applies to the addition exactly as written. Module 4 will examine actual residual CNN blocks and their shape changes in detail.

Normalisation and residual connections can complement one another, but they solve different problems. Normalisation changes the scale and dependencies of a representation; a residual addition changes how representations and derivatives can pass through the architecture. Adding either without a diagnostic reason makes its effect harder to interpret.

## Regularisation changes what training is encouraged to learn

Once the network can reduce its training loss, you still need useful predictions on new observations. Weight decay and dropout are tools for that problem, although both also affect optimisation. They are not default repairs for a network whose gradients are disconnected or whose activations have collapsed.

### L2 penalties and weight decay coincide only in particular updates

Let $w$ collect the weights selected for regularisation, and let $\lambda\geq0$ control its strength. Adding an L2 penalty gives the objective

$$
\mathcal J=\mathcal L_{\mathrm{data}}+\frac\lambda2\lVert w\rVert_2^2.
$$

The penalty's gradient is $\lambda w$. Under plain SGD, with data gradient $g_t$, the update becomes

$$
w_{t+1}=w_t-\eta_t(g_t+\lambda w_t)
=(1-\eta_t\lambda)w_t-\eta_tg_t.
$$

The penalty makes an improvement in data fit worthwhile only if it outweighs the extra cost of larger weights. It can discourage large coefficients that help fit incidental training variation, but strong penalties can also suppress relationships that would generalise. Under this plain SGD rule, the penalty is equivalent to shrinking weights by a common factor on each step. With $w_t=2$, $g_t=0.5$, $\eta_t=0.1$ and $\lambda=0.01$, the next weight is $0.999(2)-0.05=1.948$.

With Adam, adding $\lambda w$ to the gradient also feeds that term into the moving averages and coordinate-wise scaling. It no longer produces the same uniform shrinkage. **AdamW** decouples weight decay from Adam's gradient adaptation: Adam's moments use the data gradient, and a separate term $-\eta_t\lambda w_t$ shrinks the selected weights. This distinction is why “Adam with an L2 penalty” and “AdamW” are different update rules.[8](#ref-8)

State which parameters receive decay: biases and normalisation scales are often excluded, but that is a design choice. Strong decay can prevent a useful fit. For scale-invariant normalised layers, shrinking weights can also change effective optimisation behaviour, so smaller weights alone are not a complete explanation of generalisation.

### Dropout changes the training computation

**Dropout** randomly removes selected activations during training, so later units cannot always rely on the same collection of incoming features. Let $p$ be the probability of dropping an activation, and let $r_i$ independently equal 1 with probability $1-p$, otherwise 0. In *inverted dropout*, training uses

$$
\tilde h_i=\frac{r_i h_i}{1-p},\qquad 0\leq p<1.
$$

Conditional on the original activation, $E[\tilde h_i\mid h_i]=h_i$. For example, with $h_i=4$ and $p=0.5$, the training output is either 0 or 8, each with probability one half. Its average is 4. At inference, inverted dropout leaves the activation unchanged; it does not randomly discard features or scale them again.[9](#ref-9)

Matching this activation's expectation does not mean a nonlinear network's deterministic prediction equals the exact average of all dropout predictions. Dropout introduces noise and changes the training objective in effect; it can improve robustness to reliance on particular activations, but can also slow fitting or cause underfitting. Its placement near BatchNorm can create a mismatch between noisy training statistics and inference activations. Use the intended architecture's arrangement and measure the effect rather than adding dropout everywhere.

## Worked diagnosis: make a hypothesis distinguishable from alternatives

Return to the six-hidden-layer inspection regressor. For this example it receives 64 precomputed numerical image features, rather than Module 1's two features. Each hidden layer also has width 64, so every hidden unit has fan-in 64, including the first layer. The hidden activations are ReLU, the output is affine, and the loss is mean squared prediction error. Suppose the training loss barely falls, hidden activations become much smaller with depth, and early-layer gradients are tiny. The following is a worked investigation of a hypothetical case.

### Establish that the observations describe a real optimisation problem

First inspect data and the computation. Confirm that inputs and targets align, predictions and targets have matching shapes, the loss is reduced as intended, gradients are cleared between updates, backpropagation reaches the trainable parameters, and the optimiser actually updates those parameters. Check for nonfinite values. These implementation failures can imitate architectural problems.

Next, try fitting a fixed small subset in a separate diagnostic run with stochastic regularisation disabled. A model that cannot substantially reduce error on a learnable small subset warrants implementation, capacity or optimisation investigation before generalisation tuning. Success would show that this setup can fit those observations; it would not establish generalisation or the correctness of every other part of the pipeline.

For the hypothetical case, assume these checks pass. Input features are standardised using training data. The hidden weights were drawn independently with standard deviation $0.05$, and hidden biases are zero. That makes the initialisation-scale explanation concrete enough to test.

### Predict an observable consequence before changing a component

The hypothesis is that the hidden initialisation makes forward activations collapse, contributing to weak learning signals. Under the approximation derived earlier, each width-64 ReLU layer multiplies the activation second moment by $0.08$. With input second moment 1, the predicted progression is:

| Location | Predicted second moment with standard deviation 0.05 | With fan-in He initialisation |
| --- | ---: | ---: |
| Input | 1 | 1 |
| After hidden layer 1 | 0.08 | Approximately 1 |
| After hidden layer 3 | 0.000512 | Approximately 1 |
| After hidden layer 6 | 0.000000262144 | Approximately 1 |

These are calculated expectations under simplifying assumptions, not measured values. A finite network need not match them exactly. The prediction is a strong depth-related decline with the original rule and a substantially flatter initial scale with He initialisation.

### Change one factor and compare the quantities the hypothesis predicts

Compare the original hidden initialisation with fan-in He initialisation, leaving the output-layer rule unchanged. Hold the architecture, dataset split, preprocessing, mean-loss convention, optimiser, learning rate, batch order and update budget fixed. For paired seeds, use the same underlying random draws and rescale the hidden weights according to each rule. Repeat across a small fixed set of seeds so a fortunate draw is less likely to determine the conclusion. The changed factor is the hidden initialisation rule.

Before the first update, record each layer's activation root-mean-square value, which is the square root of its measured mean squared activation. During training, record the data loss, fraction of zero ReLU activations, per-layer gradient norms and actual parameter changes. A gradient norm measures the magnitude of all entries in a layer's gradient; one useful choice is the square root of their summed squares. To interpret update size relative to existing parameters, record

$$
\rho_\ell=
\frac{\lVert\Delta W_\ell\rVert_F}
{\lVert W_\ell\rVert_F+\epsilon}.
$$

Here $\Delta W_\ell$ is the actual change from one update, the matrix norm means the square root of the sum of squared entries, and $\epsilon$ only protects a zero denominator. There is no universal healthy value of $\rho_\ell$. Compare its trajectory across layers and the two conditions. With Adam, actual updates are especially informative because raw gradient magnitudes do not directly determine step magnitudes.

If the new initialisation flattens the initial activation scales, increases useful early-layer updates and consistently improves training loss at the fixed budget, the results support this hypothesis. They do not prove initialisation was the only cause or establish better held-out performance. If activation scales improve but loss does not, investigate update size, output/loss compatibility, inactive ReLUs or other bottlenecks. If the scales were already stable, the initialisation hypothesis loses support; changing more components at once would hide that result.

### Choose the next intervention from the remaining evidence

| Remaining observation | Plausible next comparison | Reason and limitation |
| --- | --- | --- |
| Loss spikes or diverges; updates are very large | Reduce the learning rate with other settings fixed | Tests overshooting; also check numerical and data errors |
| Finite but very large gradient norms precede spikes | Compare with global gradient-norm clipping | Limits occasional large updates; does not repair vanishing derivatives or diagnose their source |
| Scales are reasonable, updates are persistently tiny, loss falls slowly | Test a larger learning rate within a bounded range | Tests conservative steps; a larger rate may become unstable |
| Deeper plain model fits worse after scale and update checks | Compare a suitable residual version | Tests an architectural route to easier fitting; changes architecture, so use a separate comparison |
| Training fit is good but held-out error worsens | Test weight decay or dropout individually; inspect data and validation | Targets generalisation, with a possible cost to training fit |

For gradient clipping, let $g$ collect the batch's gradients and let $c>0$ be a chosen threshold. Global norm clipping replaces $g$ by $g\min(1,c/\lVert g\rVert_2)$ when its norm is nonzero. It preserves direction while limiting magnitude. The threshold must be interpreted with the loss reduction and model scale; clipping cannot make nonfinite gradients trustworthy or recover gradients that are zero.[12](#ref-12)

The central skill is to write down an explanation that could be contradicted by the comparison. A falling loss after changing initialisation, optimiser, normalisation and dropout together would show that the new configuration works better under that run. It would not tell you which explanation was right.

## What to carry into representation learning

Trainability depends on the entire chain from input scale to activation derivatives to actual updates. Initialisation establishes a starting scale; optimisers and schedules determine how parameters change; normalisation and residual paths modify the computation through which values and derivatives travel. Weight decay and dropout change the learning pressure and training behaviour, with potential benefits and costs for held-out performance.

Diagnose in that causal order when the evidence supports it: verify the computation, inspect scales and updates, make one controlled change, then assess both training fit and held-out behaviour. The ability to fit the training data and the usefulness of the learned representation are separate questions. Module 3 examines the latter: what information hidden features retain, what their similarities mean, and how the objective shapes that organisation.

## References

Primary sources checked on 2 November 2026. Numerical examples, the inspection scenario and the diagnostic table are original teaching constructions. Source results do not validate those hypothetical observations.

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
