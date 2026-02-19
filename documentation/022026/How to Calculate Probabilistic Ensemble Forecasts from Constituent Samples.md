How to Calculate Probabilistic Ensemble Forecasts from Constituent Samples: A Note

By Michael Colaresi

Created February 3, 2026
Updated February 8, 2026 (changed theme and corrected a few typos)

To-Do: Make notation match VIEWS  style guide where it can

# Goal

In this memo I describe how to create a probabilistic ensemble forecast from probabilistic constituent models as well as potential mistakes to avoid. 

# Notation

 We represent the multivariate pmf of forecasts for 1-D array $\mathbf{Y}$ which can take on values within the sample space $\mathcal{X} := \mathbb{Z}^{N}_{\ge0}$  with specific fixed values $\mathbf{x} \in \mathcal{X}$  as $Pr(\mathbf{Y}\mid\mathcal{I}) := \mathbb{Pr}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I})~~ \forall~~ \mathbf{x} \in \mathcal{X}$. Where $N$ is the number of observations across space and time being forecasted and the information set that the forecasts are conditioned on is $\mathcal{I}$.  Importantly,  $\mathbf{Y} := \{y_{i}~~ \forall~~i\in\{1,2, \ldots, N\}\}$ is organized by instances while $Pr(y_{i}=x)$ is the marginal pmf for the single $ith$ element of $Y$ taking on one specific value $x$. 

# Quantity of Interest: Probabilistic Ensemble

 Crucially, $\mathbb{Pr}(\mathbf{Y}=\mathbf{x}\mid \mathcal{I})~~\forall ~~ x\in\mathcal{X}$ is the target of our inferences as it represents the probability of each possible 1-D array of count value across each $N$ unit.

We estimate the target of inference potentially using a combination of constituent model forecasts $\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I})$, each indexed by $m \in \{1, \ldots, M\}$, into an ensemble model forecast, denoted as  $\mathbb{Pr}_{ens}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I})$, where the constituent models and any weights or parameters are in the information set $\mathcal{I}$.  

# Creating a Probabilistic Ensemble from Probabilistic Constituent Model Samples

To keep things simple we will start with an equally weighted ensemble where all constituent models are assigned the same importance. This is easily relaxed later into an unequally weighted ensemble below.

## Equally weighted case

The equally weighted case can be expressed as:
$\mathbb{Pr}_{ens}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I},\omega_{m}=\frac{1}{M}~~\forall~~m) = \sum_{m=1}^{M}\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid \mathcal{I})\times \frac{1}{M}$

If we have $S_{1}$ samples for each of the $N$ instances for each of the $M$ constituent models, organized into an $N \times S_{1} \times M$ 3-d array $\Omega$, then  $\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I}) \approx \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{1}}$, where $\Omega[m]$ is the $N\times S$ 2-d array for model $m$.  Notice that $\mathbf{x}$ is a $N$-length vector that is aligned across models. Therefore,

$\sum_{m=1}^{M}\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid \mathcal{I})\times \frac{1}{M} = \sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{1}} \times \frac{1}{M}$

Which can be simplified to $\sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{2}}$ where $S_{2}=S_{1}\times M$.

Practically, $S_{1}$ samples can be drawn, with replacement, from the columns of each $\Omega[m]$ 2-D array and organized into an $N\times S_{2}$ 2-d array, where the values across rows for each sample from different models are aligned in the 0-th dimension to maintain the covariance structure (see below). Specifically, one just draws across the models $S_{1}$ times with $\frac{1}{M}$ probability of sampling for each of the $m$ models, with replacement. We can reference this 2-D array object as $\Omega_{ens}$ (but see note below about stochastic versus quota sampling).

Then we have the even simpler,

$\sum_{m=1}^{M}\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid \mathcal{I})\times \frac{1}{M} = \sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{1}} \times \frac{1}{M} = \mathbb{1}[\Omega_{ens}=\mathbf{x}]\frac{1}{S_{2}}$

## Unequal weights case

If one has weights $\omega_{m}$ where $\sum_{m=1}^{M}\omega_{m} = 1$ and $0 \le\omega_{m} \le 1,~~ \forall~~ m$, then the weighted probabilistic ensemble can be created with:

$\sum_{m=1}^{M}\mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid \mathcal{I})\times \omega_{m} = \sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{1}} \times \omega_{m}$

This simplifies to $\sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{\omega_{m}}{S_{1}}$  and can be calculated by sampling with probability $\frac{\omega_{m}}{S_{1}}$ from the columns of $\Omega[m]$  for each model and creating $\Omega_{ens}$ which is $N \times S_{1}$.  This is similar to the equal weighted case above but there are now different probabilities of sampling across the constituent models.  


# Practical Steps

1. Block-sample aligned, N-length, 1-D arrays for each constituent model $m$. These represent 1 draw of the $N$ true-future or simulated-true-future instances you are interested in. Thus each sample $s\in \{1, \ldots, S_{1}\}$ is organized as  $\{ \hat{y}_{1s}^{(m)}, \hat{y}_{2s}^{(m)}, \ldots, \hat{y}_{Ns}^{(m)} \}$. 
		A. This means jointly drawing a 1-D sample of length $N$ and not independently drawing samples for each instance. The mechanisms for this will depend on the model -- eg whether the model $m$ is parametric, ML with bootstrapped uncertainty, neural network-based using dropout, etc. 
2. If using equal or unequal weights, one can jump directly to creating the 2-D $\Omega_{ens}$ array from the $S_{1}$ samples for each constituent model $m$, being careful to join the samples by instance.
3. Create $\Omega$ by aligning the $m$ $S_{1} \times N$ 2-D arrays $\Omega[m]$ across instances $N$
		A. If the 1-D samples arrays are independent (across the samples not across the instances) then the order of the 1-D samples (columns) does not matter when creating $\Omega$. However, if for some reason the 1-D samples are not independent (across columns) then the order of the columns potentially carries information about the covariance across samples. The usual case is for samples to be indepently drawn. 
4.  If there are different number of samples for distinct constituent models then this should be taken into account as the number of samples can influence the contribution of different constituent models to the ensemble. Reading up on "fair" ensemble evaluation metrics can be helpful.

# Mistakes to Avoid

1) Calculating the expected value ensemble forecast when you want the probabilistic ensemble forecast

When calculating expected value ensemble forecasts from an $M$-length set of expected value constituent models, the target is not to create $\mathbb{Pr}_{ens}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I})$ but instead $\mathbb{E}_{ens}[\mathbf{Y}\mid \mathcal{I}]$.  An equally weighted mapping from the set of expected value constituent models  $\mathbb{E}_{1}[\mathbf{Y}\mid \mathcal{I}], \ldots, \mathbb{E}_{M}[\mathbf{Y}\mid \mathcal{I}]$ to $\mathbb{E}_{ens}[\mathbf{Y}\mid \mathcal{I}]$ is accomplished by creating a simple average over the constituent expectations: $\mathbb{E}_{ens}[\mathbf{Y}\mid \mathcal{I}] = \sum_{m=1}^{M} \mathbb{E}_{m}[\mathbf{Y}\mid \mathcal{I}]\frac{1}{M}$.

It is important to note that while $\mathbb{Pr}_{ens}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I}) \neq \mathbb{E}_{ens}[\mathbf{Y}\mid \mathcal{I}]$, $\mathbb{E}_{ens}[\mathbf{Y}\mid \mathcal{I}] := \{E_{ens}[y_{1} \mid \mathcal{I}], \ldots, E_{ens}[y_{N} \mid \mathcal{I}] \}$ and each $E_{ens}[y_{i} \mid \mathcal{I}]=Pr_{ens}(y_{i}=x \mid \mathcal{I})x~~ \forall ~~ x \in \mathbb{Z_{\ge 0}}$.  Therefore, while the probabilistic ensemble forecast cannot be computed from the ensemble nor the constituent expectations the expected value ensemble forecasts can be computed from the probabilistic ensemble samples.  This fact arises because the expected value is a linear function that does not depend on the covariance across instances or models: $E[(a+b)c] = cE[a]+cE[b]$.

2) Using marginal probabilistic forecasts of constituent models to try and create the probabilistic ensemble forecast.

Calculation of the probabilistic ensemble forecasts from the constituent ensemble forecasts are nonlinear functions, utilizing the indicator function above, and thus depend on the  covariance across the models.  If we define the marginal probabilistic predictions from model $m$ for individual instances $i$ as $Pr_{m}(y_{i}=x)$, which can be estimated from samples based on $\sum_{j=1}^{S_{1}}\mathbb{1}[y^{(m)}_{ij}=x]$, where $y_{i}^{(m)}$ is the $S_{1}$-length 1-D array of samples for model $m$ at instances $i$ and $y_{ij}^{(m)}$ is the $j$-th sample from that 1-D array. 

In general, $\sum_{m=1}^{M} \mathbb{1}[\mathbf{\Omega[m]}=\mathbf{x}]\frac{1}{S_{2}} \neq \{\sum_{m=1}^{M} \sum_{j=1}^{S_{1}} \mathbb{1}[y^{(m)}_{1j}=\mathbf{x}]\frac{1}{S_{2}}, \ldots, \sum_{m=1}^{M} \sum_{j=1}^{S_{1}} \mathbb{1}[y^{(m)}_{Nj}=\mathbf{x}]\frac{1}{S_{2}} \}$

The last term could produce aggregations that are over-estimates or under-estimates depending on whether the covariance structure. Note that this fact is also true of all nonlinear function, including the variance. For example, the variance of the expected value ensemble forecast, $Var(E_{ens}[y_{i} \mid \mathcal{I}])$ is:

$$
Var\Big(\sum_{m=1}^{M} E_{m}[\mathbf{y_{i}}\mid \mathcal{I}]\frac{1}{M}\Big) = \frac{1}{M^{2}}\sum_{j=1}^{M} Var(E_{j}[y_{i}\mid \mathcal{I}]) + \frac{2}{M^{2}}\sum_{1 \ge j \ge k \ge M}Cov(E_{j}[y_{i}],E_{k}[y_{i}])$$

Where the last term relies on the fact that the covariance is symmetric. Crucially, the variance for the ensemble expected value forecast depends on the covariance between the constituent expected value models. Unless all of these covariance terms are 0, and thus the models are independent, the covariance across models for each observation must be calculated.  These covariance terms are not contained in a vector of marginal probabilistic predictions and thus the variance -- and other nonlinear aggregations -- cannot be faithfully calculated.

The blocked, sorted structure of the joint $\Omega$ samples ensures that the covariance is represented for downstream calculation.



# Open questions that need more research

1. How many samples are sufficient for different summaries of the joint ensemble probabilistic forecast? Tail probabilities, smaller quantile-based intervals, or sums over smaller intervals will need more samples as compared to an expected value, larger, central intervals, or sums over larger intervals. 
2. What are the trade-offs for quota sampling versus stochastic sampling  (in equal or unequally weighted ensemble set-ups) across models? For example, quota sampling for an equally weighted ensemble there would be exactly (subject to rounding to an integer) $\frac{1}{M}$ samples per constituent model; while in stochastic sampling there would be a $\frac{1}{M}$ chance of drawing a sample from each constituent model. My intuition is that stochastic sampling is correct because that is the only way you end up with the right $\frac{1}{M}$ or other weights over the long-term, without special cases (no rounding). Quota sampling could be systematically biased to or against a model because of rounding, even in the limit. 
