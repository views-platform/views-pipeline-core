How to Aggregate Joint Probabilistic Forecast Samples Across Arbitrary Subsets of instances Using Nonlinear Functions: A Note

By Michael Colaresi

Created February 6, 2026
Updated February 8, 2026


# Goal

In this memo I describe how to aggregate joint probabilistic forecasts that are stored as samples across arbitrary sets of observations using nonlinear functions. In addition, this note explains why marginal probabilistic forecasts for each instance cannot generally be utilized to create analytically correct aggregations except if the aggregations across instances are strictly linear or the forecasts are independent across instances. 

# Notation

 We represent the multivariate pmf of forecasts from model $m$ for 1-D array $\mathbf{Y}$ which can take on values within the sample space $\mathcal{X} := \mathbb{Z}^{N}_{\ge0}$  with specific fixed values $\mathbf{x} \in \mathcal{X}$  as $Pr_{m}(\mathbf{Y}\mid\mathcal{I}) := \mathbb{Pr}_{m}(\mathbf{Y}=\mathbf{x}\mid\mathcal{I})~~ \forall~~ \mathbf{x} \in \mathcal{X}$. Where $N$ is the number of observations across space and time being forecasted and the information set that the forecasts are conditioned on is $\mathcal{I}$.  Importantly,  $\mathbf{Y} := \{y_{i}~~ \forall~~i\in\{1,2, \ldots, N\}\}$ is organized by instances while $Pr_{m}(y_{i}=x)$ is the marginal probability mass function (pmf) for the single $ith$ element of $Y$ taking on one specific value $x$.  It will be convenient to refer to a 2-D array of samples, with dimensions $N \times S_{1}$ that represent the joint probabilities of possible outcomes over $\mathbf{Y}$ as $\Omega[n]$.  The $m$ index can be dropped if the underlying target data generation process is being referenced and not the predictions. 

In addition define $f_{v}: (Pr_{m}(\mathbf{Y}\mid\mathcal{I}),\Theta_{w}) \rightarrow \mathcal{A}_{\Theta_{w}}^{(m,f_{v})}$ as an aggregation function that will take both that joint probabilistic forecasts from trained model specification $m$   and a set $\Theta_{a}$  that represents a subset of instances that are aggregated over by $f_{v}$. $\mathcal{A}_{\Theta_{w}}^{(m,f_{v})}$ represents the output of function $f_{v}$ when computed on the joint probabilistic forecasts from trained model specification $m$ aggregated over subset $\Theta_{w}$. It is often useful to think about $\mathcal{A}_{\Theta_{w}}^{(m,f_{v})}$ as an answer to a specific question that depends on aggregation. For example, one question posed to a country-year model whose forecasts are stored as jointly sampled simulations might be: what is the probability that there will be 100 or more fatalities over the next 6 months in Mali? This can be answered by setting $f_{v}=\sum_{j=1}^{S_{1}}\sum_{i \in \Theta_{w}} \mathbb{1}[\hat{y}^{(m)}_{ij} \ge 100] \frac{1}{S_{1}}$ where $\hat{y}_{ij}^{(m)}$ is the element in the $i$th row and $j$th column of $\Omega[m]$.

# Quantity of Interest: Aggregate Summary Measure Over Some Arbitrary Sub-set of Instances

 Crucially, the target of inference here is the analytically correct aggregation output $\mathcal{A}_{\Theta_{w}}^{(m,f_{v})}$, given the joint probabilistic forecasts, subsets, and functions of interest. 

We will also explore analytical mistakes in calculating aggregations with a focus on the mistakes that occur when marginal probabilistic forecasts are substituted for joint probabilistic forecasts by mistake. In this case, the incorrect output will be denoted as $\hat{\mathcal{A}}_{\Theta_{w}}^{(m,f_{v})}$ . 


# Types of Aggregation Functions: Linear and Nonlinear

The distinction between linear and nonlinear functions is crucial for understanding how to correctly aggregate (and interpret) multiple values  across instances into outputs.  A linear function have a form that can be described as:
$$
b+ \sum_{i \in \Theta_{w}} c_{i}x_{i}
$$
The mean of $\{ x_{i}~~ \forall~~ i\in \Theta_{w} \}$ is an example of a linear transformation where $b=0$ and $c_{i}=\frac{1}{N}~~ \forall ~~ i\in\Theta_{w}$, as is the raw sum. Linear functions like these are important because by definition $g(a,b) = g(a) + g(b)$. The output of the combined inputs is the same as the sum of the output of the individual inputs.

Nonlinear functions are those that cannot be cast into a linear form. The minimum, maximum, quantiles, variance and covariance, probability of being within a specific range of values (including being above or below a specific threshold), as well as credible intervals and highest posterior density intervals are all nonlinear functions. There are an infinite number of nonlinear functions that are potentially of interest for political violence forecasts. In addition the combination of a linear and a nonlinear function is itself a nonlinear function, so computing the standard error around a sample mean as well as computing and the variance of a sum are nonlinear functions.


Examples of Linear and Nonlinear Aggregation Functions on Joint and Marginal Probabilistic Forecasts

To make clear when there are benefits to using joint probabilistic samples versus marginal samples, imagine we have a joint probabilistic forecast for model $m$, stored as $S_{1}$ samples, over 3 instances with the 2-D array:

$$
\Omega[m] = \left\{
\begin{aligned}
&s_{11} &s_{12} &~~\ldots &s_{1S_{1}} \\
&s_{21} &s_{22} &~~\ldots &s_{2S_{1}} \\
&s_{31} &s_{32} &~~\ldots &s_{3S_{1}}
\end{aligned}
\right\}
$$

We assume that each of the $S_{1}$ samples (of length $N=3$) have been draw together so that any covariance structure has been preserved across the rows.

To make this clear, let's define a specific trained model specification that produces joint probabilistic output over the $N$ instances which we can sample from.  We can set $S_{1} =5$ to keep things manageable. In addition, we will reduce the sample space to represent $\colorbox{lightblue}{\textcolor{black}{small}}$ , $\colorbox{yellow}{\textcolor{black}{medium}}$ and $\colorbox{orange}{\textcolor{black}{high}}$ fatalities. 

The model will generate predications based on the following rules

$$
\begin{align}
Pr(y_{1} = \textcolor{yellow}{\Huge \bullet}) &= .4 \\
Pr(y_{1} = \textcolor{lightblue}{\Huge \bullet}) &= .6 \\
Pr(y_{2}=\textcolor{lightblue}{\Huge \bullet}, y_{3}=\textcolor{lightblue}{\Huge \bullet} | y_{1}=\textcolor{yellow}{\Huge \bullet}) &= 1 \\
Pr(y_{2}=\textcolor{yellow}{\Huge \bullet}, y_{3}=\textcolor{orange}{\Huge \bullet} | y_{1}=\textcolor{lightblue}{\Huge \bullet}) &= 1
\end{align}
$$

When we take 5 samples from this model, jointly, we might get:

$$
\Omega[m] = \left\{
\begin{aligned}
&\textcolor{yellow}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet}~~~~~~~ &\textcolor{lightblue}{\Huge \bullet} &\textcolor{yellow}{\Huge \bullet} ~~~~~~~&\textcolor{lightblue}{\Huge \bullet}\\
&\textcolor{lightblue}{\Huge \bullet} &\textcolor{yellow}{\Huge \bullet}~~~~~~~ &\textcolor{yellow}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet} ~~~~~~~&\textcolor{yellow}{\Huge \bullet}\\
&\textcolor{lightblue}{\Huge \bullet} &\textcolor{orange}{\Huge \bullet}~~~~~~~ &\textcolor{orange}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet} ~~~~~~~ &\textcolor{orange}{\Huge \bullet}
\end{aligned}
\right\}
$$

Note that we are generating samples such that they are independent across columns but not across rows. Instead going down rows in each column encodes the covariance patterns that were knitted by the trained model object rules above.  For example, we can see that when fatalities are $\colorbox{lightblue}{\textcolor{black}{small}}$ in the first instance (top row, second, third, and fifth sample columns), fatalities are $\colorbox{yellow}{\textcolor{black}{medium}}$ in the second and $\colorbox{orange}{\textcolor{black}{high}}$ in the last instance ($\textcolor{lightblue}{\Huge \bullet}\textcolor{yellow}{\Huge \bullet}\textcolor{orange}{\Huge \bullet}$, flipped vertically in those second, third, and fifth columns). When fatalities are $\colorbox{lightblue}{\textcolor{black}{small}}$  in the last instance (bottom row, first and fourth columns) they are $\colorbox{yellow}{\textcolor{black}{medium}}$ in the first instance (top row) and $\colorbox{lightblue}{\textcolor{black}{small}}$ in the last instance ($\textcolor{yellow}{\Huge \bullet}\textcolor{lightblue}{\Huge \bullet}\textcolor{lightblue}{\Huge \bullet}$, flipped vertically in those first and fourth columns).

We could re-order the columns, and/or jointly sample additional columns and still reconstruct the covariance structures within columns across rows. That is the benefit of block sampling across instances for joint probabilistic forecasting models -- we keep the covariance structure.

For comparison, let us also create samples that only reflect the marginal pmfs for each of the three instance, from the model equations, denoting these samples as $\Omega'[m]$: 


$$
\Omega'[m] = \left\{
\begin{aligned}
&\textcolor{lightblue}{\Huge \bullet} &\textcolor{yellow}{\Huge \bullet}~~~~~~~ &\textcolor{lightblue}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet} ~~~~~~~&\textcolor{yellow}{\Huge \bullet}\\
&\textcolor{yellow}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet}~~~~~~~ &\textcolor{lightblue}{\Huge \bullet} &\textcolor{yellow}{\Huge \bullet} ~~~~~~~&\textcolor{yellow}{\Huge \bullet}\\
&\textcolor{orange}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet}~~~~~~~ &\textcolor{orange}{\Huge \bullet} &\textcolor{lightblue}{\Huge \bullet} ~~~~~~~ &\textcolor{orange}{\Huge \bullet}
\end{aligned}
\right\}
$$


Notice that while the percentages of each color by row (across the columns in a row) are similar (although they do not have to be identical because we are sampling) the covariance patterns across rows (in a given column) are not retained. This is because marginal pmfs do not hold, by definition, the shared information across instances.  Unless the forecasts across instances are independent, $Pr_{m}(\mathbf{Y}=\mathbf{x}|\mathcal{I}) \neq \prod_{i}^{N}Pr_{m}(y_{i}=x_{i}), \text{where}~ \mathbf{x}=\{x_{i},\ldots, x_{N}\}~ \text{and}~ i \in \{1,\ldots, N\}$. 

Now let us see how both the jointly sampled and marginally sampled information is transformed by linear and nonlinear aggregation functions

## Linear aggregations function examples: Sum and Mean

First, we can calculate the expected sum of $\colorbox{lightblue}{\textcolor{black}{small}}$ fatality events across all instances for the model from both $\Omega[m]$ and $\Omega'[m]$, where $f_{1}= \sum_{j=1}^{5} \sum_{i\in \Theta_{w}} \mathbb{1}[y_{ij}=\textcolor{lightblue}{\Huge \bullet}]\times \frac{1}{5}$ and $\Theta_{w}=\{1, 2, 3 \}$. 

$$
\begin{align}
f_{1}(\Omega[m], \Theta_{w}) &= 1.4 \\
f_{1}(\Omega'[m], \Theta_{w}) &= 1.4 \\
\end{align}
$$
Second, the expected value/mean is also a linear function, this time simplely:  $f_{2}= \sum_{j=1}^{5} \sum_{i\in \Theta_{w}} \mathbb{1}[y_{ij}=\textcolor{lightblue}{\Huge \bullet}]\times \frac{1}{5\times3}$

$$
\begin{align}
f_{2}(\Omega[m], \Theta_{w}) &\approx 0.4667 \\
f_{2}(\Omega'[m], \Theta_{w}) &\approx 0.4667 \\
\end{align}
$$

### Clarifying point: Inner, Instance-by-instance Nonlinear functions and Outer linear functions

These examples highlight an important point. When $f_{v}$ can be decomposed into $g(h(\cdot))$ where $g(\cdot)$ is an outer-function that is created across instances and $h(\cdot)$ is an inner function that is run instance-by-instance and then that output is fed as input to $g(\cdot)$, the inner function $h(\cdot)$ can be nonlinear but not the outer function $g(\cdot)$. Thus, in both linear examples, the inner function $h(\cdot)$ is an indicator function that only takes one instance at a time. The indicator function is nonlinear but since the input is only one instance, this is not a problem when aggregating across instances (eg the $g(\cdot)$ here) with a simple sum (which is linear).  This is why the sum of instance-by-instance squared errors can be aggregated even though terms are squared -- the square only operates instance by instance. As we will see the variance and other nonlinear functions applied across instances are a different story. 

## Nonlinear aggregation function examples: Variance;  probabilities above, below, or between thresholds; Quantiles; and Conjunctions  

When we move to nonlinear aggregations across units the problems with samples that represent solely marginal probability mass functions arise. First, let us take the variance around the expected value of $\colorbox{lightblue}{\textcolor{black}{small}}$ fatality events. Let $\hat{p}_{j}=\sum_{i\in \Theta_{w}}\mathbb{1}[y_{ij}=\textcolor{lightblue}{\Huge \bullet}] \times \frac{1}{3}$ be the probability of interest for the $jth$ sample and the variance can be expressed as  $f_{3}= \sum_{1}^{5}\hat{p_j}(1-\hat{p_{j}})  \frac{1}{5}$. 

This is not a linear function which only yields a correct answer with the joint probabilistic samples and not the marginal probabilistic forecast samples,

$$
\begin{align}
f_{3}(\Omega[m], \Theta_{w}) &= .2222 \\
f_{3}(\Omega'[m], \Theta_{w}) &= .1778 \\
\end{align}
$$

In this case the naive marginal probabilities underestimate the variance but this can change based on the covariance structure within the model and the question being asked. Therefore, any uncertainty estimates -- even around this linear function -- are likely biased if using marginal pmfs. The only exception is for independence across rows/instances.

Similarly, if we were interested in the expected probability of there being 2 or more events of higher magnitude than $\colorbox{lightblue}{\textcolor{black}{small}}$, so either $\colorbox{orange}{\textcolor{black}{high}}$ or $\colorbox{yellow}{\textcolor{black}{medium}}$ fatality levels, across all three instances, we can define the aggregation function,
 $f_{4}=\sum_{j=1}^{N}\mathbb{1}[\sum_{i\in \Theta_{w}}\mathbb{1}[y_{ij}=\textcolor{lightblue}{\Huge \bullet}] < 2]\times \frac{1}{5}$.

$$
\begin{align}
f_{4}(\Omega[m], \Theta_{w}) &= \frac{3}{5} \\
f_{4}(\Omega[m], \Theta_{w}) &= \frac{2}{5}
\end{align}
$$

Which is again not correct.  Similarly, if we wanted to know the expected probability of there being three instances of above $\colorbox{lightblue}{\textcolor{black}{small}}$ fatality levels across the instances, we would define $f_{5}=\sum_{j=1}^{N}\mathbb{1}[\sum_{i\in \Theta_{w}}\mathbb{1}[y_{ij}=\textcolor{lightblue}{\Huge \bullet}] = 0]\times \frac{1}{5}$

$$
\begin{align}
f_{4}(\Omega[m], \Theta_{w}) &= \frac{0}{5} \\
f_{4}(\Omega[m], \Theta_{w}) &= \frac{1}{5}
\end{align}
$$

So the marginal probability sampling now is an over-estimate. 

One can arbitrarily jointly sample from the trained model specification above to get the correct answers while also seeing that the marginal probability mass function sampling only recovers linear aggregation functions.


## Examples with Count Data Over Space and Time and Using Policy Relevant Aggregation Functions

ToDo: Return to VIEWS notation, set up 2 x 4 by 4 grids of values where there are 2 time points with 16 grid-cells each. Imagine that there are 4 groups of 4 non-overlapping grid-cells and these groups form admin-2 units. Over time there are 32 grid-cells and 8 admin-2 units.

Keep model output sparse, with lots of zeros and then bigger spikes.

To create more realistic examples, imagine we have 32 instances, that represent 2 time points of interest, call these months 1 and 2, and 16 grid-cells laid out in a 4 by 4 pattern at each time point. Index the 21 grid-cell months as:

$$
\overbrace{
\begin{Bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16 \\
\end{Bmatrix}}^{month=1}

\overbrace{
\begin{Bmatrix}
17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 \\
25 & 26 & 27 & 28 \\
29 & 30 & 31 & 32 \\
\end{Bmatrix}}^{month=2}
$$

We will define a map from gridcell-months to the relevant admin-2-months. This map is a dictionary, $\mathcal{G}$, of the form:

$$
\begin{align}
\{A1 &: \{1, 2, 3, 4, 17, 18, 19, 20\} \\
A2   &: \{5, 6, 7, 8, 21, 22, 23, 24\} \\ 
A3   &: \{9, 10, 11, 12, 25, 26, 27, 28\} \\
A4   &: \{13, 14, 15, 16, 29, 30, 31, 32\}

\}
\end{align}
$$

For convenience we also have the mappings $M1$ and $M2$ which map the first 16 grid-cell month indices to the first month and the 17th through 32nd grid-cell month indices to the second month, stored in $\mathcal{G}$.

The probabilistic forecasting model $m$ is blocked sampled yielding an $\Omega[m]$ that is  $32 \times S_{1}$  and set $S_{1}=3$.  The first block sample,  $\Omega[m][,1]$ might look like this:


$$
\overbrace{
\begin{Bmatrix}
10 & 8 & 0 & 0 \\
21 & 6 & 0 & 0 \\
10 & 5 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=1}

\overbrace{
\begin{Bmatrix}
12 & 10 & 4 & 0 \\
32 & 11 & 5 & 0 \\
2 & 4 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=2}
$$

The second, $\Omega[m][, 2]$:
$$
\overbrace{
\begin{Bmatrix}
120 & 83 & 0 & 0 \\
23 & 66 & 0 & 0 \\
21 & 24 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=1}

\overbrace{
\begin{Bmatrix}
125 & 80 & 24 & 0 \\
45 & 78 & 24 & 0 \\
10 & 12 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=2}
$$



And the third, $\Omega[m][,3]$:

$$
\overbrace{
\begin{Bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=1}

\overbrace{
\begin{Bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{Bmatrix}}^{month=2}
$$

Together, these samples suggest there is a 1/3 chance that no violence occurs (the last set of samples), 1/3 chance that the pattern looks as in the first set of sample (call this moderate violence) and a 1/3 chance that the pattern looks worse (call this worse violence) as in the second set of samples.

These samples, as well as marginal forecast samples were analyzed in the file `JointProbabilisticAggregationExample.py`.  

The marginal forecast samples for each grid-cell month can be represented as:

$$
\overbrace{
\begin{Bmatrix}
\{120 &   10 &  0\} & \{83  & 0  & 8\} & \{0 &  0 & 0\} & \{0 & 0 & 0\} \\
 \{ 21 &  23 &  0\} & \{ 66 &  0 & 6\} & \{0 &  0 & 0\} & \{0 & 0 & 0\} \\
 \{  0 &  10 & 21\} & \{ 0  & 5 & 24\} & \{0 &  0 & 0\} & \{0 & 0 & 0\} \\
 \{  0 &  0  &  0\} & \{ 0  & 0 &  0\} & \{0 &  0 & 0\} & \{0 & 0 & 0\} \\
\end{Bmatrix}}^{month=1}
$$
$$
\overbrace{
\begin{Bmatrix}
 \{  0 & 12 & 125\} & \{80  & 10 & 0\} & \{24 & 4 & 0\} & \{0 & 0 & 0\} \\
 \{ 0  & 32 &  45\} & \{11 & 0 &78\} & \{5 & 24 & 0\} & \{0 & 0 & 0\} \\
 \{ 10 &  0 &  2\} & \{12 & 0 & 4\} & \{0  &  0 & 0\} & \{0 & 0 & 0\} \\
 \{ 0  &  0 & 0\} & \{0  &  0 & 0\} & \{0  &  0 & 0\} & \{ 0  &  0 & 0\}
 \end{Bmatrix}}^{month=2}
 $$

Note that each element of the 4 by 4 grids for each of the two months is now a multiset of $S_{1}=3$ potentially repeated values, but these multisets -- because they are marginal pmfs -- do not have any intrinsic order. across instances. 

## The Probability that Each Admin 2 Unit Has a Sum of 25 or More Fatalities

 As a first relatively realistic example, imagine that a stake-holder wanted to know what the probability of 25 or more fatalities was for each of the 4 administrative regions across the two months?

For each value of $a \in \{A1, A2, A3, A4\}$ set $f_{v}=\sum_{j=1}^{S_{1}=3}\mathbb{1}[\sum_{i \in \mathcal{G}[k]} (\hat{y}^{m}_{ij}) \ge 25]\frac{1}{S_{1}|G[k]|}$. 

|     | Joint | Margin |
| --- | ----- | ------ |
| A1  | 2/3   | 3/3    |
| A2  | 2/3   | 3/3    |
| A3  | 1/3   | 1/3    |
| A4  | 0/3   | 0/3    |

Not that in this case the marginal samples bias is positive for A1 and A2.

## The Probability that Each Admin 2 Unit Has a Sum of 0 Fatalities

As a second example, imagine that a different stake-holder wanted to know what the probability of 0 fatalities was, again for each of the 4 administrative regions across the two months?

For each value of $k \in \{A1, A2, A3, A4\}$ set $f_{v}=\sum_{j=1}^{S_{1}=3} \mathbb{1}[(\sum_{i \in \mathcal{G}[k]}\hat{y}^{m}_{ij}) = 0]\frac{1}{S_{1}|G[k]|}$. 

|     | Joint | Margin |
| --- | ----- | ------ |
| A1  | 1/3   | 0/3    |
| A2  | 1/3   | 0/3    |
| A3  | 1/3   | 0/3    |
| A4  | 3/3   | 3/3    |

In this example, the marginal samples bias is negative for A1, A2, and A3. When compared to the prior example, we can highlight that the bias from incorrectly using marginal pmf samples to calculate nonlinear aggregations can be positive, negative, or none depending the covariance structure and the question asked.

## The Probability That A2 or A3 Will Have More Than 50 Fatalities

We can also combine inferences across groupings. As a third example, image that someone wants to know the probability that either A2 or A3 will have more than 50 fatalities.

$f_{v}=\sum_{j=1}^{S_{1}=3} \sum_{k \in \{A2, A3 \}} \mathbb{1}\big[\big(\sum_{i \in \mathcal{G}[k]} \mathbb{1}[\hat{y}^{m}_{ij} \ge 50]\big)>0\big]\frac{1}{S_{1}}$

|          | Joint | Margin |
| -------- | ----- | ------ |
| A2 or A3 | 2/3   | 3/3    |

The marginal distribution misses the covariance structure that that 0's cluster together. The joint distribution keeps this shared shared structure as a sample with many zeroes together within that vector/array. 

## Variance of Sum By Time Point

As a final example, we are often interested in using the variance to calculate standard errors or to summarize the uncertainty across samples. If someone wants to know the estimated variance of the expected sum of fatalities across all grid-cells for each of the two time points, we can define $f_{v} = \sum_{j=1}^{S_{1}=3} \sum_{i \in \mathcal{G}[k]} \big(\hat{y}_{ij}-\hat{\mu}_{j} \big)^{2}\frac{1}{(S_{1}-1)|\mathcal{G}[k]|}$, where $\hat{\mu}_{j} = \sum_{i \in \mathcal{G}[k]}\hat{y}_{ij}\frac{1}{|\mathcal{G}[k]|}$ for $k \in \{M1, M2\}$. Note that I am using the formula for the sample variance here.

|         | Joint | Margin |     |
| ------- | ----- | ------ | --- |
| Month 1 | 28.66 | 31.03  |     |
| Month 2 | 31.5  | 37.87  |     |

In this case the bias in the calculations from the marginal pmf samples is positive, but that does not have to to be the case as it depends on the covariance structure and the aggregation being used.



## Practical Take-aways

1. Block sample for the N observations from parametric/Bayesian, models that rely on bootstrap aggregation (eg random forest), or neural network models that are using dropout.
2. Do not shuffle the 1-D N-length arrays of simulated values across samples as this breaks the covariance structure.
3. If you are forecasting a vector quantity -- say multiple horizons $h \in \{1,\ldots, H\}$ for all N observations -- then your blocks will include both H and N either stacked or in a 2-D array. 
4. Samples are assumed to be independent across draws not across observations


Open Questions
- How many samples are sufficient for different types of questions being answered with joint probabilistic forecasts?
- What is the most efficient way to store joint probabilistic forecasts?
- How can aggregation functions be checked and validated systematically so that mistakes in computation are caught?

