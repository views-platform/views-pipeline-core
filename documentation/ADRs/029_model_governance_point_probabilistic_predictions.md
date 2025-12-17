## Ensemble Governance Protocol - Point & Probabilistic Predictions

| ADR Info | Details |
|---------|---------|
| **Subject** | Governance protocol for point & probabilistic predictions |
| **ADR Number** | 029 |
| **Status** | Proposed |
| **Author** | [Sonja, Simon, Håvard] |
| **Date** | [25.11.2025] |


## Context
The VIEWS forecasting system currently relies on one production ensemble to generate its point predictions. These predictions are distributed to stakeholders and partners through both our API and the public dashboard. The current production ensemble has been in place for several years.

As newer modeling approaches become available, and because VIEWS aims to consistently deliver high-quality forecasts, we need clear guidelines that encourage the development of potential replacement ensembles. A clear governance protocol is necessary for the following reasons: 1. It ensures that changes are traceable, explainable, consistent and reproducible. 2. Without governance, introducing a new ensemble carries a risk of degraded performance, conceptual drift, or unanticipated behaviour in high-stakes environments. The protocol reduces the likelihood of large errors or unstable forecasts reaching stakeholders. 3. As staff, infrastructure, and models evolve over time, governance ensures that institutional knowledge is embedded in a documented process rather than held implicitly by individuals. This reflects state-of-the-art MLOps principles. 

The purpose of these rules is to ensure that the production ensemble continues to produce forecasts that:

1. Maximize performance in terms of the evaluation metrics defined below. 

2. Ensure decisions are robust to random variation and drift in the validation data (i.e. not driven by tiny, unstable differences).

3. Retain the strengths of our existing models while supporting stable, incremental innovation.

4. Mitigate systematic under-prediction in a consistent and reliable manner.


## Decision

All calculations of performance metrics are done on the validation partition. 
**Definition of the validation partition:** the prediction parallelogram that fits inside the four most recent calendar years -- i.e., the four most recent years for which we have final UCDP GED data. This results in 12 evaluation sequences.

In the following, the metrics for point and probabilistic predictions are defined. 

## Metrics for Point Predictions

A VIEWS production ensemble **P** (with forecasts both at cm and pgm levels) will be replaced by a new shadow ensemble **$S_i$** if the following conditions hold:

### **Rule 1 — Performance Comparison (MSLE, $\hat{\bar{y}}$, MSE)**

- Look at MSLE first, if $\hat{\bar{y}}$ is higher for the ensemble model better ranked by MSLE, choose the
ensemble model ranked best by MSLE.
- If $\hat{\bar{y}}$ is lower for the ensemble model ranked higher by MSLE, then look at MSE.
- If MSE also chooses the model ranked higher by MSLE, then choose that model.
- If the MSE is better (lower) for the model that was worse for MSLE but had a higher $\hat{\bar{y}}$, then
choose the model with the better MSE.

### **Rule 2 — Diversity Requirement**

A new ensemble must not reduce diversity. To replace the production ensemble **P**, the shadow ensemble $S_i$ has to be as least as diverse as the production ensemble **P** in terms of Page’s ensemble diversity score:

Page’s ensemble diversity score:


$D_e = \frac{1}{n} \sum_{i=1}^{N} \left( \frac{1}{M} \sum_{m=1}^{M} (p_i^m - p_i^e)^2 \right)$


Diversity should be calculated in the non-log form. The diversity score is added as a rule because it offers additional protection against overfitting on the validation data. 

### **Rule 3 — Stability Requirement**

A shadow ensemble **$S_i$** must:

- Show **no irregularities** and produce correct forecast files for *≥ 2 consecutive months**

As soon as Online Evaluation is implemented: 
- Run for **≥ 3 consecutive months** and outperform **P** in the **Performance Comparison** and the **Diversity Requirement**


## Metrics for Probabilistic Predictions
A VIEWS production ensemble **P** (with forecasts both at cm and pgm levels) will be replaced by a new shadow ensemble $S_i$ if the following conditions hold:

### **Rule 1 — Performance Comparison (CRPS, MIS, average $\hat{\bar{y}}$)**

- CRPS for $S_i$ is lower than for **P**
- Log Score for $S_i$ is at least equal or lower than for **P** 

**Suggestion**:
We do not want to reward conservative models.
- Compare the aggregate prediction distribution of the new shaddow ensemble $S_i$ a) to the one of **P** and b) to the aggregate observed distributions. Compute distance metrics that put special emphasis on the right hand tail.
- Suggested metrics: Kolmogorov–Smirnov (KS) Test and Wasserstein Distance on the whole distribution and the upper tail. Again, we need to define a treshold for the upper tail. 

### **Rule 2 — Diversity Requirement (Proposed Revision)**

A new ensemble must not reduce diversity. The shadow ensemble S is at least as diverse as the production ensemble P.
- The diversity of P is at least equal or lower than that of S (TBD define the diversity score)

Score: TBD

- Diversity should be calculated in the non-log form (confirm with HH). The diversity score is added as a rule because it offers additional protection against overfitting on the validation data.

### **Rule 3 — Stability Requirement**

A shadow ensemble **$S_i$** must:

- Show **no irregularities** and produce correct forecast files for *≥ 2 consecutive months**

As soon as Online Evaluation is implemented: 
- Run for **≥ 3 consecutive months** and outperform **P** in the **Performance Comparison** and the **Diversity Requirement**



### **Notes on Metrics**

For all the decision rules that compare metrics (point predictions: MSLE, MSE, $\hat{\bar{y}}$, diversity; probabilitic predictions: CRPS, Log-Score, diversity), someone (HH or similar) needs to define an epsilon (`eps`) for each comparison such that the promotion/demotion is meaningful (e.g. we do not promote based on changes in the 6th decimal). The following questions have to be answered:
- How much better on **MSLE/CRPS** does $S_i$ need to be before we treat it as meaningfully better?
- How much better on **MSE/Log-Score**?
- How large a difference in average prediction $\hat{\bar{y}}$ (and equivalent for probabilistic predictions) counts as “less conservative” in a meaningful way?
- Same for diversity, if relevant.


### **Ensemble Performance Monitoring (Governance & Expert Review) **

An expert team consisting of 2-5 researchers will examine the ensemble runs bi-monthly (confirm with Simon and HH) and compare shadow ensembles to the production ensemble. They will adhere to the rules defined the ADRs regarding Ensemble Governance Protocols and consult the evaluation reports - both offline and online evaluation - to take a decision on if to promote or demote ensembles. They will also examine and discuss the effects of the defined rules and suggest changes if unanticipated/undesirable patterns emerge. 

When evaluating shadow ensembles, VIEWS also monitors aggreagtes:

- Our ability to predict correctly in the aggregate (e.g. MSE across all time steps) for a country or PG cell, and for geographical aggregations including global total, EMD and pEMDiv at both cm and pgm.
- The distributions of predictions in comparison to the distribution of historical actuals. 

Both of these should be added to the evaluation reports!

### General Notes
If a production ensemble is replaced by a shadow ensemble, the old production ensemble is relegated to shadow ensemble. The new production ensemble becomes the point of departure of all new shadow ensembles.


## Consequences
### Positive Effects
- Ensures new ensembles outperform existing ones across multiple dimensions  
- Reduces overfitting risk via diversity requirement  
- Produces more reliable point forecasts  
- Ensures monitoring of high-level behavior (aggregation)
- clear guidelines should simplify ensemble development

### Negative Effects
- Slower promotion of new ensembles  
- Requires continuous monitoring  
- More complex evaluation procedure
- New ensembles always start from the production ensemble, potentially slowing down innovation


## Rationale
This protocol represents a first MVP as the baseline for demotion and promotion guidelines. They are vital for VIEWS as an organization to ensure that changes are traceable, explainable, consistent and reproducible. Especially in the context of an operationalized pipeline, it is vital to have clear rules to set the tone for model development. The proposed metrics, additional health checks and the overall protocol should be reviewed monthly by an expert team to discuss positive and negative effects. These meetings are the basis for future updates of the protocol and its components and will rely heavily on the evaluation reports (both offline and online evaluation) to identify potential unwanted effects. Decisions to update any of the components need to be documented in ADRs. 

## Potential Future Extensions
- For now, we only have one production ensemble and once that production ensemble is replaced, it will continue to run in shadow mode as
 a) a fallback option in case of unexpected issues with the new production ensemble and b) for interesting retrospective evaluation. As a future extension, one could also imagine 2 production ensembles: young buck vs old faithful. Young buck contains always the best predictions in terms of performance metrics while old faithful is more consistent over time. The **Rule 3 — Stability Requirement** would differ between the two. For young buck, the above stated 3 months could be enough while for old faithful we would at least require 12 months. This is an interesting addition but would require further discussions. 
- Defining thresholds a-priori is difficult without a-priori experience. A solution would be to leave the metric-specific threshold 
 question open and come back to this question after a couple of iterations and bi-monthly sessions of our expert term.
- Currently, the validation persiod consists of 12 evaluation sequences. HH suggested to increase the validation partition size to 36 
 sequences (six years) to obtain more stable evaluation metrics. 

## Additional Considerations
While the decision to start shadow ensembles from the production ensemble might restrict creativity, it is a viable solution in the context of severe time constraints. If this should change in the future and developers get ample time to experiment around, this rule could also be updated. 

We also discussed the inclusion of correctly predicting onsets as a potential addition to the Governance Protocol, but decided against it due to the difficulties in operationalizing what exactly constitutes an onset. The reader is referred to Simon if in doubt.

A separate ADR defines governance for probabilistic/uncertainty forecast ensembles and the procedure to include models in ensembles.

## Feedback and Suggestions
Feedback welcome!



## Helper for Point Predictions: Metric Comparison with Tolerance

This section contains pseudocode to illustrate the rules outlined above.

direction: "lower_is_better" or "higher_is_better"
returns: "better", "similar", or "worse"

```
function compare_metric(new_value, old_value, direction, eps):

    if direction == "lower_is_better":
        diff = old_value - new_value    # positive diff → new is better
    else:  # "higher_is_better"
        diff = new_value - old_value    # positive diff → new is better

    if diff > eps:
        return "better"
    elif diff < -eps:
        return "worse"
    else:
        return "similar"
```

### Rule 1 — Performance Comparison
#### (MSLE → MSE, with conservativeness)

```
function performance_decision(S_i, P, eps_msle, eps_mse, eps_avg):

    msle_comp = compare_metric(MSLE(S_i), MSLE(P), "lower_is_better", eps_msle)
    mse_comp  = compare_metric(MSE(S_i),  MSE(P), "lower_is_better", eps_mse)
    avg_comp  = compare_metric(avg_pred(S_i), avg_pred(P), "higher_is_better", eps_avg)

    # Step 1 — clear MSLE win
    if msle_comp == "better":

        # Prefer models that are not more conservative
        if avg_comp in ["better", "similar"]:
            return Promote

        # If more conservative, require better MSE to compensate
        if mse_comp == "better":
            return Promote
        else:
            return Keep

    # Step 2 — MSLE similar: use MSE + conservativeness
    if msle_comp == "similar":

        if mse_comp == "better" and avg_comp in ["better", "similar"]:
            return Promote
        else:
            return Keep

    # Step 3 — MSLE worse: very restrictive path
    if msle_comp == "worse":

        # Optional: allow promotion only if big overall gain + less conservative
        if mse_comp == "better" and avg_comp == "better":
            return Promote
        else:
            return Keep
```



### Rule 2 — Diversity Requirement

```
function diversity_decision(S_i, P, eps_div):

    div_comp = compare_metric(diversity(S_i), diversity(P), "higher_is_better", eps_div)

    if div_comp in ["better", "similar"]:
        return Pass
    else:
        return Fail
    # Note: diversity computed in non-log space (to be confirmed).

```


### Rule 3 — Stability Requirement

```
function stability_decision(S_i):

    if S_i.passed_rules_1_and_2_for >= 3_consecutive_months
       and S_i.has_no_irregularities
       and S_i.output_files_are_valid:
        return Stable
    else:
        return Unstable
```



### Governance & Expert Review

```
function governance_decision(S_i, P, eps_msle, eps_mse, eps_avg, eps_div):

    perf_result  = performance_decision(S_i, P, eps_msle, eps_mse, eps_avg)
    div_result   = diversity_decision(S_i, P, eps_div)
    stab_result  = stability_decision(S_i)

    # Must pass all automated checks first
    if perf_result == Promote
       and div_result == Pass
       and stab_result == Stable:

        # Final human oversight
        expert_team.review(S_i, P)

        if expert_team.approves:
            return Promote
        else:
            return Keep

    else:
        return Keep
```


### Final Operational Protocol

```
function promotion_protocol(S_i, P):

    # Example: choose epsilons as small relative improvements
    eps_msle = 0.01 * MSLE(P)         # 1% MSLE improvement threshold
    eps_mse  = 0.01 * MSE(P)          # 1% MSE improvement threshold
    eps_avg  = 0.01 * avg_pred(P)     # 1% avg prediction shift
    eps_div  = 0.01 * diversity(P)    # 1% diversity change

    decision = governance_decision(S_i, P, eps_msle, eps_mse, eps_avg, eps_div)

    if decision == Promote:
        promote(S_i)   # Make S_i the new production ensemble
    else:
        keep(P)        # Keep current production ensemble
```


## Helper for Probabilistic Predictions: Metric Comparison with Tolerance

This section contains pseudocode to illustrate the rules outlined above.

direction: "lower_is_better" or "higher_is_better"
returns: "better", "similar", or "worse"

```
function compare_metric(new_value, old_value, direction, eps):

    if direction == "lower_is_better":
        diff = old_value - new_value    # positive diff → new is better
    else:  # "higher_is_better"
        diff = new_value - old_value    # positive diff → new is better

    if diff > eps:
        return "better"
    elif diff < -eps:
        return "worse"
    else:
        return "similar"
```

### Rule 1 — Performance Comparison
#### (CRPS → MIS, with conservativeness)

```
function performance_decision(S_i, P, eps_crps, eps_mis, eps_avg):

    crps_comp = compare_metric(CRPS(S_i), CRPS(P), "lower_is_better", eps_crps)
    mis_comp  = compare_metric(MIS(S_i),  MIS(P), "lower_is_better", eps_mis)
    avg_comp  = compare_metric(avg_pred(S_i), avg_pred(P), "higher_is_better", eps_avg)

    # Step 1 — clear CRPS win
    if crps_comp == "better":

        # Prefer models that are not more conservative
        if avg_comp in ["better", "similar"]:
            return Promote

        # If more conservative, require better MIS to compensate
        if mis_comp == "better":
            return Promote
        else:
            return Keep

    # Step 2 — CRPS similar: use MIS + conservativeness
    if crps_comp == "similar":

        if mis_comp == "better" and avg_comp in ["better", "similar"]:
            return Promote
        else:
            return Keep

    # Step 3 — CRPS worse: very restrictive path
    if crps_comp == "worse":

        # Optional: allow promotion only if big overall gain + less conservative
        if mis_comp == "better" and avg_comp == "better":
            return Promote
        else:
            return Keep
```



### Rule 2 — Diversity Requirement

```
function diversity_decision(S_i, P, eps_div):

    div_comp = compare_metric(diversity(S_i), diversity(P), "higher_is_better", eps_div)

    if div_comp in ["better", "similar"]:
        return Pass
    else:
        return Fail
    # Note: diversity computed in non-log space (to be confirmed).

```


### Rule 3 — Stability Requirement

```
function stability_decision(S_i):

    if S_i.passed_rules_1_and_2_for >= 3_consecutive_months
       and S_i.has_no_irregularities
       and S_i.output_files_are_valid:
        return Stable
    else:
        return Unstable
```



### Governance & Expert Review

```
function governance_decision(S_i, P, eps_crps, eps_mis, eps_avg, eps_div):

    perf_result  = performance_decision(S_i, P, eps_crps, eps_mis, eps_avg)
    div_result   = diversity_decision(S_i, P, eps_div)
    stab_result  = stability_decision(S_i)

    # Must pass all automated checks first
    if perf_result == Promote
       and div_result == Pass
       and stab_result == Stable:

        # Final human oversight
        expert_team.review(S_i, P)

        if expert_team.approves:
            return Promote
        else:
            return Keep

    else:
        return Keep
```


### Final Operational Protocol

```
function promotion_protocol(S_i, P):

    # Example: choose epsilons as small relative improvements
    eps_crps = 0.01 * CRPS(P)         # 1% CRPS improvement threshold
    eps_mis  = 0.01 * MIS(P)          # 1% MIS improvement threshold
    eps_avg  = 0.01 * avg_pred(P)     # 1% avg prediction shift
    eps_div  = 0.01 * diversity(P)    # 1% diversity change

    decision = governance_decision(S_i, P, eps_crps, eps_mis, eps_avg, eps_div)

    if decision == Promote:
        promote(S_i)   # Make S_i the new production ensemble
    else:
        keep(P)        # Keep current production ensemble
```