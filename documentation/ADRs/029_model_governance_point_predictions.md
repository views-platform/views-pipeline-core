## Ensemble Governance Protocol - Point Predictions

| ADR Info | Details |
|---------|---------|
| **Subject** | Governance protocol for point predictions |
| **ADR Number** | 029 |
| **Status** | Proposed |
| **Author** | [Sonja, Simon] |
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
A shadow ensemble **$S_i$** may replace the production ensemble **P** only if it satisfies all governance rules below (both for pgm and cm):

### **Rule 1 — Performance Comparison (MSLE, $\hat{\bar{y}}$, MSE)**

Ensemble selection follows this ordered procedure; taking into account MSLE, $\hat{\bar{y}}$ and the MSE:

1. **Start with MSLE**  
   - If the ensemble with lower MSLE also predicts higher fatalities on average ($\hat{\bar{y}}$) - less conservative - , select it.

2. **Compare MSE:**  
   - If the ensemble with a lower MSLE also has a lower MSE → select it.  
   - If the ensemble with a higher MSLE has lower MSE *and* higher average predictions ($\hat{\bar{y}}$) → select the ensemble with lower MSE.

### **Rule 2 — Diversity Requirement**

A new ensemble must not reduce diversity. The shadow ensemble **$S_i$** is at least as diverse as the production ensemble **P**.

Page’s ensemble diversity score:


$D_e = \frac{1}{n} \sum_{i=1}^{N} \left( \frac{1}{M} \sum_{m=1}^{M} (p_i^m - p_i^e)^2 \right)$


Diversity should be calculated in the non-log form (confirm with HH). The diversity score is added as a rule because it offers additional protection against overfitting on the validation data. 

### **Rule 3 — Stability Requirement**

A shadow ensemble **$S_i$** must:

- Run for **≥ 3 consecutive months** and outperform **P** in the **Performance Comparison** and the **Diversity Requirement**
- Show **no irregularities**  
- Produce correct forecast files


### **Notes on Metrics**

For all the decision rules that compare metrics (MSLE, MSE, $\hat{\bar{y}}$, diversity), someone (HH or similar) needs to define an epsilon (`eps`) for each comparison such that the promotion/demotion is meaningful (e.g. we do not promote based on changes in the 6th decimal). The following questions have to be answered:
- How much better on **MSLE** does $S_i$ need to be before we treat it as meaningfully better?
- How much better on **MSE**?
- How large a difference in average prediction $\hat{\bar{y}}$ counts as “less conservative” in a meaningful way?
- Same for diversity, if relevant.


### **Ensemble Performance Monitoring (Governance & Expert Review) **

An expert team consisting of 2-5 researchers will examine the ensemble runs bi-monthly (confirm with Simon and HH) and compare shadow ensembles to the production ensemble. They will adhere to the rules defined the ADRs regarding Ensemble Governance Protocols and consult the evaluation reports - both offline and online evaluation - to take a decision on if to promote or demote ensembles. They will also examine and discuss the effects of the defined rules and suggest changes if unanticipated/undesirable patterns emerge. 

When evaluating shadow ensembles, VIEWS also takes aggregates into account:

- Our ability to predict correctly in the aggregate (e.g. MSE across all time steps) for a country or PG cell, and for geographical aggregations including global total, EMD and pEMDiv at both cm and pgm.
- The distributions of predictions in comparison to the distribution of historical actuals. 

Both of these should be added to the evaluation reports!


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
One could also imagine 2 production ensembles: young buck vs old faithful. Young buck contains always the best predictions in terms of performance metrics while old faithful is more consistent over time. The **Rule 3 — Stability Requirement** would differ between the two. For young buck, the above stated 3 months could be enough while for old faithful we would at least require 12 months. This is an interesting addition but would require further discussions. 

## Additional Considerations
While the decision to start shadow ensembles from the production ensemble might restrict creativity, it is a viable solution in the context of severe time constraints. If this should change in the future and developers get ample time to experiment around, this rule could also be updated. 

We also discussed the inclusion of correctly predicting onsets as a potential addition to the Governance Protocol, but decided against it due to the difficulties in operationalizing what exactly constitutes an onset. The reader is referred to Simon if in doubt.

A separate ADR defines governance for probabilistic/uncertainty forecast ensembles and the procedure to include models in ensembles.

## Feedback and Suggestions
Feedback welcome!



## Helper: Metric Comparison with Tolerance

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