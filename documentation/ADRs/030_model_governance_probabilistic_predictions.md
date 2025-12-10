## Ensemble Governance Protocol - Probabilistic Predictions

| ADR Info | Details |
|---------|---------|
| **Subject** | Governance protocol for probabilistic predictions |
| **ADR Number** | 030 |
| **Status** | Proposed |
| **Author** | [Sonja, Simon] |
| **Date** | [25.11.2025] |


## Context
The VIEWS forecasting system currently has one production ensemble that generates point predictions, but none that produces probabilistic forecasts. For different projects, stakeholders requested probabilistic forecasts, hence we need rules to create a production ensemble. The rationale for developing the guidelines is the same as for point forecasts (see ADR 029)

## Decision
A VIEWS production ensemble P (with forecasts both at cm and pgm levels) will be replaced by a new shadow ensemble S if the following conditions hold:

### **Rule 1 — Performance Comparison (CRPS, MIS, average $\hat{\bar{y}}$)**

1. The shadow ensemble S is chosen both at the cm and pgm levels if the following hold:
- CRPS for S is lower than for P
- MIS for S is at least equal or lower than for P (to do: HH wanted to talk with Mike regarding if the second metric should be MIS or Log Score)

Suggestion (HH and Simon?):
- Transfer $\hat{\bar{y}}$ from point predictions to sample predictions: average $\hat{\bar{y}}$ 
- The average $\hat{\bar{y}}$ for S is at least as high as for P (i.e. S is not more conservative than P).

### **Rule 2 — Stability Requirement**

A shadow ensemble must:

- Run for **≥ 3 consecutive months**  
- Show **no irregularities**  
- Produce correct forecast files

### **Rule 3 — Diversity Requirement (Proposed Revision)**

A new ensemble must not reduce diversity. The shadow ensemble S is at least as diverse as the production ensemble P.
- The diversity of P is at least equal or lower than that of S (TBD define the diversity score)

Score: TBD

- Diversity should be calculated in the non-log form (confirm with HH). The diversity score is added as a rule because it offers additional protection against overfitting on the validation data. 

### **Notes on Metrics**

For all the decision rules that compare metrics (CRPS, MIS, mean $\hat{\bar{y}}$, diversity), someone (HH or similar) needs to define an epsilon (`eps`) for each comparison such that the promotion/demotion is meaningful (e.g. we do not promote based on changes in the 6th decimal). The following questions have to be answered:
- How much better on **CRPS** does $S_i$ need to be before we treat it as meaningfully better?
- How much better on **MIS**?
- How large a difference in average prediction $\hat{\bar{y}}$ counts as “less conservative” in a meaningful way?
- Same for diversity, if relevant.

### **Ensemble Performance Monitoring (Governance & Expert Review) **

An expert team consisting of 2-5 researchers will examine the ensemble runs bi-monthly (confirm with Simon and HH) and compare shadow ensembles to the production ensemble. They will adhere to the rules defined the ADRs regarding Ensemble Governance Protocols and consult the evaluation reports - both offline and online evaluation - to take a decision on if to promote or demote ensembles. They will also examine and discuss the effects of the defined rules and suggest changes if unanticipated/undesirable patterns emerge. 

When evaluating shadow ensembles, VIEWS also takes aggregates into account:

- Our ability to predict correctly in the aggregate for a country or PG cell, and for geographical aggregations including global total, EMD and pEMDiv at both cm and pgm.
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

While the decision to start shadow ensembles from the production ensemble might restrict creativity, it is a viable solution in the context of severe time constraints. If this should change in the future and developers get ample time to experiment around, this rule could also be updated. 

## Additional Notes
A separate ADR defines governance for point prediction ensembles.

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