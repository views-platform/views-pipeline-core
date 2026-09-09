## Ensemble Governane Protocol - Protocol for Including Models in Shadow Ensembles

| ADR Info | Details |
|---------|---------|
| **Subject** | Protocol for Including Models in Shadow Ensembles |
| **ADR Number** | 031 |
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


**Reminder**:  We compute all performance metrics on the validation partion. The validation partition is the prediction parallelogram that fits inside the four most recent calendar years -- i.e., the four most recent years for which we have final UCDP GED data.

## Decision
### **Protocol for Including Models in Shadow Ensembles**

Currently, there is one production ensemble **P** and multiple shadow ensembles **$S_i$**. Shadow ensembles are potential successors of the production ensemble and always start out with the current production ensemble as its point of departure. If a production ensemble is replaced by a shadow ensemble, the old production ensemble is relegated to shadow ensemble. The old production ensemble will continue to run in shadow mode for at least 12 months to a) have the option to roll back quickly if the new production ensemble shows irregularities and b) for interesting retrospective evaluation. The new production ensemble becomes the point of departure of all new shadow ensembles. 

As a starting point there should be 3 shadow ensembles. Once the production ensemble is replaced, an effort should be made to replace the 3 shadow ensembles as well.However, this takes time and cannot happen instantly. 

Out of practicality, there should be an upper limit to the number of models an ensemble can contain. However, to introduce variance in shadow ensembles, different shadow ensembles can have different upper limits.

While different projects and stakeholders could potentially receive their own forecasts, there should only be one production ensemble as a starting point. For both point and probabilistic predictions the metrics and criteria to consider for promotion/demotion are defined in ADR 029.

### **Rules**: 

1. Ensure that all input data is part of the current data ingestion protocol. If a feature requires an additional ingestion routine, its inclusion should be justified by a cost–benefit analysis informed by its contribution to the shadow ensemble’s performance.
2. Promote a candidate model to a shadow ensemble **$S_i$** if it improves all metrics by a (metric-specific) threshold $\tau_m$.
3. If the maximum number of models for **$S_i$** is exceeded, roll over all models in the ensemble and remove models that, if removed, the ablated ensemble improves for all metrics by a minimum threshold (metric-specific) $\tau_m$.

Once, a production ensemble is replaced, it will become a shadow ensemble itself for at least a year before it gets rolled out and archived. 


**Different rules for different shadow ensembles:**

- Different maximum number of models
- Different values for $\tau_m$:
    - 0 is acceptable for a non-conservative ensemble
- Possible definition of $\tau_m$ is in terms of standard deviations of the metrics across the 12 (or so) time series/sequences that go into evaluation.

### **Starting Shadow Ensembles**
HH - review and adjust thresholds. 
All shadow models start out with the current production ensemble. When the current production ensemble is replaced by a shadow ensemble this process starts anew. Old shadow and production ensembles are retained as shadow models for 12 months for observation.

In the suggested ensembles below the term **metrics** refers to both the metrics in the **Performance Comparison** and the **Diversity Requirement** for point and probablistic predictions. See ADR 029 for further information.

1. **Shadow Ensemble A – Non-conservative / exploratory**
   - Thresholds: $\tau_m = 0$ for all **metrics** (no deterioration allowed, but no strict improvement required).
   - Size: Fewer models than the production ensemble (at least **two fewer**).

2. **Shadow Ensemble B – Moderate improvement**
   - Thresholds: $\tau_m > 0$ for all **metrics** (each metric must improve by at least a small, metric-specific amount).
   - Size: Number of models **at least as large** as in the production ensemble, but can include **up to 4 more**.

3. **Shadow Ensemble C – High-confidence improvement**
   - Thresholds: $tau_m = k*\sigma_m$ where $\sigma_m$ is the standard deviation of a metric across the evaluation time series, and k > 0. Start with $k=0.1$ 
   - Size: Number of models **at least as large** as in the production ensemble, but can include **up to 4 more**.


## Consequences
### Positive Effects
- Predictable evolution of production models: shadow ensembles provide a structured path toward replacement, avoiding ad-hoc or abrupt changes.
- Encourages innovation without sacrificing quality: multiple shadow ensembles with different strictness levels allow experimentation (non-conservative ensemble) while still maintaining high standards for potential promotion.
- Retention of institutional knowledge: embedding rules in the governance process ensures continuity even when staff or infrastructure changes, preventing loss of expertise.
- Improved robustness across time and data drift: using thresholds and variance-based criteria (e.g., standard-deviation-scaled improvements) avoids decisions driven by noise, making forecast changes more trustworthy in operational settings.


### Negative Effects
- Potential innovation bottleneck: since all shadow ensembles start from the production ensemble, radically different model 
architectures or data strategies may struggle to enter the pipeline or be explored sufficiently.
- Increased procedural complexity: implementing, maintaining, and monitoring multiple shadow ensembles requires additional engineering, experimentation, and documentation effort.
- Maintenance overhead for metric thresholds: metric-specific thresholds (including σ-based definitions) must be calibrated, validated, and sometimes redefined, introducing ongoing tuning work.


## Rationale
This protocol represents a first MVP as the baseline for demotion and promotion guidelines. They are vital for VIEWS as an organization to ensure that changes are traceable, explainable, consistent and reproducible. Especially in the context of an operationalized pipeline, it is vital to have clear rules to set the tone for model development. The proposed metrics, additional health checks and the overall protocol should be reviewed monthly by an expert team to discuss positive and negative effects. These meetings are the basis for future updates of the protocol and its components and will rely heavily on the evaluation reports (both offline and online evaluation) to identify potential unwanted effects. Decisions to update any of the components need to be documented in ADRs. 


## Potential Future Extensions
- For now, we only have one production ensemble and once that production ensemble is replaced, it will continue to run in shadow mode as
 a) a fallback option in case of unexpected issues with the new production ensemble and b) for interesting retrospective evaluation. As a future extension, on could imagine 2 production ensembles: young buck vs old faithful. Young buck contains always the best predictions in terms of performance metrics while old faithful is more consistent over time. The **Rule 3 — Stability Requirement** would differ between the two. For young buck, the above stated 3 months could be enough while for old faithful we would at least require 12 months. This is an interesting addition but would require further discussions. 
- Suggestion by HH: Should we evaluate the shadow ensembles against a calibration partition? This would help a lot to avoid overfitting, but we might get into conflicts with hyper-parameter sweeps etc.

## Additional Notes
While the decision to start shadow ensembles from the production ensemble might restrict creativity, it is a viable solution in the context of severe time constraints. If this should change in the future and developers get ample time to experiment around, this rule could also be updated. 

A separate ADR defines metrics and criteria for promotion/demotion for point and probabilistic predictions.

## Feedback and Suggestions
Feedback welcome!