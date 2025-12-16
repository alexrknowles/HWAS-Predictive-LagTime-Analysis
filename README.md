# HWAS-Predictive-LagTime-Analysis
Python time-series validation of a legacy meteorological forecasting model (H-WAS), utilizing complex triple-filtered conditional logic to precisely quantify predictive lead time (lag time) for high-wind events.
H-WAS Predictive Model Re-Validation and Lag Time Analysis
Project Overview
This project involves the rigorous re-validation of the High Wind Alert System (H-WAS) Rule of Thumb, a specific set of meteorological conditions developed in 1993 to predict severe downslope wind events. The goal of this analysis is to use modern Python data science tools to precisely quantify the predictive lead time (lag time) and success rate of the H-WAS rule when all necessary meteorological criteria are met simultaneously.

The core challenge was transforming the analytical approach from a post-event investigation (triggered by wind hits) to a predictive, condition-triggered analysis, directly addressing the question: When the atmosphere is ready, how long is the warning window before impact?

Methodology: The Triple-Filtered Trigger
The analysis is structured around a Triple-Filtered Trigger event, requiring all three critical H-WAS conditions to be satisfied at the high-elevation station (Rampart Range) at the exact moment the event is considered active.

| Condition | Threshold | Role |
| :--- | :--- | :--- |
| **Instability** ($\Delta T$) | $\mathbf{\geq 11^\circ\text{F}}$ (Lower $T$ - Rampart $T$) | The atmosphere is unstable (the "door is open"). |
| **Force Direction** | Rampart Direction $\mathbf{230.0^\circ}$ to $\mathbf{310.0^\circ}$ (West Wind) | The flow is aligned for mountain wave descent. |
| **Penetration Force** | Rampart Speed $\mathbf{\geq 35 \text{ KT}}$ | The force required to push the wind through the boundary layer is present. |

### Lag Time Measurement

The primary metric quantified is the **Lag Time**, which represents the predictive warning window available to forecasters. It is measured in minutes:

$$\text{Lag Time} = (\text{Time Lower Station Hits } 35 \text{ KT}) - (\text{Time Triple Filter Conditions Are Met})$$

A translation is counted as a **success** if the lower station hits $\mathbf{\geq 35 \text{ KT}}$ within a **6-hour search window** following the trigger time, consistent with the original 1993 study's verification period.

## Key Technical Features

* **Time Series Alignment & Cleaning:** Robust loading and preprocessing of multi-station time-series CSV data using Pandas, including precise index alignment and handling of duplicate entries.
* **Restructured Analysis Logic:** Developed a Python function (`run_analysis_delta_t_trigger`) that iterates through the data based on the simultaneous *condition* trigger time rather than the wind-event time, allowing for the reliable measurement of predictive lead time.
* **Conditional Filtering:** Implementation of complex boolean logic (`delta_t_hit & west_wind_at_trigger & strong_wind_at_trigger`) to isolate only the events where all three H-WAS criteria were simultaneously met.
* **Statistical Validation:** Calculation of key performance indicators, including overall success rate, per-station success rate, and mean lag time, to provide quantitative evidence of the model's performance.
* **Visualization:** Generated customized Matplotlib scatter plots to visualize the relationship between lag time and key meteorological variables (e.g., Rampart Wind Speed, Lower Station Temperature) at the moment of the predictive trigger.

## Dependencies

The project relies on standard Python data science libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `os`
* `datetime`
