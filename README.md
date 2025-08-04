# TWAGA: Thigh-Worn Accelerometer Gait Analysis

Thigh-worn accelerometers are becoming more and more popular in physical activity research because they let researchers monitor how people behave physically in real-world settings. Several large-scale cohort studies in the Prospective Physical Activity, Sitting and Sleep consortium (ProPASS) have now started using thigh-worn accelerometers. Data from these devices can be used to classify different types of activity and postures when they're used with data-driven algorithms. This can help researchers to find out more about how different activities affect our health.

TWAGA represents a new pipeline for analysing human gait using raw data from a single thigh-worn accelerometer. TWAGA combines an activity classification model with subsequent walking speed estimation and gait event detection algorithms. The pipeline is building upon previous research results by myself [[Lendt et al. 2024](https://doi.org/10.1186/s12966-024-01646-y)], Loubna Baroudi [[Baroudi et al. 2020](https://doi.org/10.3389/fspor.2020.583848)] and Robbin Romijnders [[Romijnders et al. 2023](https://doi.org/10.3389/fneur.2023.1247532)].

The developed pipeline involves several key steps:

1. **Pre-processing**: We resample and filter the raw acceleration signal to a common sampling frequency.
2. **Walking sequence detection**: An activity classification model identifies walking sequences based on fixed time-intervals (4 seconds) of acceleration data. Additional smoothing of the classification labels is performed to reduce false negative predictions.
3. **Walking speed estimation:** The walking speed is estimated by exploiting the relationship between stride frequency and walking speed. Stride frequency is estimated using an autocorrelation approach.
4. **Gait event prediction**: A second model predicts the probability for gait events for each timepoint within each walking sequence. Gait events are identified based on the probability distributions. Identified gait events are filtered using heuristic approaches.
5. **Gait analysis**: Gait-specific parameters (e.g. stride time, walking speed, duty factor) are calculated for each stride and then summarised for each walking sequence.

![Fig1](figures/pipeline.jpg)




## 💻 Code

⌛ **We are currently cleaning, documenting and testing a packaged Python module. Stay tuned for the code!**



## 📄 Papers

⌛ **A preprint with more details and validation results will be published in August 2025.**

Some information regarding the gait event detection and activity type classification can already be found here:

Lendt, Claas and Stewart, Tom (2024) "Gait Event Detection During Walking Using Deep Learning and Thigh-Worn Accelerometry", *ISBS Proceedings Archive*: Vol. 42: Iss. 1, Article 165. Available at: https://commons.nmu.edu/isbs/vol42/iss1/165

Lendt, C., Hansen, N., Froböse, I. et al. Composite activity type and stride-specific energy expenditure estimation model for thigh-worn accelerometry. *Int J Behav Nutr Phys Act* 21, 99 (2024). https://doi.org/10.1186/s12966-024-01646-y



## 🔓 Open data

This pipeline has been developed and validated using several open datasets:

* HARTH and HAR70+ [[GitHub](https://github.com/ntnu-ai-lab/harth-ml-experiments)]
* WearGait-PD [[Synapse](https://www.synapse.org/Synapse:syn52540892/wiki/623751)]

The gait data collected as part of this work will be openly published very soon.



## 💸 Funding

TWAGA is partially based upon the results of the research project 'Estimation of activity induced energy expenditure using thigh-worn accelerometry and machine learning approaches', funded by the Internal Research Funds of the German Sport University Cologne. Some of the underlying work has been done while being supported by a fellowship of the German Academic Exchange Service (DAAD).



## 🙌🏽 Acknowledgements

We appreciate all research teams who made the extra effort to openly share their datasets as well as all participants in all underlying studies.
