# DACT Qualitative Analyis Toolkit (qualkit)

This project is a collection of utilities for conducting qualitative
analysis.

It currently consists of the following modules:

* **clean**: a utility for cleaning up text prior to use with other tools
* **sentiment**: a wrapper around SciKit's SentimentIntensityAnalyzer
* **anchored_topic_model**: creates topic models using the Corex algorithm (Gallagher et. al., 2017) with user-supplied anchors to 'steer' the model using domain knowledge
* **stopwords**: a standard set of stopwords
* **topics**: a wrapper around SciKit's LatentDirichletAllocation
* **keywords**: a wrapper around NLTK's RAKE (Rapid Keyword Extraction) algorithm for finding
keywords in text.
  
For more details on each module, see the 'docs' folder.

## Installing the toolkit and its requirements

Install using:

    pip install qualkit

Or add 'dactqal' to your requirements.txt file, or add as
a dependency in project properties in PyCharm.

## References

Gallagher, R. J., Reing, K., Kale, D., and Ver Steeg, G. "Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge." Transactions of the Association for Computational Linguistics (TACL), 2017.