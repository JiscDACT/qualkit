# DACT Qualitative Analyis Toolkit (dact-qat)

This project is a collection of utilities for conducting qualitative
analysis.

It currently consists of the following modules:

* **clean**: a utility for cleaning up text prior to use with other tools
* **sentiment**: a wrapper around SciKit's SentimentIntensityAnalyzer
* **anchored_topic_model**: creates topic models using the Corex algorithm with user-supplied anchors to 'steer' the model using domain knowledge
* **stopwords**: a standard set of stopwords
* **topics**: a wrapper around SciKit's LatentDirichletAllocation
* **keywords**: a wrapper around NLTK's RAKE (Rapid Keyword Extraction) algorithm for finding
keywords in text.
  
For more details on each module, see the 'docs' folder.

## Installing the toolkit and its requirements

Install using:

    pip install dactqal

Or add 'dactqal' to your requirements.txt file, or add as
a dependency in project properties in PyCharm.