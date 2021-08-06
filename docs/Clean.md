# Clean

The clean module contains utility classes for cleaning up text
ready for use with other techniques.

## Functions

### clean(DataFrame, column, inplace=False)
The basic clean method takes as input a pandas DataFrame and a 
single named column and applies several cleaning steps:

* change case to lowercase
* remove apostrophes
* replace punctuation with spaces
* remove duplicate whitespace, and spaces at the start and end of a string
* replace domain-specific synonyms with their generic terms

(The last step is currently hard-coded for some specific terms
such as the names of HE products, but in future this could be 
user-supplied)

If the result is an empty string, the row is removed.

Returns a copy of the DataFrame.

### replace_domain_terms(text, domain_terms, replacement)
Replaces any matches for the list of domain terms with the 
replacement term.

### remove_dont_knows(DataFrame, column_name)
Removes any rows from the DataFrame where the specified column
only contains 'don't know' or one of its synonyms.

Returns a copy of the DataFrame.

### lemmatize(DataFrame, column_name)
Lemmatizes all the text in the specified column of the DataFrame.

Uses the NLTK WordNetLemmatizer in conjunction with the 
Penn Treebank part of speech model.

Returns a copy of the DataFrame.