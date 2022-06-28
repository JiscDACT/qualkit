# Clean

The clean module contains utility classes for cleaning up text
ready for use with other techniques.

## Functions

### clean(DataFrame, columns, inplace=False)
The basic clean method takes as input a pandas DataFrame and a 
one or more column names (string or list) and applies several 
cleaning steps:

* change case to lowercase
* remove apostrophes
* replace punctuation with spaces
* remove duplicate whitespace
* remove spaces at the start and end of a string
* replace domain-specific synonyms with their generic terms

(The last step is currently hard-coded for some specific terms
such as the names of HE products, but in future this could be 
user-supplied)

If the result is an empty string, it is replaced with np.nan (Null)

If inplace is False, then the result is returned in a new column
called 'cleaned' ('cleaned_yourcolumnname' for each subsequent
column if there is more than one column being cleaned). If inplace
is True then the original content is overwritten.

Returns a copy of the DataFrame.

### clean_without_domain(data, columns, inplace=False)

As above but does not perform the replace domain specific synonym 
function as sometimes this information needs to be retained. 
For example, when asked which app has been most useful.

### replace_domain_terms(text, domain_terms, replacement)
Replaces any matches for the list of domain terms with the 
replacement term.

### remove_dont_knows(DataFrame, columns)
Removes any rows from the DataFrame where the specified column
only contains 'don't know' or one of its synonyms.

Returns a copy of the DataFrame.

### lemmatize(DataFrame, columns)
Lemmatizes all the text in the specified column(s) of the DataFrame.

Uses the NLTK WordNetLemmatizer in conjunction with the 
Penn Treebank part of speech model.

Returns a copy of the DataFrame.