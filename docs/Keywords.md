# Keyword extraction
It can be useful to look at the keywords in a corpus
of text, for example to look for terms used by particular
groups of users, or to inform other types of analysis.

This module uses the RAKE algorithm to extract keywords,
and then connects them back to the original source data.

## Example use
In this example we just extract keywords and key phrases for each response and add them to the output

This is useful for exploring the use of particular key phrases in the responses.

First, we read in the data and clean it:

    data = pd.read_csv('data/dei_student_all.csv')
    data['Unique Response Number'] = data['Unique Response Number'].astype(str)
    df = clean(data, 'Q26')

Then we add keywords using the cleaned column:

    df = add_keywords(df, 'cleaned')
    df.dropna(subset=['Unique Response Number'], inplace=True)
    df.to_csv("output/keywords.csv")

Note that we have to filter the output to remove 'extra' rows created when
merging the keywords into the dataset.

The output contains a `keywords` column. Note that the output is expanded e.g.:

    Unique Response Number, Q26
    10123, More beer and crisps

Becomes:

    Unique Response Number, Q26, keywords
    10123, More beer and crisps, beer
    10123, More beer and crisps, crisps

So in your analysis, you'll need to use a COUNTD on Unique Response Number
rather than just count rows.