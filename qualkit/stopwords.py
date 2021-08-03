import nltk

nltk.download('wordnet')

# initiate stopwords from nltk
stopwords = nltk.corpus.stopwords.words('english')

# add additional missing terms
stopwords.extend(
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'y', 'z', "about", "across", "after", "all", "also", "an", "and", "another", "added",
     "any", "are", "as", "at", "basically", "be", "because", 'become', "been", "before", "being", "between", "both",
     "but", "by", "came", "can", "come", "could", "did", "do", "does", "each", "else", "every", "either", "especially",
     "for", "from", "get", "given", "gets",
     'give', 'gives', "got", "goes", "had", "has", "have", "he", "her", "here", "him", "himself", "his", "how", "if",
     "in", "into", "is", "it", "its", "just", "lands", "like", "make", "making", "made", "many", "may", "me", "might",
     "more", "most", "much", "must", "my", "never", "provide",
     "provides", "perhaps", "no", "now", "of", "on", "only", "or", "other", "our", "out", "over", "re", "said", "same",
     "see", "should", "since", "so", "some", "still", "such", "seeing", "see", "take", "than", "that", "the", "their",
     "them", "then", "there",
     "these", "they", "this", "those", "through", "to", "too", "under", "up", "use", "using", "used", "underway",
     "very", "want", "was", "way", "we", "well", "were", "what", "when", "where", "which", "while", "whilst", "who",
     "will", "with", "would", "you", "your", "i", "i m", "im",
     'ha', 'le', 'u', 'wa',
     'etc', 'via', 'eg', 'e g', 'ie'])

