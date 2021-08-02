import pandas as pd
from clean import clean, remove_dont_knows
from topics import lda, lda_with_keywords
from sentiment import add_sentiment_score
from keywords import add_keywords
from anchored_topic_model import anchored_topic_model

# LDA with keywords

# read data
# data = pd.read_csv('data/feedback.csv', header = None)
# data = clean(data)
# df = lda(data, num_topics=7)
# df.to_csv('output/lda.csv')

# df = lda_with_keywords(df)
# df.to_csv('output/lda_with_keywords.csv')
#
# df = lda(data, num_topics=7, output='features')
# df.to_csv('output/test.csv')


# 3 mix metadata with tokenisation and/or keyword extraction "who mentions what"
# data = pd.read_csv('data/dei_student_all.csv')
# data['Unique Response Number'] = data['Unique Response Number'].astype(str)
#
# df = clean(data, 'Q26')
# df = lda(df, num_topics=10)
# df.rename(columns={
#     "Dominant_topic": "Q26 dominant topic",
#     "Topic_keywords": "Q26 topic keywords"
# }, inplace=True)
# df.drop(columns=['Topic_number', 'Topic1','Topic2','Topic3','Topic4','Topic5','Topic6','Topic7','Topic8','Topic9','Topic10', 'cleaned', 'tokens'], inplace=True)
# df.dropna(subset=['Unique Response Number'], inplace=True)
#
# df2 = clean(data, 'Q18')
# df2 = lda(df2, num_topics=10)
# df2.rename(columns={
#     "Dominant_topic": "Q18 dominant topic",
#     "Topic_keywords": "Q18 topic keywords"
# }, inplace=True)
# df2 = df2[['Unique Response Number','Q18 dominant topic', 'Q18 topic keywords']].copy()
# df2.dropna(subset=['Unique Response Number'], inplace=True)
#
# df['Unique Response Number'] = df['Unique Response Number'].astype(str)
# df2['Unique Response Number'] = df2['Unique Response Number'].astype(str)
# df3 = pd.merge(df, df2, on=['Unique Response Number'], how='inner')
#
# df3.to_csv('output/lda_2.csv')

# 1. Extracting keywords
# ======================
# In this example we just extract keywords and key phrases for each response and add them to the output
# This is useful for exploring the use of particular key phrases in the responses

# data = pd.read_csv('data/dei_student_all.csv')
# data['Unique Response Number'] = data['Unique Response Number'].astype(str)

# always clean data first! The output is called "cleaned" in the result rather than overwrite the original data

# df = clean(data, 'Q26')
# df = add_keywords(df, 'cleaned')
# df.dropna(subset=['Unique Response Number'], inplace=True)
# df.to_csv("output/keywords.csv")

#  2. Using an anchored topic model
#  ================================
#  In this example we start by creating a 'naive' topic model to get an idea of the content.
#  We then apply domain knowledge to create a set of "anchor" terms for the topics, and re-run the model
#  using the anchors as a guide to the algorithm creating the model.
#  Finally we create a topic model for the documents that don't match the resulting model, to
#  see if we need to modify the anchor terms.

data = pd.read_csv('data/dei_student_all.csv')
data['Unique Response Number'] = data['Unique Response Number'].astype(str)

# always clean data first! The output is called "cleaned" in the result rather than overwrite the original data
data = clean(data, 'Q26')
data = remove_dont_knows(data, 'cleaned')

print("\nNon-anchored\n")
df1 = anchored_topic_model(data, 'cleaned', number_of_topics=12, print_topic_details=True)

topic_names = [
    'Library resources',
    'Course materials',
    'Live/in-person',
    'Help & support',
    'Workload',
    'Technology',
    'Wifi and equipment',
    'Communication',
    'Interactivity',
    'Personal & small group learning',
    'Nothing/OK',
    'Organisation']
anchors = [
    ["resources", "library", "books", 'access'],
    ["pre recorded", "recorded", 'record', "pre recorded lectures", "video", "videos","content",'materials','powerpoint','material','accessible', 'subtitles'],
    ["live lectures", "face", "physical", 'on campus', 'in person','live lessons','live','less online'],
    ["help", "support", 'motivate', "guidance", "supportive", "mental health", 'training', 'safe', 'more guidance', 'more information'],
    ["time", 'pressure', "workload", "slow", "overload", "deadlines", "work", "pace", "deadline", 'breaks','reduce','shorter','slower'],
    ["vle", "platform", 'software', 'interface', 'platforms', 'onlinemeetingtool','technology', 'system'],
    ['wifi', "connection", "internet", "data", 'laptops', 'laptop', 'computers', "equipment", 'specialist software', 'specialised software'],
    ['reply','response','explain','respond', 'contact','listen', "emails", "communication", "communicate", "personal tutor", "better communication", 'clear', 'clearer', 'clarity', "feedback",'ask questions'],
    ['participation','involvement','interactive', 'interactivity', 'engage', 'engagement', 'engaging', 'interesting', 'interaction', 'discussion', 'quizzes', 'quiz', 'activity', 'activities'],
    ['personal', 'individual', 'seminars', 'tutorials', 'tutorial', 'group', 'groupwork', 'small groups', 'workshops', 'smaller', 'onetoone'],
    ['happy', 'nothing', 'keep same', 'keep doing', 'good', 'same', 'continue', 'great', 'fine'],
    ['organised', 'organized', 'organisation', 'structure', 'structured', 'planned', 'timetable', 'detailed','manage','schedule', 'consistent']
]
print("\nAnchored\n")
df2 = anchored_topic_model(data, 'cleaned', topic_names=topic_names, anchors=anchors, print_topic_details=True)
df2.to_csv('output/corex.csv')

print("\nUnmatched\n")
unmatched = df2[(df2['Topic label'] == 'No matching topic')].copy()
df2 = anchored_topic_model(unmatched, 'cleaned', number_of_topics=12, print_topic_details=True)

# 3 Sentiment scoring
# In this example we append a sentiment score to several fields
# We've re-labelled the "best thing" and "worst thing" questions to "Positive" and "Negative"
# and then calculate the sentiment score for each answer.

# data = pd.read_csv('data/dei_subset.csv')
# data = clean(data, 'Positive')
# data = add_sentiment_score(data, 'Positive')
# data['positive_score'] = data['sentiment']
# data = clean(data, 'Negative')
# data = add_sentiment_score(data, 'Negative')
# data['negative_score'] = data['sentiment']
# data.to_csv('output/sentiment.csv')