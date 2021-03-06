import pandas as pd

from qualkit.clean import clean, remove_dont_knows, lemmatize
from qualkit.anchored_topic_model import anchored_topic_model, topic_metrics

#  Using an anchored topic model
#  =============================
#  In this example we start by creating a 'naive' topic model to get an idea of the content.
#  We then apply domain knowledge to create a set of "anchor" terms for the topics, and re-run the model
#  using the anchors as a guide to the algorithm creating the model.
#  Finally we create a topic model for the documents that don't match the resulting model, to
#  see if we need to modify the anchor terms.

data = pd.read_csv('data/DEI_TS_2021.csv')
# data['Unique Response Number'] = data['Unique Response Number'].astype(str)

# always clean data first! The output is called "cleaned" in the result rather than overwrite the original data
data = clean(data, 'Q11')
data = remove_dont_knows(data, 'cleaned')
data = lemmatize(data, 'cleaned')

topic_metrics(data,'cleaned', number_of_topics=20)

#
# Create a topic model with no anchor topics as a baseline
#
print("\nNon-anchored\n")
df1 = anchored_topic_model(data, 'cleaned', number_of_topics=20, print_topic_details=True)

#
# Create an anchored topic model with suggested topics anchor terms
#
topic_names = [
    'Library resources',
    'Course materials',
    'Live/in-person',
    'Help & support',
    'Information & guidance',
    'Workload',
    'Technology',
    'Wifi and equipment',
    'Communication',
    'Interactivity',
    'Personal & small group learning',
    'Nothing/OK',
    'Organisation',
    'Assessment',
    'Teaching',
    'Online learning',
    'Accessibility',
    'Don\'t knows',
    'Digital skills'
]
anchors = [
    ["resource", "library", "book", 'access', 'textbook', 'ebooks', 'reading', 'list'],
    ["pre", "prerecord", "recorded", 'recording', 'record', "prerecorded", "video", "content", 'powerpoint', 'material', 'powerpoints'],
    ["live", "face", "physical", 'campus', 'person'],
    ["help", "support", 'advice', 'motivate', "supportive", "mental", "health", 'safe', 'safety', 'assistance', 'awareness', 'outreach', 'check in'],
    ['guidance', 'guide', 'info', 'information', 'show', 'how', 'instruction', 'clear', 'clearer', 'explain',  'clarity'],
    ["time", 'pressure', "workload", "overload", "work", "deadline", 'spread', 'spaced', 'space', 'pace', 'pacing', 'paced', 'break', 'slower', 'slow', 'shorter', 'short', 'extension'],
    ["vle", "platform", 'software', 'interface', 'onlinemeetingtool', 'technology', 'system', 'tech', 'navigation'],
    ['wifi', 'network', 'broadband', "connection", "internet", "data", 'laptop', 'computer', "equipment", 'specialist', 'specialised', 'software', 'camera', 'mic', 'device'],
    ['talk', 'reply', 'interact', 'response', 'respond', 'contact', 'listen', 'email', "communication", "communicate", "communicating", "feedback", 'questions'],
    ['participation', 'interactive', 'interactivity', 'interaction', 'discussion', 'quiz', 'activity', 'variety', 'kahoot', 'kahoots'],
    ['personal', 'individual', 'seminar', 'tutorial', 'group', 'groupwork', 'small', 'workshop', 'smaller', 'onetoone', 'onetoones', 'collaboration'],
    ['nothing', 'happy', 'keep', 'good', 'same', 'continue', 'great', 'fine', 'satisfied', 'perfect'],
    ['more organised', 'organised', 'arrange', 'staff', 'organized', 'organisation', 'structure', 'structured', 'planned', 'detailed', 'manage', 'consistent', 'consistency', 'management', 'plan', 'timetable', 'timetabled', 'earlier', 'later', 'time',  'schedule'],
    ['assessment', 'assignment', 'exam', 'revision', 'results', 'practice', 'test', 'revise', 'track', 'progress'],
    ['teaching', 'effectively', 'teacher', 'engaged', 'involved', 'involvement', 'engage', 'engagement', 'engaging', 'interesting', 'boring'],
    ['online', 'learning', 'dont', 'not', 'less', 'stop', 'back', 'rid'],
    ['accessible', 'accessibility', 'subtitle', 'caption', 'blind', 'deaf', 'dyslexia', 'dyslexic', 'captioning'],
    ['dont', 'know', 'not', 'sure', 'idea'],
    ['tip', 'skill', 'basic', 'digital', 'training', 'train', 'online', 'knowledge']
]

#    ['long', 'screen', 'time', 'early', 'late']
print("\nAnchored\n")
df2 = anchored_topic_model(data, 'cleaned', topic_names=topic_names, anchors=anchors, print_topic_details=True, return_csv=True)
# manual fix for a model problem - the single word 'nothing' doesn't get allocated to a topic
terms = df2[(df2['Topic name'] == 'Nothing/OK')]['Topic label'].values[0]
df2.loc[df2['cleaned'] == 'nothing', 'Topic label'] = terms
df2.loc[df2['cleaned'] == 'nothing', 'Topic name'] = 'Nothing/OK'
df2.to_csv('output/corex.csv')

#
# Create a topic model for all the terms that weren't matched to see if there are any patterns
#
print("\nUnmatched\n")
unmatched = df2[(df2['Topic label'] == 'No matching topic')].copy()
df2 = anchored_topic_model(unmatched, 'cleaned', number_of_topics=18, print_topic_details=True)
df2.to_csv('output/corex_unmatched.csv')
