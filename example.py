import pandas as pd
from qualkit.clean import clean, remove_dont_knows, lemmatize
from qualkit.anchored_topic_model import anchored_topic_model

#  Using an anchored topic model
#  =============================
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
data = lemmatize(data, 'cleaned')

#
# Create a topic model with no anchor topics as a baseline
#
print("\nNon-anchored\n")
df1 = anchored_topic_model(data, 'cleaned', number_of_topics=19, print_topic_details=True)

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
    'Session length',
    'Digital skills'
]
anchors = [
    ["resource", "library", "book", 'access', 'textbook', 'ebooks'],
    ["pre recorded", "prerecord", "recorded", 'record', "prerecorded", "video", "content", 'powerpoint', 'material', 'powerpoints'],
    ["live", "face", "physical", 'on campus', 'in person'],
    ["help", "support", 'advice', 'motivate', "supportive", "mental health", 'safe', 'safety', 'assistance', 'awareness', 'outreach', 'check in'],
    ['guidance', 'info', 'information', 'show how', 'how to', 'more guidance', 'more information', 'instruction', 'clear', 'clearer', 'explain',  'clarity'],
    ["time", 'pressure', "workload", "overload", "work", "deadline", 'spread', 'spaced', 'space', 'pace', 'pacing', 'paced', 'break', 'slower', 'slow', 'shorter', 'short', 'extension'],
    ["vle", "platform", 'software', 'interface', 'onlinemeetingtool', 'technology', 'system', 'tech', 'navigation'],
    ['wifi', 'broadband', "connection", "internet", "data", 'laptop', 'computer', "equipment", 'specialist software', 'specialised software', 'camera', 'mic', 'device'],
    ['talk', 'reply', 'interact', 'response', 'respond', 'contact', 'listen', 'email', "communication", "communicate", "communicating", "personal tutor", "better communication", "feedback", 'ask questions'],
    ['participation', 'interactive', 'interactivity', 'interaction', 'discussion', 'quiz', 'activity', 'variety', 'kahoot', 'kahoots'],
    ['personal', 'individual', 'seminar', 'tutorial', 'group', 'groupwork', 'small group', 'workshop', 'smaller', 'onetoone', 'onetoones', 'one one', 'collaboration'],
    ['nothing', 'nothing improve', 'happy', 'keep', 'good', 'same', 'continue', 'great', 'fine', 'satisfied', 'all good'],
    ['more organised', 'staff', 'organised', 'organized', 'organisation', 'structure', 'structured', 'planned', 'detailed', 'manage', 'consistent', 'consistency', 'management', 'plan', 'timetable', 'timetabled', 'earlier', 'later', 'on time',  'schedule'],
    ['assessment', 'assignment', 'exam', 'revision', 'results', 'practice', 'test', 'revise','track','progress'],
    ['teaching', 'effectively', 'teacher', 'engaged', 'involved', 'involvement', 'engage', 'engagement', 'engaging', 'interesting', 'less boring'],
    ['online learning', 'dont online learning', 'not online learning', 'more online learning', 'less online learning', 'stop online learning', 'not online', 'no online', 'stop online learning', 'go back', 'stop doing', 'rid online learning'],
    ['accessible', 'accessibility', 'subtitle', 'caption', 'blind', 'deaf', 'dyslexia', 'dyslexic', 'captioning'],
    ['dont know', 'not sure', 'no idea'],
    ['long', 'screen time', 'early', 'late'],
    ['tip', 'skill', 'basic', 'digital', 'basic skill', 'digital skill', 'training session', 'train', 'training online', 'knowledge']
]
print("\nAnchored\n")
df2 = anchored_topic_model(data, 'cleaned', topic_names=topic_names, anchors=anchors, print_topic_details=True)
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