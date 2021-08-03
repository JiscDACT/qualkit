import pandas as pd
from qualkit.clean import clean, remove_dont_knows
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

print("\nNon-anchored\n")
df1 = anchored_topic_model(data, 'cleaned', number_of_topics=12, print_topic_details=True)

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
    'Organisation']
anchors = [
    ["resources", "library", "books", 'access'],
    ["pre recorded", "recorded", 'record', "pre recorded lectures", "video", "videos","content",'materials','powerpoint','material','accessible', 'subtitles'],
    ["live lectures", "face", "physical", 'on campus', 'in person','live lessons','live','less online','not online'],
    ["help", "support", 'motivate', "supportive", "mental health", 'safe', 'assistance', 'awareness', 'outreach'],
    ['guidance', 'information', 'training', 'more guidance', 'more information'],
    ["time", 'pressure', "workload", "overload", "deadlines", "work", "deadline", 'spread out', 'space out', 'pace', 'pacing', 'shorter', 'breaks'],
    ["vle", "platform", 'software', 'interface', 'platforms', 'onlinemeetingtool', 'technology', 'system'],
    ['wifi', "connection", "internet", "data", 'laptops', 'laptop', 'computers', "equipment", 'specialist software', 'specialised software'],
    ['reply', 'response', 'explain', 'respond', 'contact', 'listen', 'email', "emails", "communication", "communicate", "personal tutor", "better communication", 'clear', 'clearer', 'clarity', "feedback", 'ask questions'],
    ['participation', 'involvement', 'interactive', 'interactivity', 'engage', 'engagement', 'engaging', 'interesting', 'interaction', 'discussion', 'quizzes', 'quiz', 'activity', 'activities'],
    ['personal', 'individual', 'seminars', 'tutorials', 'tutorial', 'group', 'groupwork', 'small groups', 'workshops', 'smaller', 'onetoone', 'onetoones'],
    ['happy', 'nothing', 'keep same', 'keep doing', 'good', 'same', 'continue', 'great', 'fine'],
    ['staff', 'organised', 'organized', 'organisation', 'structure', 'structured', 'planned', 'timetable', 'detailed', 'manage', 'schedule', 'consistent', 'management']
]
print("\nAnchored\n")
df2 = anchored_topic_model(data, 'cleaned', topic_names=topic_names, anchors=anchors, print_topic_details=True)
df2.to_csv('output/corex.csv')

print("\nUnmatched\n")
unmatched = df2[(df2['Topic label'] == 'No matching topic')].copy()
df2 = anchored_topic_model(unmatched, 'cleaned', number_of_topics=12, print_topic_details=True)