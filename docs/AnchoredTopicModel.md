# Anchored Topic Modelling
This is a Python project that helps you to create, evaluate and 
tune an Anchored Topic Model.

An Anchored Topic Model is a semi-supervised type of topic model; rather 
than being completely automatic, it uses the domain 
knowledge you supply to 'steer' the model towards certain topics in the form 
of 'anchor terms'.

How do you do this?

The process is iterative, with the following stages:

1. Create a first run of the model without any 'anchors' to see what the 
algorithm suggests as topics.

2. Create a set of 'anchors' (see below) and re-run the model.
   
3. Open the results in Tableau and experiment. 
   
In particular, pay attention to how much of the input it can match to a topic 
(the 'no matching topic' vs. the total) and whether the content with 'no matching topic'
has any recurring terms that are missing from the topics that could be useful anchors. 

(Some content just won't match any model, for example, empty text random characters, 
single words like 'No', and things that just aren't a good match.)

4. Refine the anchor terms, and re-run, then repeat steps 3-4 until you're happy
with the model.
   
## Anchor terms
Anchor terms are specified as a list of single words and phrases of 1-3 words.

For example, the following is a set of anchors for two topics:
    
    ["resources", "library", "books"],
    ["pre recorded lectures"]

You can use as few or as many anchor terms as you like. It generally works best to
start with just a few terms and iteratively add them - the topic model algorithm 
will probably pick up many of the related terms itself without any more help, but
may miss important synonyms that rely on domain knowledge.

You can also supply a name for each 'anchored' topic.

## Cleaning the input text
The project includes a generic cleaning utility that you can run to clean up your 
input:

    data = pd.read_csv('data/dei_student_all.csv')
    data = clean(data, 'Q26')

The output is written into a new column, 'cleaned'.

You can also use it to remove any 'don't know' answers including "N/A" and variou
other synonyms:

    data = remove_dont_knows(data, 'cleaned')

## Creating the first model without anchors
The following code will load and clean data, then print the topics found in the
text to the console as a list of terms:

    import pandas as pd
    import anchored_topic_model

    data = pd.read_csv('data/dei_student_all.csv')
    data = clean(data, 'Q26')
    data = lemmatize(data, 'cleaned')
    data = remove_dont_knows(data, 'cleaned')
    df = anchored_topic_model(data, 'cleaned', number_of_topics=12, print_topic_details=True)

## Create a model with anchors:
The following code will load and clean the data, and then create a topic model
using the topic names and anchors supplied, and write it out as a CSV file
that includes the topics as well as the original data you supplied:

    data = pd.read_csv('data/dei_student_all.csv')
    data = clean(data, 'Q26')
    data = remove_dont_knows(data, 'cleaned')

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
        ["resource", "library", "book", 'access'],
        ["pre recorded", "recorded", 'record', "pre recorded lectures", "video", "content", 'powerpoint', 'material', 'accessible', 'subtitle'],
        ["live lecture", "face", "physical", 'on campus', 'in person', 'live lesson', 'live', 'less online', 'not online'],
        ["help", "support", 'motivate', "supportive", "mental health", 'safe', 'assistance', 'awareness', 'outreach'],
        ['guidance', 'information', 'training', 'more guidance', 'more information'],
        ["time", 'pressure', "workload", "overload", "work", "deadline", 'spread out', 'space out', 'pace', 'pacing', 'shorter', 'break'],
        ["vle", "platform", 'software', 'interface', 'onlinemeetingtool', 'technology', 'system'],
        ['wifi', "connection", "internet", "data", 'laptop', 'computer', "equipment", 'specialist software', 'specialised software'],
        ['reply', 'response', 'explain', 'respond', 'contact', 'listen', 'email', "communication", "communicate", "personal tutor", "better communication", 'clear', 'clearer', 'clarity', "feedback", 'ask questions'],
        ['participation', 'involvement', 'interactive', 'interactivity', 'engage', 'engagement', 'engaging', 'interesting', 'interaction', 'discussion', 'quiz', 'activity'],
        ['personal', 'individual', 'seminar', 'tutorial', 'group', 'groupwork', 'small group', 'workshop', 'smaller', 'onetoone'],
        ['happy', 'nothing', 'keep same', 'keep doing', 'good', 'same', 'continue', 'great', 'fine'],
        ['staff', 'organised', 'organized', 'organisation', 'structure', 'structured', 'planned', 'timetable', 'detailed', 'manage', 'schedule', 'consistent', 'management']
    ]
    df = anchored_topic_model(data, 'cleaned', topic_names=topic_names, anchors=anchors, print_topic_details=True)
    df.to_csv('output.csv')

## Analysing unmatched terms
It can be useful, certainly in the initial stages of modelling, to see if there are
any patterns in the unmatched terms that suggest changes to the model. To do this,
extract the unmatched items from the results and run a model without anchors.

Add the following code after the code for creating an anchored model:

    unmatched = df[(df['Topic label'] == 'No matching topic')].copy()
    df = anchored_topic_model(unmatched, 'cleaned', number_of_topics=12, print_topic_details=True)

## Further reading

https://medium.com/pew-research-center-decoded/overcoming-the-limitations-of-topic-models-with-a-semi-supervised-approach-b947374e0455

https://transacl.org/ojs/index.php/tacl/article/view/1244

## API reference

### anchored_topic_model(_data, column, topic_names=None, anchors=None, number_of_topics=10, print_topic_details=False_)

**data**: a pandas dataframe containing the data.

**column** a string with the column name to use as the source text.

**topic_names** _(optional)_ a list of names to use for topics.

**anchors** _(optional)_ a list of list of anchor terms to use to steer the topic model.

**number_of_topics** _default=10_ the number of topics to generate.

**print_topic_details** _default=False_ prints to the console the topics generated and their top 10 terms