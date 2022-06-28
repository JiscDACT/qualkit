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

## Output

The topic model will append a column for each topic and will indicate whether that row of data (for a particular response) is True or False for that topic. 
This enables you to assess responses which may fall into multiple topics.

For instance, the last response here (fake responses) covers two topics:

| Q12 (What are positive aspects of online learning?) | topic_1 (time) | topic_2 (sleep) | topic_3 (Replay) |
|-----------------------------------------------------|----------------|-----------------|------------------|
| Learning online gives me more time                  | True           | False           | False            |
| I can rewatch lectures                              | False          | False           | True             |
| I can get more sleep in and gives me more time      | True           | True            | False            |

**To Do / In Progress**

Model also appends probability a document belongs to a topic given that document's words (using log_p_y_given_x) but as this
is not a discriminative model, CorEx estimates probability a document belongs to a topic separately for each topic (probabilities don't add to 1)

Methods to get document for each topic in future.
1. Use the p_y_given_x attribute or log_p_y_given_x attributes to rank which documents are most probable for each topic. 
2. Get a binary classification of each document in each topic from labels (which applies a softmax from p_y_given_x).
3. You can also use log_z to rank which documents are "explained" the most by each topic according to pointwise total correlation. 

Most simple: Using labels or p_y_given_x


## Analysing unmatched terms
It can be useful, certainly in the initial stages of modelling, to see if there are
any patterns in the unmatched terms that suggest changes to the model. To do this,
extract the unmatched items from the results and run a model without anchors.

Add the following code after the code for creating an anchored model:

    unmatched = df[(df['Topic label'] == 'No matching topic')].copy()
    df = anchored_topic_model(unmatched, 'cleaned', number_of_topics=12, print_topic_details=True)

## Different approaches to Anchor terms
Anchor terms can be included in different ways:
1. **Anchoring a single set of words to a single topic**. This can help promote a topic that 
did not naturally emerge when running an unsupervised instance of the CorEx topic model. 
For example, one might anchor words like "snow," "cold," and "avalanche" to a topic if 
one suspects there should be a snow avalanche topic within a set of disaster relief articles. 
    
   
    anchor_words = [['snow', 'cold', 'avalanche']]
2. **Anchoring single sets of words to multiple topics**. This can help find different aspects 
of a topic that may be discussed in several different contexts. For example, one might anchor 
"protest" to three topics and "riot" to three other topics to understand different framings 
that arise from tweets about political protests.


    anchor_words = ['protest', 'protest', 'protest', 'riot', 'riot', 'riot']
3. **Anchoring different sets of words to multiple topics**. This can help enforce topic separability 
if there appear to be chimera topics. For example, one might anchor "mountain," "Bernese," and "dog" 
to one topic and "mountain," "rocky," and "colorado" to another topic to help separate topics that 
merge discussion of Bernese Mountain Dogs and the Rocky Mountains.


    anchor_words = [['bernese', 'mountain', 'dog'], ['mountain', 'rocky', 'colorado']]


Examples of a more detailed strategy:

Here we anchor "nasa" by itself, as well as in two other topics each with "politics" and "news" to find different 
aspects around the word "nasa". We also create a fourth anchoring of "war" to a topic.

    anchor_words = ['nasa', ['nasa', 'politics'], ['nasa', 'news'], 'war']

Information taken from the following [resource](https://notebook.community/gregversteeg/corex_topic/corextopic/example/corex_topic_example).

## Anchor Strength

This can be added as an argument, if not specified defaults to 2:

`anchored_topic_model(data, column, topic_filename=None, topic_names=None, anchors=None, number_of_topics=10,
                         print_topic_details=False, anchor_strength_int=2)`


Choosing anchor strength: the anchor strength controls how much weight CorEx puts 
towards maximizing the mutual information between the anchor words and their respective topics. 
Anchor strength should always be set at a value greater than 1, since setting anchor strength 
between 0 and 1 only recovers the unsupervised CorEx objective. Empirically, setting anchor strength 
from 1.5-3 seems to nudge the topic model towards the anchor words. Setting anchor strength 
greater than 5 is strongly enforcing that the CorEx topic model find a topic associated with the anchor words.

Anchor strength represents the relative amount of weight that CorEx assigns to a word relative to other words. 
So if the anchor strength is 2, then CorEx gives twice the weight to that word relative to all other words. 
The second is that the higher the anchor strength, the less room the topic model has to find topics 
because you have already dictated what the topics should be. Avoid setting anchor strength in the thousands, 
or even the hundreds, because that's likely going to force all of the anchor words to be the top topic words, 
and at that point you're probably better off with a keyword approach than a topic model. 
So I would check that the topics you're getting still have some flexibility to them after you set the anchor
strength. If you're only seeing the anchor words as the top topic words, the anchor strength may be set too 
aggressively.


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

**anchor_strength_int** _default=2_ An integer to assign anchor strength for the topic model