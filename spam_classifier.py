## Names: Yeo Wan Ling, Ng Hiau Chin, Nur Amisha Binte Maludin
##
## The following code gives some initial direction in building a spam detection
## tool using resources from the nltk library. The code use several publicly
## available datasets that have been processed into a common format for ease of
## use. You may find other online sources that can give you further ideas on how
## to improve on analyzing and detecting spam, and you are welcome to incorporate
## such code into your project.
import nltk, re
from nltk.corpus import stopwords, names
nltk.download('averaged_perceptron_tagger_eng')
from statistics import mean
from random import shuffle
import pandas as pd
# The import below is a progress bar library, which may require installation
# via pip. Uncomment these lines and corresponding lines with `tqdm` below
# if you want to visualize loop progress.
#
from tqdm import tqdm

## The corpora you will primarily be using are located in the `datasets`
## subfolder, and a brief explanation of each are given here:
## `SMSSpamCollection.txt` - this is the full UCI dataset, which is compiled
##        from various sources. It contains both `ham` and `spam` labeled
##        messages. You can view and read more about the dataset at:
##        https://archive.ics.uci.edu/dataset/228/sms+spam+collection
## `SMSSpamCollection_500.txt` - this is a set of the first 500 messages from
##        the dataset above, and is used in the code below for testing
## `sms_corpus_NUS-ham.txt` - this is the full SMS corpus developed by NUS Singapore
##        for machine learning & NLP. All of these messages are actual messages
##        collected in Singapore from study participants. I have reformatted the
##        data to match the format of the other datasets. You can read more at:
##        https://github.com/kite1988/nus-sms-corpus
## `email_corpus_lingspam.txt` - this is a set of real emails from the Linguist
##        List (a listserve site for dissemination information about linguistics)
##        along with spam emails. I have reformatted the data to match the
##        other datasets, including `ham` and `spam` labels. This is the
##        dataset you will be using for evaluation, and you should not use it
##        for training your classifier. You can download the original dataset at:
##        http://nlp.cs.aueb.gr/software_and_datasets/lingspam_public.tar.gz
##        The zipped folder contains a `readme` file explaining the data.

# Now let's take a look at the corpus we will use to test the classifier
# sms = pd.read_table('datasets/SMSSpamCollection_500.txt', header=None, encoding='utf-8') # use this for testing/illustration because it is smaller
sms = pd.read_table("Spam_Text_Classifier/datasets/SMSSpamCollection.txt", header=None, encoding='utf-8') # use this for the final version of the project
print(sms.head()) # print the firse 5 lines of the dataframe
print(sms.info()) # print general info about the dataframe
print(sms[0].value_counts()) # print the labels located in the first column of the dataframe

# exit() # delete/comment-out this line when you're ready for the next step

# The corpus is a single text file with tab-delimited lines, where the first
# item in the line is the label ('spam' or 'ham') and the second item is the
# actual message.
# Above we read the messages into a pandas dataframe (`sms`), and we can further
# separate out the different kinds of messages by creating new dataframes below.
ham_sms = sms[sms[0]=='ham'] # get all lines where the first column has the label 'ham'
print("number of ham messages:", len(ham_sms)) # print our number (this should match output from line 48 above)
print(ham_sms.head()) # print the first 5 lines in the dataframe
spam_sms = sms[sms[0]=='spam'] # get all lines where the first column has the label 'spam'
print("number of spam messages:", len(spam_sms)) # print our number (this should match output from line 48 above)
print(spam_sms.head()) # print the first 5 lines in the dataframe
all_sms = ham_sms + spam_sms # combine the two dataframes in order, all 'ham' messages, then all 'spam' messages
print("number of total messages:", len(all_sms))

# exit() # delete/comment-out this line when you're ready for the next step

# We can now process the text in the messages.
# First let's remove the less 'contentful' words
unwanted = stopwords.words("english") # create a list of 'stopwords'
unwanted.extend([w.lower() for w in names.words()]) # extend this with a list of names
print("Number of 'unwanted' words:", len(unwanted)) # print the information to the terminal

# exit() # delete/comment-out this line when you're ready for the next step

# Let's now create a function to go through sentences tagged with Part-Of-Speech
# (POS) at the word level and skip them if they meet some criteria
def skip_unwanted(pos_tuple):
    word, tag = pos_tuple # for the two items in the pos-tagged tuple
    # check if the 'word' is alphabetic or in the `unwanted` list
    if not word.isalpha() or word in unwanted:
        # if it is, return the value `False`
        return False
    # also check if the 'word' is tagged as a noun (see https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk)
    if tag.startswith("NN"):
        # if so, return the value `False`
        return False
    # if neither of these conditions are met, keep the word
    return True

# We may be dealing with nested lists when using nltk's `pos_tag()` on
# paragraphs, so let's write a simple function to flatten these nested lists
def flatten(nested_list):
    flat = []
    for item in nested_list:
        if type(item) == list:
            for wd in item:
                flat.append(wd)
        else:
            flat.append(item)
    return flat

# Now let's use our function to create a list of words based on the words in the
# `spam` and `ham` categories of our corpus (using list comprehension).
# Here the function `filter()` uses our function to filter the output of the
# nltk `pos_tag()` function.

print("Getting the ham words...")
# first get the words found in the `ham` category
ham_words = [x for x in ham_sms[1].apply(nltk.word_tokenize)] # tokenize the `ham` words
ham_tags = [nltk.pos_tag(x) for x in ham_words] # tag the `ham` words
ham_flat = flatten(ham_tags) # flatten the nested lists of tuples (tagged words)
# now reduce the set to only words we want, using our function above
ham_words = [word.lower() for word, tag in filter(
    # first argument for `filter` is the function we created
    skip_unwanted,
    # second argument is the pos-tagged set of words from the `ham` category of text messages
    ham_flat
    )]

print("number of `ham` words:", len(ham_words))

# exit() # delete/comment-out this line when you're ready for the next step

print("Getting the `spam` words...")
# next get the words found in the `spam` category
spam_words = [x for x in spam_sms[1].apply(nltk.word_tokenize)] # tokenize the `spam` words
spam_tags = [nltk.pos_tag(x) for x in spam_words] # tag the `spam` words
spam_flat = flatten(spam_tags) # flatten the nested lists of tuples (tagged words)
# now reduce the set to only words we want, using our function above
spam_words = [word.lower() for word, tag in filter(
    # first argument for `filter` is the function we created
    skip_unwanted,
    # second argument is the pos-tagged set of words from the `neg` category
    # of movie reviews
    spam_flat
    )]

print("number of `spam` words:", len(spam_words))

# exit() # delete/comment-out this line when you're ready for the next step

print("Getting the frequency distributions...")
# now let's create frequency distributions of each list of words
ham_fd = nltk.FreqDist(ham_words)
spam_fd = nltk.FreqDist(spam_words)

# get the set of words that are common to both
common_set = set(ham_fd).intersection(spam_fd)

# remove words from each frequency distribution that occur in common, as these
# don't meaningfully distinguish between the types of messages
for word in common_set:
    del ham_fd[word]
    del spam_fd[word]

# create a dictionary of the top 100 words remaining in each frequency distribution
top_100_ham = {word for word, count in ham_fd.most_common(100)}
top_100_spam = {word for word, count in spam_fd.most_common(100)}

# We now have a set of features consisting of the top 100 words occurring uniquely
# in `ham` and `spam` messages.
print(top_100_ham)
print(top_100_spam)

# exit() # delete/comment-out this line when you're ready for the next step

# We can now define a function to count these 'features' (words) in each sentence

# Sample function to demonstrate integrating new features into your current `extract_features` function

def extract_features(text):
    # for each text we're given
    features = dict() # instantiate a new dictionary
    wordcountham = 0 # instantiate a counter
    #wordcountspam = 0
    text_lower = text.lower()
    
    # loop through each sentence in the paragraph
    for sentence in nltk.sent_tokenize(text):
        # loop through each word in the sentence
        for word in nltk.word_tokenize(sentence):
            # if the word is in our set of positive words
            if word.lower() in top_100_ham:
                wordcountham += 1 # increment the counter
            # note that our function only creates a 'feature' that counts
            # `ham` words - we could similarly create a feature for
            # counting `spam` words
            #if word.lower() in top_100_spam:
               #wordcountspam += 1

    # Excessive Exclamation Marks
    exclamation_count = text.count("!")
    features["exclamation_count"] = exclamation_count >= 3

    # Spammy Words
    features["has_spam_trigger"] = bool(re.search(
    r"\b(free (gift|entry|membership|ringtone|trial|offer|prize|cash)|win (cash|money|now|prize)?|claim (now|reward|offer|prize|your))\b",
    text_lower))

    # Excessive URLs
    urls = re.findall(r"http[s]?://|www\.|\S+\.com", text_lower)
    features["has_url"] = len(urls) >= 3

    # Low Lexical Diversity
    tokens = nltk.word_tokenize(text_lower)
    unique_tokens = set(tokens)
    features["low_lexical_diversity"] = (len(unique_tokens) / len(tokens)) < 0.5 if tokens else False

    features["wordcount_ham"] = wordcountham # store the count of ham words
    #features["wordcount_spam"] = wordcountspam #store the count of spam words
    
# return the dictionary of features for each text/paragraph
    return features

print("Extracting features...")
# Now we can create a new list consisting of a dictionary of features for each
# text in the sms dataset.
# First we extract features for the `ham` messages:
features = [
    (extract_features(text), "ham")
    # for text in ham_sms[1]
    for text in tqdm(ham_sms[1]) # uncomment this line and comment out the line above to view progress
    ]

# Then we extend this to the `spam` messages so that all messages have the same
# number of features:
features.extend([
    (extract_features(text), "spam")
    # for text in spam_sms[1]
    for text in tqdm(spam_sms[1]) # uncomment this line and comment out the line above to view progress
    ])

print(len(features))

# exit() # delete/comment-out this line when you're ready for the next step

# Now that we have 'featurized' all our text, we can use these features to train
# our classifier.

# First we get the length of a quarter of the dataset
train_count = len(features) // 4

shuffle(features) # shuffle the dataset

print(F"Training classifier on {train_count} samples")
# instantiate a classifier that trains on the first quarter of the dataset
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
# print out the 10 features that most clearly distinguish between positive/negative classes
classifier.show_most_informative_features(10)


print(F"Assessing classifier on {len(features)-train_count} samples")
# assess how accurate the trained classifier is on the last three quarters of the dataset
print("Accuracy of classifier:", nltk.classify.accuracy(classifier, features[train_count:]))

# Now let's import an unseen dataset of different data that has also been
# labeled as spam/ham. This is email data from the Linguist List, which I have
# processed into the same format as our SMS data.

# First we load the data and print some info on it
emails = pd.read_table('Spam_Text_Classifier/datasets/email_corpus_lingspam.txt', header=None, encoding='utf-8')
print(emails.head())
print(emails.info())
print(emails[0].value_counts())

# Then we featurize it
efeatures = [
    (extract_features(text), label)
    for text, label in zip(emails[1], emails[0])
    # for text, in tqdm(ham_sms[1]) # uncomment this line and comment out the line above to view progress
    ]
# Then we assess our classifier's accuract on the featurized texts
print(F"Assessing classifier on {len(efeatures)} samples")
print("Accuracy of classifier on email data:", nltk.classify.accuracy(classifier, efeatures[:]))

# Let's now try to add another feature to our classifier, to see whether we can
# improve our score.

# You may notice that every time you run this code, your final accuracy score
# will differ. This is because you are shuffling (randomizing) your set of
# features used to train the classifier. It is common practice in machine
# learning to score a classifier based on the average accuracy value over a
# set number of runs, typically around 10. There are also ways of ensuring
# consistency (replicability) between runs, but taking the average of 10 runs
# will suffice for our purposes here. It is also common practice to train on a
# larger portion of the dataset and validate on a smaller portion of the dataset,
# so feel free to experiment with different "splits". To train a more robust
# model, you should also use a technique known as "cross-validation", but for
# now we will be content with simply taking the average of 10 runs.

# The code below runs 10 iterations of the classifier and writes each score to
# a text file, along with a final average score (mean) across all 10 iterations.
with open('outputresults.txt', 'w') as f:
    iterations = []
    for x in list(range(10)):
        print(x)
        shuffle(features) # shuffle the dataset
        classifier = nltk.NaiveBayesClassifier.train(features[:]) # train a new classifier on all the sms data
        accuracy = nltk.classify.accuracy(classifier, efeatures[:]) # check its accuracy on all the email data
        iterations.append(accuracy) # store the accuracy value for this run
        f.write(str(accuracy)+"\n") # write the value to our output file
    f.write("\n"+str(mean(iterations))) # write the mean value to our output file

# Now that you understand the basics of how to build a classifier, continue to
# experiment with developing features that may help your classifier to
# distinguish better between ham and spam emails.