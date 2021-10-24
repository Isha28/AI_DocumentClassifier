import os
import math
from numpy import log as ln

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r',encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}

    # This function creates bag of words giving count for each word in vocabulary,
    # if word not present in vocabulary, it is added to None
    
    noneflag = 0
    
    contents = open(filepath,"r",encoding='utf-8')
    for eachline in contents:
        eachline = eachline.strip()    
        eachline = eachline.lower()
        eachwords = eachline.split(" ")

        for eachword in eachwords:
            if eachword in vocab:
                if eachword in bow:
                    bow[eachword] = bow[eachword] + 1
                else:
                    bow[eachword] = 1
            else:
                if noneflag == 0:
                    noneflag = 1
                    bow[None] = 1
                else:
                    bow[None] = bow[None] + 1
                    
    return bow

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """    
    smooth = 1 # smoothing factor
    logprob = {}

    labelCntDict = {}
    for eachlabel in label_list:
        labelCntDict[eachlabel] = 0
        
    for eachlabel in label_list:
        for eachtraindata in training_data:
            if eachtraindata['label'] == eachlabel:
                 labelCntDict[eachlabel] += 1

    tot = sum(labelCntDict.values())

    for eachlabel in label_list:
        temp = (labelCntDict[eachlabel]+1)/(tot+2)
        logprob[eachlabel] = math.log(temp)

    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}

    sizevocab = len(vocab)
    totword = 0
    for traindata in training_data:
        if traindata['label'] == label:
            for word in traindata['bow']:
                totword += traindata['bow'][word]

    for word in vocab:
        wordintrain = 0
        for traindata in training_data:
            if traindata['label'] == label and word in traindata['bow']:
                wordintrain += traindata['bow'][word]

        word_prob[word] = math.log((wordintrain + 1)/(totword+sizevocab+1))

    word_prob[None] = 0

    wordintrain = 0
    for traindata in training_data:
        if (label == traindata['label']):
            for word in traindata['bow']:
                if word not in vocab:
                    wordintrain += traindata['bow'][word]

    word_prob[None] = math.log((wordintrain + 1)/(totword+sizevocab+1))
    return word_prob


def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)

    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(retval['vocabulary'],training_directory)
    retval['log prior'] = prior(training_data, label_list)
    for label in label_list:
        if label == '2016':
            retval['log p(w|y=2016)'] = p_word_given_label(retval['vocabulary'],training_data,label)
        else:
            if label == '2020':
                retval['log p(w|y=2020)'] = p_word_given_label(retval['vocabulary'],training_data,label)
            
    return retval

def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    bow = create_bow(model['vocabulary'], filepath)
        
    totword = 0 
    for vocword in bow:
        if vocword in model['log p(w|y=2016)']:
            totword += (model['log p(w|y=2016)'][vocword])*(bow[vocword])
        else:
            totword += (model['log p(w|y=2016)'][None])*(bow[vocword])

    priorval = model['log prior']['2016']

    firstval = totword + priorval

    totword = 0 
    for vocword in bow:
        if vocword in model['log p(w|y=2020)']:
            totword += (model['log p(w|y=2020)'][vocword])*(bow[vocword])
        else:
            totword += (model['log p(w|y=2020)'][None])*(bow[vocword])

    priorval = model['log prior']['2020']

    secondval = totword + priorval

    retval['log p(y=2020|x)'] = secondval
    retval['log p(y=2016|x)'] = firstval
    if (firstval > secondval):
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'    
        
    return retval
