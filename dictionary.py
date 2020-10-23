import sys, re, collections
import collections, nltk
from nltk.tokenize import RegexpTokenizer


def readDict(dictionaryPath):
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = collections.OrderedDict()

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections; throw error and die
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        try:
            if re.split(r'\t+', line)[0] == '':
                catList[re.split(r'\t+', line)[1]] = [re.split(r'\t+', line.rstrip())[2]]
            else:
                catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+', line.rstrip())[1]]
        except:  # likely category tags
            pass

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        if "(" in key and ")" in key:
            key = key.replace("(", "").replace(")", "")
        # these words are ambiguous and cause errors
        if key == "kind" or key == "like":
            continue
        if not key in finalDict:
            finalDict[key] = []
        for catnum in values:
            try:  # catch errors (e.g. with dic formatting)
                workingValue = catList[catnum][0]
                finalDict[key].append(workingValue)
            except:
                print(catnum)
    return (finalDict, catList.values())


def wordCount(data, dictOutput, catList):
    # Create a new dictionary for the output
    outList = collections.OrderedDict()

    # Number of non-dictionary words
    nonDict = 0

    # Convert to lowercase
    data = data.lower()

    # Tokenize and create a frequency distribution
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)

    fdist = nltk.FreqDist(tokens)
    wc = len(tokens)

    # bad stems
    bad_stems = []

    # Using the Porter stemmer for wildcards, create a stemmed version of the data
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in tokens]
    # handle bad stems
    # some words get counted twice due to created stem being different from the actual word
    # e.g. "happy" gets stemmed to "happi*" which produces an additional distinct match later on
    # so we fix this
    for stem in stems:
        good_token = False
        for token in tokens:
            if stem in token:
                good_token = True
        if good_token == False:
            bad_stems.append(stem)

    fdist_stem = nltk.FreqDist(stems)

    # Access categories and populate the output dictionary with keys
    for cat in catList:
        outList[cat[0]] = 0

    # Dictionaries are more useful
    fdist_dict = dict(fdist)
    fdist_stem_dict = dict(fdist_stem)
    # print(bad_stems)
    for stem in bad_stems:
        fdist_stem_dict.pop(stem, None)
    # print(fdist_stem_dict)

    # Number of classified words
    classified = 0

    for key in dictOutput:
        if "*" in key and key[:-1] in fdist_stem_dict:
            classified = classified + fdist_stem_dict[key[:-1]]
            for cat in dictOutput[key]:
                outList[cat] = outList[cat] + fdist_stem_dict[key[:-1]]
        elif key in fdist_dict:
            classified = classified + fdist_dict[key]
            for cat in dictOutput[key]:
                outList[cat] = outList[cat] + fdist_dict[key]

    # Calculate the percentage of words classified
    if wc > 0:
        percClassified = (float(classified) / float(wc)) * 100
    else:
        percClassified = 0

    # Return the categories, the words used, the word count, the number of words classified, and the percentage of words classified.
    return [outList, tokens, wc, classified, percClassified]


dictIn, catList = readDict("data/LIWC2015_English.dic")
# run the wordCount function

def countWords(text):
    return wordCount(text, dictIn, catList)
