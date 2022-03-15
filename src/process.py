import json, os, unicodedata, re
from run_utils import Voc
from train_p2 import MAX_LENGTH
# print(json.dumps(data[2], indent=4, sort_keys=True))

def extractSentencePairs_ConvAI2(data_names):
    save_dir = os.path.join("save")
    qa_pairs = []
    for data_name in data_names:
        # data_name = "data_intermediate.json"
        with open(os.path.join(save_dir, data_name),'r') as file:
            data = json.load(file)
        for conversation in data:
            dialog = conversation['dialog']
            if len(dialog) > 1:
                for i in range(0, len(dialog), 2):
                    if(i >= len(dialog) - 1):
                        break
                    qa_pairs.append([dialog[i]["text"],dialog[i+1]["text"]])
    return qa_pairs

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def normalizePairs(pairs):
    print("Reading pairs...")
    pairs = [[normalizeString(s) for s in pair] for pair in pairs]
    return pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadData(corpus_name, pairs):
    print("Start preparing training data ...")
    pairs = normalizePairs(pairs)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    voc = Voc(corpus_name)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs