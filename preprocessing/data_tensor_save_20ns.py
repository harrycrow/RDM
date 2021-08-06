import codecs
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from model import *
from sklearn.feature_extraction.text import CountVectorizer


def train(tfboard=False):
    window_max = 5
    torch.manual_seed(42)
    if tfboard:
        writer = SummaryWriter('runs/fb15k')
    trainset, _ = fetch_20newsgroups(data_home='20_newsgroups',
                                                      subset='train', categories=None,
                                        shuffle=True, random_state=42, remove=(),
                                        download_if_missing=True, return_X_y=True)
    testset, _ = fetch_20newsgroups(data_home='20_newsgroups',
                                                     subset='test', categories=None,
                                        shuffle=True, random_state=42, remove=(),
                                        download_if_missing=True, return_X_y=True)
    stopwords = frozenset(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                           "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
                           "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
                           "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
                           "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                           "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
                           "with", "about", "against", "between", "into", "through", "during", "before", "after",
                           "above",
                           "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                           "further",
                           "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
                           "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                           "so",
                           "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

    dfreq = defaultdict(int)
    new_trainset = []
    vectorizer = CountVectorizer(analyzer='word')
    analyzer = vectorizer.build_analyzer()
    for i in range(len(trainset)):
        doc = analyzer(trainset[i])
        for word in np.unique(doc):
            dfreq[word] += 1
        new_trainset.append(doc)
    for i in range(len(testset)):
        doc = analyzer(testset[i])
        for word in np.unique(doc):
            dfreq[word] += 1
        new_trainset.append(doc)

    useless_words = frozenset([w for w in dfreq if dfreq[w] < 2])
    for i in range(len(new_trainset)):
        new_trainset[i] = list(filter(lambda w: w not in useless_words, new_trainset[i]))
    trainset = []
    voc = []
    for doc in new_trainset:
        d = []
        for word in doc:
            if word not in stopwords:
                d.append(word)
                voc.append(word)
        trainset.append(d)
    freq = defaultdict(int)
    for i in range(len(trainset)):
        for word in trainset[i]:
            freq[word] += 1
    voc = frozenset(voc)
    voc = dict(zip(voc, range(len(voc))))
    s_set = [[] for _ in range(window_max)]
    d_set = [[] for _ in range(window_max)]
    words_freq = np.zeros(len(voc))
    for w in voc:
        words_freq[voc[w]] = freq[w]
    words_freq /= np.sum(words_freq)
    for i in range(len(trainset)):
        doc = []
        for j in trainset[i]:
            w = voc[j]
            doc.append(w)
        window_fin = min(len(doc), window_max)
        for window_size in range(window_fin):
            for j in range(len(doc) - window_size):
                a = doc[j: j + window_size + 1][:]
                # a += [len(voc)] * (window_max - len(a))
                d_set[window_size].append(i)
                s_set[window_size].append(a)
    words_entropy = 0
    for p in words_freq:
        if p > 0:
            words_entropy -= p * np.log(p)
    print(words_entropy)
    print(len(trainset))
    print(len(words_freq))
    with codecs.open('20ns_params', mode='w', encoding='ascii') as f:
        f.write('Words Entropy: ' + str(words_entropy) + '\n')
        f.write('Num docs: ' + str(len(trainset)) + '\n')
        f.write('Num words: ' + str(len(words_freq)) + '\n')
    for window_size in range(window_max):
        with codecs.open('20ns' + str(2 + window_size), mode='w', encoding='ascii') as f:
            for i in range(len(d_set[window_size])):
                f.write(str(d_set[window_size][i]))
                for s in s_set[window_size][i]:
                    f.write('-' + str(s))
                f.write('\n')


if __name__ == '__main__':
    train()
