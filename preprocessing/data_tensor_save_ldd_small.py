import tqdm
import pandas as pd
import os
import codecs
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from model import *
from sklearn.feature_extraction.text import CountVectorizer


def translate_triple(dataset):
    ent2id = []
    rel2id = []
    with open(dataset+'/entities.txt') as f:
        for i, line in enumerate(f):
            ent2id.append(line.split()[0].strip())
    with open(dataset+'/relations.txt') as f:
        for i, line in enumerate(f):
            rel2id.append(line.split()[0].strip())
    return ent2id, rel2id

def train(tfboard=False):
    min_count = 5
    skip = 1e-5  # 0.001
    window_max = 1
    torch.manual_seed(42)
    if tfboard:
        writer = SummaryWriter('runs/fb15k')

    basepath = '/Users/igorvorona/Desktop/Long document dataset'
    labels = """cs.cv
    cs.CE
    math.ST
    math.GR
    math.AC
    cs.SY
    cs.PL
    cs.NE
    cs.IT
    cs.DS
    cs.AI"""
    labels = [l.strip() for l in labels.split('\n')]
    labels = {k: v for v, k in enumerate(labels)}
    pbar = tqdm.tqdm(range(33388))
    df = pd.DataFrame()
    for l in labels:
        path = os.path.join(basepath, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
    df.columns = ['text', 'cls']



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
    cls = []
    pbar = tqdm.tqdm(range(len(df) // 1000))
    for i in range(len(df) // 2):
        doc = analyzer(df.loc[i, 'text'])
        cls.append(df.loc[i, 'cls'])
        if i % 1000 == 0:
            pbar.update()
        for word in np.unique(doc):
            dfreq[word] += 1
        new_trainset.append(doc)

    useless_words = frozenset([w for w in dfreq if dfreq[w] < 2])
    trainset = []
    pbar = tqdm.tqdm(range(len(new_trainset) // 1000))
    dfreq = defaultdict(int)
    deleted = []
    for i, doc in enumerate(new_trainset):
        if i % 1000 == 0:
            pbar.update()
        d = list(filter(lambda w: w not in useless_words, doc))
        d = [word for word in d if word not in stopwords]
        if len(d) < 1000:
            deleted.append(i)
            continue
        if len(d) > 5000:
            d = d[:5000]
        for word in np.unique(d):
            dfreq[word] += 1
        trainset.append(d)
    k = 0
    while k == 0 or len(deleted) > 0:
        deleted = frozenset(deleted)
        cls = [c for i, c in enumerate(cls) if i not in deleted]
        useless_words = frozenset([w for w in dfreq if dfreq[w] < 2])
        pbar = tqdm.tqdm(range(len(trainset) // 1000))
        new_trainset = []
        doc_len = 0
        dfreq = defaultdict(int)
        deleted = []
        for i in range(len(trainset)):
            if i % 1000 == 0:
                pbar.update()
            d = list(filter(lambda w: w not in useless_words, trainset[i]))
            if len(d) < 1000:
                deleted.append(i)
                continue
            for word in np.unique(d):
                dfreq[word] += 1
            new_trainset.append(d)
            doc_len += len(d)
        doc_len /= len(trainset)
        trainset = new_trainset
        k += 1
    freq = defaultdict(int)
    for i in range(len(trainset)):
        if i % 1000 == 0:
            pbar.update()
        for word in trainset[i]:
            freq[word] += 1
    voc = frozenset(freq.keys())
    voc = dict(zip(voc, range(len(voc))))
    s_set = [[] for _ in range(window_max)]
    d_set = [[] for _ in range(window_max)]
    words_freq = np.zeros(len(voc))
    for w in voc:
        words_freq[voc[w]] = freq[w]
    words_freq /= np.sum(words_freq)
    pbar = tqdm.tqdm(range(len(trainset) // 1000))
    for i in range(len(trainset)):
        if i % 1000 == 0:
            pbar.update()
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
    with codecs.open('data_tensors/ldd_small_params', mode='w', encoding='utf-8') as f:
        f.write('Words Entropy: ' + str(words_entropy) + '\n')
        f.write('Num docs: ' + str(len(trainset)) + '\n')
        f.write('Num words: ' + str(len(words_freq)) + '\n')
        cl, counts = np.unique(cls, return_counts=True)
        for i in range(len(cl)):
            f.write('{} {}\n'.format(cl[i], counts[i]))
        f.write('Avg doc len: ' + str(doc_len) + '\n')

    with codecs.open('data_tensors/ldd_small_freq', mode='w', encoding='utf-8') as f:
        for word in freq:
            f.write('{} {}\n'.format(word, freq[word]))

    with codecs.open('data_tensors/ldd_small', mode='w', encoding='utf-8') as f:
        for i in range(len(trainset)):
            for word in trainset[i]:
                f.write('{} '.format(word))
            f.write('\n')

    with codecs.open('data_tensors/ldd_small_cls', mode='w', encoding='ascii') as f:
        for i in cls:
            f.write('{}\n'.format(i))

    for window_size in range(window_max):
        with codecs.open('data_tensors/ldd_small' + str(2 + window_size), mode='w', encoding='ascii') as f:
            for i in range(len(d_set[window_size])):
                f.write(str(d_set[window_size][i]))
                for s in s_set[window_size][i]:
                    f.write('-' + str(s))
                f.write('\n')


if __name__ == '__main__':
    train()