
import tqdm
import time
import codecs
import math
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, ConcatDataset
from optim import ProjAdagrad, Radagrad
from model import *


def train():
    dim = 20
    window_max = 3
    batch_size = 128
    n_epochs = 15
    n_words = 75752
    n_docs = 18846
    kappa_init = 1
    path = '20ns'
    torch.manual_seed(42)

    d_set = [[] for _ in range(window_max)]
    w_set = [[] for _ in range(window_max)]

    for i in range(window_max):
        with codecs.open(path + str(i+2), mode='r') as f:
            for s in f:
                a = s.split('-')
                d_set[i].append(int(a[0]))
                idx = []
                for j in range(1, i+2, 1):
                    idx.append(int(a[j]))
                w_set[i].append(idx)

    model = RDM(n_words=n_words, n_docs=n_docs, kappa_init=kappa_init,
                dim=dim, window_max=window_max, sparse=True)
    model.init()
    # model = model.cuda()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer_weights = Radagrad(params[1:], dim=dim, lr=2.0, lap_reg=0)
    #optimizer_weights = torch.optim.Adagrad(params[1:], lr=1e-2)
    optimizer_kappa = ProjAdagrad([params[0]], lr=1e-3, weight_decay=15)
    loaders = []
    max_size = 0
    for i in range(window_max):
        max_size = max(max_size, len(d_set[i]))
    max_size = math.ceil(max_size / batch_size)
    for i in range(window_max):
        dataset = TensorDataset(torch.tensor(d_set[i]), torch.tensor(w_set[i]))
        loaders.append(DataLoader(dataset, batch_size, shuffle=False,
                                  sampler=RandomSampler(dataset, replacement=True,
                                                        num_samples=max_size * batch_size)))

    del d_set
    del w_set
    nll = torch.nn.CrossEntropyLoss(reduction='mean')
    target_idx = torch.tensor(list(range(batch_size)), dtype=torch.long)
    print("Starting training...")
    for epoch in range(n_epochs):
        start_train = time.time()
        model.train()
        sum_losses = 0
        progbar = tqdm.tqdm(range(max_size))
        trainers = []
        for i in range(window_max):
            trainers.append(iter(loaders[i]))
        for _ in progbar:
            optimizer_weights.zero_grad()
            optimizer_kappa.zero_grad()
            full_loss = 0
            for i in range(window_max):
                d, w = trainers[i].next()
                loss = nll(model.forward(d, w), target_idx)
                full_loss += loss.detach().item()
                loss.backward()
            sum_losses += full_loss
            progbar.set_description('Epoch {}/{}'.format(epoch + 1, n_epochs))
            s = 'loss = {:.3} '
            kappa = []
            for i in range(window_max):
                s += 'kappa (' + str(i) + ') = {:.3} '
                kappa.append(params[0][i].detach().item())
            progbar.set_postfix_str(s=s.format(full_loss, *kappa))
            optimizer_weights.step()
            optimizer_kappa.step()
        print('Working time: {}'.format(time.time() - start_train))
        print('[TRAIN] Sum Loss: loss = {}'.format(sum_losses))
        print('[TRAIN] Mean Loss: loss = {}'.format(sum_losses / max_size))
    docs_emb, words_emb = model.get_embedding()
    print('Start Saving Embeddings')
    with codecs.open('embedding.vec.json', mode='w') as f:
        json.dump({'docs_embeddings': docs_emb.tolist(), 'words_embeddings': words_emb.tolist()}, f)
    with codecs.open('docs_emb2.tsv', mode='w') as f:
        for i in range(docs_emb.shape[0]):
            f.write(str(i))
            for j in range(docs_emb.shape[1]):
                f.write('\t' + str(docs_emb[i, j]))
            f.write('\n')
    with codecs.open('words_emb2.tsv', mode='w') as f:
        for i in range(words_emb.shape[0]):
            f.write(str(i))
            for j in range(words_emb.shape[1]):
                f.write('\t' + str(words_emb[i, j]))
            f.write('\n')

if __name__ == '__main__':
    train()
