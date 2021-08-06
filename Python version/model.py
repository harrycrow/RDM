import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _special_orthogonal(tensor):
    n = tensor.size(0)
    flattened = tensor.new(n, n).normal_(0.0, 1.0 / n)
    with torch.no_grad():
        flattened = torch.eye(n) + 0.5 * (flattened - flattened.t())
        # Compute the qr factorization
        q, r = torch.qr(flattened)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        tensor.view_as(q).copy_(q)
    return tensor


class RDM(nn.Module):
    """
    Rotation Document Model
    """

    def __init__(self, n_words, n_docs, kappa_init, dim, window_max, sparse):
        super(RDM, self).__init__()
        self.n_words = n_words
        self.n_docs = n_docs
        self.dim = dim
        self.docs = nn.Embedding(n_docs, dim * dim, sparse=sparse)
        self.words = nn.Embedding(n_words + 1, dim * dim, sparse=sparse, padding_idx=n_words)
        self.eye = torch.eye(dim, requires_grad=False)
        self.kappa_init = kappa_init / dim
        self.kappa = nn.Parameter(torch.zeros((window_max)))

    def init(self):
        torch.nn.init.uniform_(self.kappa, 0.9*self.kappa_init, 1.1 * self.kappa_init)
        for i in range(self.n_docs):
            a = self.docs.weight.data[i].view(self.dim, self.dim)
            _special_orthogonal(a)
        self.words.weight.data[self.n_words].view_as(self.eye).copy_(self.eye)
        for i in range(self.n_words):
            a = self.words.weight.data[i].view(self.dim, self.dim)
            _special_orthogonal(a)

    def forward(self, d, w):
        n = w.shape[1] - 1
        context = self.words(w[:, 0]).view(-1, self.dim, self.dim)
        for i in range(1, w.shape[1], 1):
            words = self.words(w[:, i]).view(-1, self.dim, self.dim)
            context = torch.bmm(context, words)
        context = context.view(-1, self.dim * self.dim)
        doc = self.docs(d).view(-1, self.dim * self.dim)
        out = torch.mm(doc, context.t())
        return out * self.kappa[n]

    def forward_r(self, d, w):
        context = self.words(w[:, 0]).view(-1, self.dim, self.dim)
        for i in range(1, w.shape[1], 1):
            words = self.words(w[:, i]).view(-1, self.dim, self.dim)
            context = torch.bmm(context, words)
        context = context.view(-1, self.dim * self.dim)
        doc = self.docs(d).view(-1, self.dim * self.dim)
        out = torch.mm(doc, context.t())
        return out

    def get_embedding(self):
        docs_emb = self.docs.weight.detach().cpu().numpy()
        words_emb = self.words.weight.detach().cpu().numpy()[:-1]
        return docs_emb, words_emb

    def load(self, emb_path='embedding.vec.json'):
        embeddings = None
        with open(emb_path, 'r') as f:
            embeddings = json.loads(f.read())
        docs_emb = torch.tensor(np.array(embeddings['docs_embeddings']))
        words_emb = torch.cat((torch.tensor(np.array(embeddings['words_embeddings'])),
                              torch.eye(self.dim).view(-1)))
        self.docs = nn.Embedding.from_pretrained(docs_emb)
        self.words = nn.Embedding.from_pretrained(words_emb)
