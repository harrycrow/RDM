#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "mkl.h"
#include "mkl_cblas.h"
#include "mkl_blas.h"
#include "mkl_lapack.h"
#include "mkl_lapacke.h"

typedef struct {
    double *weights;
    double *first_moments;
    double *second_moments;
    double *buf;
    double *cdf;
    int *step;
} Embedding;

typedef struct {
    int num_words;
    int num_docs;
    int max_order;
    int *num_elem;
    int **coords;
} Tensors;

uint64_t prng_state;


void prng_init(uint64_t seed) {
    // initialize xorshift64*
    prng_state = seed;
}

uint64_t prng_next() {
    // xorshift64*
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * UINT64_C(2685821657736338717);
}

int uniform_int(int low, int high) {
    return low + (prng_next() % (high - low));
}

double uniform_double() {
    return (prng_next() >> 11) * 0x1.0p-53;
}

double gaussian(double mu, double sigma) {
    // Sample from random normal distribution N(mu, sigma)
    // using Box–Muller transform

    double x, y, r2, z;

    do {

        /* choose x,y in uniform square (-1,-1) to (+1,+1) */
        x = -1 + 2 * uniform_double();
        y = -1 + 2 * uniform_double();

        /* see if it is in the unit circle */
        r2 = x * x + y * y;

    } while (r2 >= 1.0 || r2 == 0);

    z = y * sqrt (-2.0 * log (r2) / r2);

    return z * sigma + mu;
}

void print_mat(double *mat, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            printf("%lf ", *(mat + i * dim + j));
        }
        printf("\n");
    }
}

void transpose(double *res, double *mat, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            res[i * dim + j] = mat[j * dim + i];
        }
    }
}

void mat_mul_t(double *res, double *mat1, double *mat2, int dim) {
    // R = M1 * M2^T
    int u, v;
    int mat_size = dim * dim;
    for (int i = 0; i < mat_size; i += dim) {
        for (int j = 0; j < dim; ++j) {
            u = i + j;
            v = j * dim;
            res[u] = 0;
            for (int k = 0; k < dim; ++k) {
                res[u] += mat1[i + k] * mat2[v + k];
            }
        }
    }
}

void zero(double *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0;
    }
}

void normalize(double *data, int size) {
    double norm = 0;
    for (int i = 0; i < size; ++i) {
        norm += data[i] * data[i];
    }
    norm = sqrt(norm) + 1e-12;
    for (int i = 0; i < size; ++i) {
        data[i] /= norm;
    }
}

double norm(double *data, int size) {
    double norm = 0;
    for (int i = 0; i < size; ++i) {
        norm += data[i] * data[i];
    }
    return sqrt(norm);
}


void my_orth(double* a, double *b, int emb_dim) {
    //orthogonalization via modified gram-schmidt
    transpose(b, a, emb_dim);
    for (int i = 0; i < emb_dim; ++i) {
        double *b_i = b + i * emb_dim;
        normalize(b_i, emb_dim);
        for (int j = i + 1; j < emb_dim; ++j) {
            double *b_j = b + j * emb_dim;
            double r = 0;
            for (int k = 0; k < emb_dim; ++k) {
                r += b_i[k] * b_j[k];
            }
            for (int k = 0; k < emb_dim; ++k) {
                b_j[k] = b_j[k] - r * b_i[k];
            }
        }
    }
    transpose(a, b, emb_dim);
}

void orth(double* a, double *b, int dim) {
    //orthogonalization via Householder reflection
    double *c = b + dim * dim;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim, dim, a, dim, c);
    // Copy the upper triangular Matrix R (rank x _n) into position
    for(int row = 0; row < dim; ++row) {
        memset(b+row*dim, 0, row*sizeof(double)); // Set starting zeros
        memcpy(b+row*dim+row, a+row*dim+row, (dim-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
    }
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dim, dim, dim, a, dim, c);
    for(int row = 0; row < dim; ++row) {
        c[row] = (b[row * dim + row] >= 0) ? 1 : -1;
    }
    for(int i = 0; i < dim; ++i) {
        for(int j = 0; j < dim; ++j) {
            a[i * dim + j] *= c[j];
        }
    }
}

void eye(double* mat, int dim) {
    int emb_size = dim * dim;
    cblas_dscal(emb_size, 0, mat, 1);
    for (int i = 0; i < dim; ++i) {
        mat[i*dim + i] = 1;
    }
}

double trace(double* mat, int dim) {
    double res = 0;
    for (int i = 0; i < dim; ++i) {
        res += mat[i * dim + i];
    }
    return res;
}

double dot(double* vec1, double *vec2, int size) {
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

void copy(double* vec1, double *vec2, int size) {
    for (int i = 0; i < size; ++i) {
        vec1[i] = vec2[i];
    }
}

void add(double* vec1, double *vec2, double c, int size) {
    for (int i = 0; i < size; ++i) {
        vec1[i] += c * vec2[i];
    }
}

void orth_init(double* mat, double *buf, int n_mats, int emb_dim) {
    int mat_size = emb_dim * emb_dim;
    for (int i = 0; i < n_mats; ++i) {
        double *p = mat + i * mat_size;
        for (int j = 0; j < mat_size; ++j) {
            p[j] = gaussian(0, 1.0 / emb_dim);
        }
        transpose(buf, p, emb_dim);
        for (int j = 0; j < mat_size; ++j) {
            p[j] = 0.5 * (p[j] - buf[j]);
        }
        eye(buf, emb_dim);
        for (int j = 0; j < mat_size; ++j) {
            p[j] += buf[j];
        }
        orth(p, buf, emb_dim);
    }
}

void shuffle(int *s, int length) {
    int i, r, tmp;
    for (i = length - 1; i >= 1; i--) {
        r = (int) (uniform_double() * (i + 1));
        tmp = s[i];
        s[i] = s[r];
        s[r] = tmp;
    }
}

void load_tensor(Tensors *dataset, int order, const char *path) {
    FILE* f = fopen(path, "r");
    int capacity = 1024;
    int num_elem = 0;
    int *coords = malloc(capacity * order * sizeof(int));
    char *s = calloc(33 * order + 1, sizeof(char));
    for(; fscanf(f, "%s", s) != EOF; ++num_elem) {
        // Work with capacity
        if (capacity - num_elem == 0) {
            capacity *= 2;
            coords = realloc(coords, capacity * order * sizeof(int));
        }
        int k = 0;
        int l = 0;
        for(int i = 0; i < strlen(s) + 1; ++i) {
            if (i == strlen(s) || s[i] == '-') {
                int val = 0;
                int dec = 1;
                for(int j = 1; i - j >= k; ++j) {
                    val += ((int) s[i - j] - '0') * dec;
                    dec *= 10;
                }
                coords[order * num_elem + l] = val;
                k = i + 1;
                ++l;
            }
        }
    }
    coords = realloc(coords, num_elem * order * sizeof(int));
    dataset->coords[order - 2] = coords;
    dataset->num_elem[order - 2] = num_elem;
    free(s);
    fclose(f);
}

void save_bin(double *mat, int dim1, int dim2, const char *path) {
    FILE* f = fopen(path, "w");
    for (int i = 0; i < dim1; ++i) {
        fwrite(mat + i * dim2, sizeof(double), dim2, f);
    }
    fclose(f);
}

void save_txt(double *mat, int dim1, int dim2, const char *sep, const char *path) {
    FILE* f = fopen(path, "w");
    for (int i = 0; i < dim1; ++i) {
        fprintf(f, "%d%s", i, sep);
        for (int j = 0; j < dim2 - 1; ++j) {
            fprintf(f, "%lf%s", mat[i * dim2 + j], sep);
        }
        fprintf(f, "%lf\n", mat[i * dim2 + dim2 - 1]);
    }
    fclose(f);
}

void adagrad(Embedding model, int dim, int idx, double *agrad, double lr, double lam) {
    int size = dim * dim;
    double *weights = model.weights + idx * size;
    double *sm = model.second_moments + idx * size;
    double *grad = agrad + idx * size;
    double eps = 1e-8;
    if (lam > 0) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                int k = i * dim + j;
                double v = (i == j) ? weights[k] - 1 : weights[k];
                if (v > 0) {
                    v = 1;
                } else if (v < 0) {
                    v = -1;
                } else {
                    v = 0;
                }
                grad[k] = grad[k] + lam * v;
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        sm[i] = sm[i] + grad[i] * grad[i];
    }
    for (int i = 0; i < size; ++i) {
        weights[i] -= lr * grad[i] / (sqrt(sm[i]) + eps);
    }
}

void kappa_adagrad(double *kappa, double *sm, double grad, double lr, double lam) {
    double eps = 1e-8;
    if (lam > 0) {
        grad += lam * *kappa;
    }
    *sm += grad * grad;
    *kappa -= lr * grad / (sqrt(*sm) + eps);
    *kappa = (*kappa > 0) ? *kappa : 0;
}

void rsgd(Embedding model, int dim, int idx, double *agrad, double lr, double lam) {
    int size = dim * dim;
    double *weights = model.weights + idx * size;
    double *sm = model.second_moments + idx * size;
    double *grad = agrad + idx * size;
    double *buf1 = model.buf;
    double *buf2 = model.buf + size;
    double eps = 1e-8;
    if (lam > 0) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                int k = i * dim + j;
                double v = (i == j) ? weights[k] - 1 : weights[k];
                if (v > 0) {
                    v = 1;
                } else if (v < 0) {
                    v = -1;
                } else {
                    v = 0;
                }
                grad[k] = grad[k] + lam * v;
            }
        }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                dim,dim,dim,1.0,weights,dim,
                grad,dim,
                0.0,buf1,dim);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                dim,dim,dim,1.0,buf1,dim,
                weights,dim,
                0.0,buf2,dim);

    for (int i = 0; i < size; ++i) {
        grad[i] = grad[i] - buf2[i];
    }
    double denom = 0;
    for (int i = 0; i < size; ++i) {
        sm[i] += grad[i] * grad[i];
        denom += sm[i];
    }
    for (int i = 0; i < size; ++i) {
        weights[i] -= lr * grad[i] / (sqrt(denom) + eps);
    }
    orth(weights, buf1, dim);
}

int categorical_sample(double *cdf, int cdf_len) {
    double p = uniform_double();
    for(int j = 0; j < cdf_len; ++j) {
        if (p <= cdf[j]) {
            return j;
        }
    }
}

void optimize_model(Embedding words, Embedding docs, Tensors *data, FILE *verbose_file,
                    int emb_dim, int num_epochs, int batch_size, double lr_init,
                    double kappa_init, double lam, int orth_flag) {
    int *num_elem = data->num_elem;
    int num_words = data->num_words;
    int num_docs = data->num_docs;
    int max_order = data->max_order;
    int emb_size = emb_dim * emb_dim;
    int size_per_epoch = 0;
    for (int i = 0; i < max_order - 1; ++i) {
        size_per_epoch = (size_per_epoch < num_elem[i]) ? num_elem[i] : size_per_epoch;
    }
    double *words_weights = words.weights;
    double *docs_weights = docs.weights;
    double *words_cdf = words.cdf;
    double *docs_cdf = docs.cdf;
    orth_init(words_weights, words.buf, num_words, emb_dim);
    orth_init(docs_weights, docs.buf, num_docs, emb_dim);

    int *words_batch = malloc(batch_size * ((max_order - 1) * max_order / 2) * sizeof(int));
    int *docs_batch = malloc((max_order - 1) * batch_size * sizeof(int));
    int *upd_words = calloc(num_words, sizeof(int));
    int *upd_docs = calloc(num_docs, sizeof(int));
    double *mul_res = malloc(emb_size * sizeof(double));
    double *tmp_buf1 = malloc(emb_size * sizeof(double));
    double *tmp_buf2 = malloc(emb_size * sizeof(double));
    double *sum_docs = malloc(batch_size * emb_size * sizeof(double));
    double *words_grad = malloc(num_words * emb_size * sizeof(double));
    double *docs_grad = malloc(num_docs * emb_size * sizeof(double));
    double *local_docs_grad = malloc(batch_size * batch_size * emb_size * sizeof(double));
    double *scores = malloc(batch_size * batch_size * sizeof(double));
    double *logits = malloc(batch_size * batch_size * sizeof(double));
    double *kappa = malloc((max_order - 1) * sizeof(double));
    for (int i = 0; i < max_order-1; ++i) {
        if (orth_flag == 1) {
            kappa[i] = kappa_init * (2 * uniform_double() + 9) / 10;
        } else {
            kappa[i] = 1.0;
        }
    }
    double *kappa_grad = calloc(max_order - 1, sizeof(double));
    double *sm_kappa_grad = calloc(max_order - 1, sizeof(double));

    if (verbose_file != NULL) {
        fprintf(verbose_file, "\n");
        fprintf(verbose_file, "===================================================");
        fprintf(verbose_file, "===================================================");
        fprintf(verbose_file, "\n");
        fprintf(verbose_file, "Optimization Start\n");
        fprintf(verbose_file, "Number of words: %d\n", num_words);
        fprintf(verbose_file, "Number of docs: %d\n", num_docs);
        for (int i = 2; i <= max_order; ++i) {
            double tensor_size = (double) num_docs;
            for (int j = 1; j < i; ++j) {
                tensor_size *= (double) num_words;
            }
            double tensor_density =  num_elem[i - 2] / tensor_size;

            fprintf(verbose_file, "Tensor %d Size: %d (x) %d x % d = %g\n", i - 1, num_words, i - 1, num_docs, tensor_size);
            fprintf(verbose_file, "Number of elements (nnz): %d\n", num_elem[i - 2]);
            fprintf(verbose_file, "Density of Tensor: %g\n", tensor_density);

        }
        fprintf(verbose_file, "Dimensions of embeddings: %d x %d\n", emb_dim, emb_dim);
        fprintf(verbose_file, "Memory Complexity: %d\n", num_words * emb_size + num_docs * emb_size);
        fprintf(verbose_file, "Number of epochs: %d\n", num_epochs);
        fprintf(verbose_file, "Size per epoch: %d\n", size_per_epoch);
        fprintf(verbose_file, "Batch size: %d\n", batch_size);
        fprintf(verbose_file, "Learning rate: %g\n", lr_init);
        fprintf(verbose_file, "Concentration (scaling hyperparameter): %g\n", kappa_init);
        fprintf(verbose_file, "Regularizer strength: %g\n", lam);
        fprintf(verbose_file, "Orthogonality: %d\n", orth_flag);
        fprintf(verbose_file, "===================================================");
        fprintf(verbose_file, "===================================================");
        fprintf(verbose_file, "\n");
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        double loss = 0;
        double lr = lr_init;// / 2 * (1 + cos(epoch / num_epochs * M_PI));
        for (int iter = 0; iter < size_per_epoch; iter += batch_size) {
            double local_loss = 0;
            int *db = docs_batch;
            int *wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                // Sample i.i.d batch of elements
                int wlen = order - 1;
                int *coords = data->coords[order - 2];
                for (int i = 0; i < batch_size; ++i) {
                    int idx = uniform_int(0, num_elem[order - 2]);
                    db[i] = coords[idx * order];
                    for (int j = 0; j < wlen; ++j) {
                        wb[i * wlen + j] = coords[idx * order + j + 1];
                    }
                }
                db += batch_size;
                wb += batch_size * wlen;
            }
            db = docs_batch;
            wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                int wlen = order - 1;
                for (int i = 0; i < batch_size; ++i) {
                    if (upd_docs[db[i]] == 0) {
                        cblas_dscal(emb_size, 0, docs_grad + db[i] * emb_size, 1);
                        upd_docs[db[i]] = 1;
                    }
                    for (int j = 0; j < wlen; ++j) {
                        if (upd_words[wb[i * wlen + j]] == 0) {
                            cblas_dscal(emb_size, 0, words_grad + wb[i * wlen + j] * emb_size, 1);
                            upd_words[wb[i * wlen + j]] = 1;
                        }
                    }
                }
                kappa_grad[order - 2] = 0;
                db += batch_size;
                wb += batch_size * wlen;
            }
            db = docs_batch;
            wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                int wlen = order - 1;
                for (int i = 0; i < batch_size; ++i) {
                    upd_docs[db[i]] = 0;
                    for (int j = 0; j < wlen; ++j) {
                        upd_words[wb[i * wlen + j]] = 0;
                    }
                }
                db += batch_size;
                wb += batch_size * wlen;
            }
            db = docs_batch;
            wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                int wlen = order - 1;
                // Docs Forward & Backward pass
                for (int i = 0; i < batch_size; ++i) {
                    eye(mul_res, emb_dim);
                    for (int j = 0; j < wlen; ++j) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    emb_dim, emb_dim, emb_dim, 1.0, mul_res, emb_dim,
                                    words_weights + wb[i * wlen + j] * emb_size, emb_dim,
                                    0.0, tmp_buf1, emb_dim);
                        cblas_dcopy(emb_size, tmp_buf1, 1, mul_res, 1);
                    }
                    for (int j = 0; j < batch_size; ++j) {
                        cblas_dcopy(emb_size, mul_res, 1,
                                    local_docs_grad + (j * batch_size + i) * emb_size, 1);
                        scores[j * batch_size + i] =
                                cblas_ddot(emb_size, mul_res,
                                           1, docs_weights + db[j] * emb_size, 1);
                        logits[j * batch_size + i] = kappa[order - 2] * scores[j * batch_size + i];

                    }
                }
                for (int i = 0; i < batch_size; ++i) {
                    cblas_dscal(emb_size, 0, sum_docs + i * emb_size, 1);
                }
                for (int i = 0; i < batch_size; ++i) {
                    double stability_bias = -INFINITY;
                    for (int j = 0; j < batch_size; ++j) {
                        stability_bias = (logits[i * batch_size + j] > stability_bias) ?
                                         logits[i * batch_size + j] : stability_bias;
                    }
                    double denom = 0;
                    for (int j = 0; j < batch_size; ++j) {
                        logits[i * batch_size + j] = exp(logits[i * batch_size + j] - stability_bias);
                        denom += logits[i * batch_size + j];
                    }
                    for (int j = 0; j < batch_size; ++j) {
                        logits[i * batch_size + j] /= denom;
                    }
                    local_loss += -log(logits[i * batch_size + i]) / batch_size;
                    double coeff = -1.0 / batch_size;
                    for (int j = 0; j < batch_size; ++j) {
                        double w = coeff;
                        if (i == j) {
                            w *= 1 - logits[i * batch_size + j];
                        } else {
                            w *= -logits[i * batch_size + j];
                        }
                        add(docs_grad + db[i] * emb_size,
                            local_docs_grad + (i * batch_size + j) * emb_size,
                            w * kappa[order - 2], emb_size);
                        add(sum_docs + j * emb_size,
                            docs_weights + db[i] * emb_size,
                            w * kappa[order - 2], emb_size);
                        kappa_grad[order - 2] += w * scores[i * batch_size + j];
                    }
                }
                // Words Forward & Backward pass
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 1; j < order; ++j) {
                        eye(mul_res, emb_dim);
                        for (int k = 1; k < order; ++k) {
                            int idx = (j + k) % order;
                            if (idx > 0) {
                                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                            emb_dim, emb_dim, emb_dim, 1.0, mul_res, emb_dim,
                                            words_weights + wb[i * wlen + idx - 1] * emb_size, emb_dim,
                                            0.0, tmp_buf1, emb_dim);
                                cblas_dcopy(emb_size, tmp_buf1, 1, mul_res, 1);
                            } else {
                                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                            emb_dim, emb_dim, emb_dim, 1.0, mul_res, emb_dim,
                                            sum_docs + i * emb_size, emb_dim,
                                            0.0, tmp_buf1, emb_dim);
                                cblas_dcopy(emb_size, tmp_buf1, 1, mul_res, 1);
                            }
                        }
                        mkl_dimatcopy('r', 't', emb_dim, emb_dim, 1, mul_res, emb_dim, emb_dim);
                        add(words_grad + wb[i * wlen + j - 1] * emb_size,
                            mul_res, 1, emb_size);
                    }
                }
                db += batch_size;
                wb += batch_size * wlen;
            }
            db = docs_batch;
            wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                int wlen = order - 1;
                for (int i = 0; i < batch_size; ++i) {
                    if (upd_docs[db[i]] == 0) {
                        if (orth_flag == 1) {
                            rsgd(docs, emb_dim, db[i], docs_grad, lr, 0);
                        } else {
                            adagrad(docs, emb_dim, db[i], docs_grad, lr, 0);
                        }
                        upd_docs[db[i]] = 1;
                    }
                    for (int j = 0; j < wlen; ++j) {
                        if (upd_words[wb[i * wlen + j]] == 0) {
                            if (orth_flag == 1) {
                                rsgd(words, emb_dim, wb[i * wlen + j], words_grad, lr, 0);
                            } else {
                                adagrad(words, emb_dim, wb[i * wlen + j], words_grad, lr, 0);
                            }
                            upd_words[wb[i * wlen + j]] = 1;
                        }
                    }
                }
                if (orth_flag == 1) {
                    kappa_adagrad(kappa + order - 2, sm_kappa_grad + order - 2,
                                  kappa_grad[order - 2], 1e-3, lam);
                }
                db += batch_size;
                wb += batch_size * wlen;
            }
            db = docs_batch;
            wb = words_batch;
            for (int order = 2; order <= max_order; ++order) {
                int wlen = order - 1;
                for (int i = 0; i < batch_size; ++i) {
                    upd_docs[db[i]] = 0;
                    for (int j = 0; j < wlen; ++j) {
                        upd_words[wb[i * wlen + j]] = 0;
                    }
                }
                db += batch_size;
                wb += batch_size * wlen;
            }
            loss += local_loss;
            if (verbose_file != NULL && iter % (batch_size * 1000) == 0) {
                fprintf(verbose_file,
                        "Epoch %d / %d: %d / %d Loss = %.3lf\n", epoch + 1, num_epochs,
                        iter, size_per_epoch, local_loss);
                for (int order = 2; order <= max_order; ++order) {
                    fprintf(verbose_file,
                            "Kappa (%d) = %.3lf ", order, kappa[order - 2]);
                }
                fprintf(verbose_file, "\n");
            }
        }
        if (verbose_file != NULL) {
            fprintf(verbose_file, "===================================================");
            fprintf(verbose_file, "===================================================");
            fprintf(verbose_file, "\n");
            fprintf(verbose_file, "Summary of Epoch %d / %d: \n", epoch + 1, num_epochs);
            fprintf(verbose_file, "[SUM] Loss = %.3lf\n", loss);
            fprintf(verbose_file, "[MEAN] Loss = %.3lf\n", loss / size_per_epoch * batch_size);
            fprintf(verbose_file, "===================================================");
            fprintf(verbose_file, "===================================================");
            fprintf(verbose_file, "\n\n");
        }
    }
    free(words_batch);
    free(docs_batch);
    free(mul_res);
    free(tmp_buf1);
    free(tmp_buf2);
    free(sum_docs);
    free(words_grad);
    free(docs_grad);
    free(local_docs_grad);
    free(logits);
    free(scores);
    free(upd_docs);
    free(upd_words);
    free(kappa);
    free(kappa_grad);
    free(sm_kappa_grad);
}

int main(int argc, const char *argv[]) {
    if (argc < 12) {
        printf("Not correct number of parameters: %d / 10\n", argc - 1);
        printf("Specify:\n"
               "1) dataset path;\n2) number of words;\n3) number of documents;\n"
               "4) max order of tensor;\n5) number of epoch;\n6) batch size;\n7) embedding dimension;\n"
               "8) learning rate;\n9) softmax temperature;\n10) regularizer strength;\n11) orthogonality (1/0).\n");
        exit(1);
    }
    prng_init( 42);
    Tensors data;
    data.num_words = atoi(argv[2]);
    data.num_docs = atoi(argv[3]);
    data.max_order = atoi(argv[4]);
    data.num_elem = calloc(data.max_order - 1, sizeof(int));
    data.coords = calloc(data.max_order - 1, sizeof(int*));
    char *path = calloc(strlen(argv[1])+2, sizeof(char));
    path[0] = '\0';
    strcat(path,argv[1]);
    path[strlen(path)] = '0';
    path[strlen(path)] = '\0';
    for (int i = 2; i <= data.max_order; ++i) {
        path[strlen(path) - 1] = '0' + (char) i;
        load_tensor(&data, i, path);
    }
    int num_epochs = atoi(argv[5]);
    int batch_size = atoi(argv[6]);
    int emb_dim = atoi(argv[7]);
    double lr = atof(argv[8]);
    double kappa = atof(argv[9]) / emb_dim;
    double lam = atof(argv[10]);
    int orth_flag = atoi(argv[11]);
    int emb_size = emb_dim * emb_dim;
    Embedding words;
    words.weights = calloc(data.num_words * emb_size, sizeof(double));
    words.first_moments = calloc(data.num_words * emb_size, sizeof(double));
    words.second_moments = calloc(data.num_words * emb_size, sizeof(double));
    words.buf = calloc(3 * emb_size, sizeof(double));
    words.step = calloc(data.num_words, sizeof(int));

    Embedding docs;
    docs.weights = calloc(data.num_docs * emb_size, sizeof(double));
    docs.first_moments = calloc(data.num_docs * emb_size, sizeof(double));
    docs.second_moments = calloc(data.num_docs * emb_size, sizeof(double));
    docs.buf = calloc(3 * emb_size, sizeof(double));
    docs.step = calloc(data.num_docs, sizeof(int));

    optimize_model(words, docs, &data, stdout, emb_dim, num_epochs, batch_size, lr, kappa, lam, orth_flag);

    save_bin(words.weights, data.num_words,  emb_size, "./words.dat");
    save_bin(docs.weights, data.num_docs,  emb_size, "./docs.dat");

    save_txt(words.weights, data.num_words,  emb_size, "\t", "./words.tsv");
    save_txt(docs.weights, data.num_docs,  emb_size, "\t", "./docs.tsv");

    for (int i = 0; i < data.max_order - 1; ++i) {
        free(data.coords[i]);
    }

    free(words.weights);
    free(words.first_moments);
    free(words.second_moments);
    free(words.buf);
    free(words.step);

    free(docs.weights);
    free(docs.first_moments);
    free(docs.second_moments);
    free(docs.buf);
    free(docs.step);

    return 0;
}
