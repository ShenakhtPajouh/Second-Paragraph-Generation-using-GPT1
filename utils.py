import numpy as np
import tensorflow as tf
from text_utils import TextEncoder
import pickle
import math
import pandas as pd
import logging
from random import shuffle


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


class Logger(object):
    def __init__(self, path):
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self._logger = logging.getLogger('trainlogger')
        self._logger.info('Train-Logger started ...')

    def log(self, **kwargs):
        # print(kwargs)
        self._logger.info(kwargs)


def get_paragraphs():
    paragraphs = pd.read_csv('./Data/prediction_train.tsv', sep='\t', encoding='latin1')
    p = paragraphs['paragraph_text_without_last_sentence'] + paragraphs['paragraph_last_sentence']
    return list(p.dropna())


def encode_dataset(n_ctx=512, n_vocab=40478, n_special=1, n_cut=256):
    '''
        return list of paragraphs with their respective masks
        IDs:
            0 : n_vocab -> vocab
            n_vocab : n_vocab + n_special -> special tokens
            n_vocab + n_special : n_vocab + n_special + n_ctx -> positions
            n_vocab + n_special + n_ctx : n_vocab + n_special + n_ctx + n_segment -> segments
    '''
    with open('Data/tokens.pkl', 'rb') as pkl:
        tokens = pickle.load(pkl)

    triple_pars_list = []
    tokens_masks_list = []
    preds_masks_list = []

    for i in range(len(tokens) - 2):
        fst = tokens[i][-(n_cut - 1):]
        snd = tokens[i + 1][:n_ctx]
        trd = tokens[i + 2][:n_cut - 1]
        a = np.zeros((len(fst) + len(snd) + len(trd) + 2, 3), dtype=np.int32)
        tm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)
        pm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)

        a[: len(fst), 0] = fst
        a[len(fst), 0] = n_vocab
        a[1 + len(fst): 1 + len(fst) + len(snd), 0] = snd
        a[1 + len(fst) + len(snd), 0] = n_vocab
        a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 0] = trd
        tm[1: 2 + len(fst) + len(snd) + len(trd)] = 1
        pm[: 1 + len(fst) + len(snd) + len(trd)] = 1

        a[: 1 + len(fst), 1] = np.arange(n_vocab + n_special, n_vocab + n_special + len(fst) + 1)
        a[1 + len(fst): 2 + len(fst) + len(snd), 1] = np.arange(n_vocab + n_special, n_vocab
                                                                + n_special + len(snd) + 1)
        a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 1] = \
            np.arange(n_vocab + n_special, n_vocab + n_special + len(trd))

        a[: 1 + len(fst), 2] = n_vocab + n_ctx + n_special
        a[1 + len(fst): 2 + len(fst) + len(snd), 2] = n_vocab + n_ctx + n_special + 1
        a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 2] = n_vocab + n_ctx + n_special + 2
        triple_pars_list.append(a)
        tokens_masks_list.append(tm)
        preds_masks_list.append(pm)

    with open('Data/triple_pars_list.pkl', 'wb') as pkl:
        pickle.dump(triple_pars_list, pkl)

    with open('Data/tokens_masks.pkl', 'wb') as pkl:
        pickle.dump(tokens_masks_list, pkl)

    with open('Data/preds_masks.pkl', 'wb') as pkl:
        pickle.dump(preds_masks_list, pkl)


def reform_dataset_for_segment(path = "Data/Segment/", n_ctx=512, n_vocab=40478, n_special=1, n_cut=256):
    '''
        return list of paragraphs with their respective masks
        IDs:
            0 : n_vocab -> vocab
            n_vocab : n_vocab + n_special -> special tokens
            n_vocab + n_special : n_vocab + n_special + n_ctx -> positions
            n_vocab + n_special + n_ctx : n_vocab + n_special + n_ctx + n_segment -> segments
    '''

    print("Reloading ...")

    with open('Data/encoded_triples.pkl', 'rb') as pkl:
        books = pickle.load(pkl)

    print("Reloaded!")

    i = 0
    while bool(books):
        triple_pars_list = []
        tokens_masks_list = []
        preds_masks_list = []
        print("\n Bucket {}".format(i + 1))

        j = 0
        for bookid in list(books):

            if j % 50 == 0:
                print("Book Number {} and len {}".format(j + 1, len(books.keys())))
            j += 1

            pars = books[bookid]
            for tokens in pars:
                fst = tokens[0][-(n_cut - 1):]
                snd = tokens[1][:n_ctx]
                trd = tokens[2][:n_cut - 1]
                a = np.zeros((len(fst) + len(snd) + len(trd) + 2, 3), dtype=np.int32)
                tm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)
                pm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)

                a[: len(fst), 0] = fst
                a[len(fst), 0] = n_vocab
                a[1 + len(fst): 1 + len(fst) + len(snd), 0] = snd
                a[1 + len(fst) + len(snd), 0] = n_vocab
                a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 0] = trd
                tm[1: 2 + len(fst) + len(snd) + len(trd)] = 1
                pm[: 1 + len(fst) + len(snd) + len(trd)] = 1

                a[: 1 + len(fst), 1] = np.arange(n_vocab + n_special, n_vocab + n_special + len(fst) + 1)
                a[1 + len(fst): 2 + len(fst) + len(snd), 1] = np.arange(n_vocab + n_special, n_vocab
                                                                        + n_special + len(snd) + 1)
                a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 1] = \
                    np.arange(n_vocab + n_special, n_vocab + n_special + len(trd))

                a[: 1 + len(fst), 2] = n_vocab + n_ctx + n_special
                a[1 + len(fst): 2 + len(fst) + len(snd), 2] = n_vocab + n_ctx + n_special + 1
                a[2 + len(fst) + len(snd): 2 + len(fst) + len(snd) + len(trd), 2] = n_vocab + n_ctx + n_special + 2
                triple_pars_list.append(a)
                tokens_masks_list.append(tm)
                preds_masks_list.append(pm)

            del books[bookid]

            if j == 100:
                break

        with open(path + 'tokens{}_masks.pkl'.format(i), 'wb') as pkl:
            pickle.dump(tokens_masks_list, pkl)
        del tokens_masks_list

        with open(path + 'preds{}_masks.pkl'.format(i), 'wb') as pkl:
            pickle.dump(preds_masks_list, pkl)
        del preds_masks_list

        with open(path + 'triple{}_pars_list.pkl'.format(i), 'wb') as pkl:
            pickle.dump(triple_pars_list, pkl)
        del triple_pars_list

        i += 1


def reform_dataset_for_model1(path = "Data/", n_ctx=512, n_vocab=40478, n_special=1, n_cut=256):
    '''
        return list of paragraphs with their respective masks
        IDs:
            0 : n_vocab -> vocab
            n_vocab : n_vocab + n_special -> special tokens
            n_vocab + n_special : n_vocab + n_special + n_ctx -> positions
            n_vocab + n_special + n_ctx : n_vocab + n_special + n_ctx + n_segment -> segments
    '''

    print("Reloading ...")

    with open('Data/encoded_triples.pkl', 'rb') as pkl:
        books = pickle.load(pkl)

    print("Reloaded!")

    i = 0
    while bool(books):
        triple_pars_list = []
        tokens_masks_list = []
        preds_masks_list = []
        metadata_list = []

        print("\n Bucket {}".format(i + 1))

        j = 0
        for bookid in list(books):

            if j % 100 == 0:
                print("Book Number {} and len {}".format(j + 1, len(books.keys())))
            j += 1

            pars = books[bookid]
            for k, tokens in enumerate(pars):
                fst = tokens[0][-(n_cut - 1):]
                snd = tokens[1][:n_ctx]
                trd = tokens[2][:n_cut - 1]
                a = np.zeros((len(fst) + len(snd) + len(trd) + 2, 3), dtype=np.int32)
                tm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)
                pm = np.zeros(len(fst) + len(snd) + len(trd) + 2, dtype=np.int32)

                a[: len(fst), 0] = fst
                a[len(fst), 0] = n_vocab
                a[1 + len(fst): 1 + len(fst) + len(trd), 0] = trd
                a[1 + len(fst) + len(trd), 0] = n_vocab
                a[2 + len(fst) + len(trd): 2 + len(fst) + len(trd) + len(snd), 0] = snd
                tm[2 + len(fst) + len(trd): 2 + len(fst) + len(trd) + len(snd)] = 1
                pm[1 + len(fst) + len(trd): 1 + len(fst) + len(trd) + len(snd)] = 1

                a[: 1 + len(fst), 1] = np.arange(n_vocab + n_special, n_vocab + n_special + len(fst) + 1)
                a[1 + len(fst): 2 + len(fst) + len(trd), 1] = np.arange(n_vocab + n_special, n_vocab
                                                                        + n_special + len(trd) + 1)
                a[2 + len(fst) + len(trd): 2 + len(fst) + len(snd) + len(trd), 1] = \
                    np.arange(n_vocab + n_special, n_vocab + n_special + len(snd))

                a[: 1 + len(fst), 2] = n_vocab + n_ctx + n_special
                a[1 + len(fst): 2 + len(fst) + len(trd), 2] = n_vocab + n_ctx + n_special + 2
                a[2 + len(fst) + len(trd): 2 + len(fst) + len(snd) + len(trd), 2] = n_vocab + n_ctx + n_special + 1
                triple_pars_list.append(a)
                tokens_masks_list.append(tm)
                preds_masks_list.append(pm)
                metadata_list.append((bookid, k, i))

            del books[bookid]

            if j == 500:
                break

        with open(path + 'metadata{}.pkl'.format(i), 'wb') as pkl:
            pickle.dump(metadata_list, pkl)
        del metadata_list

        with open(path + 'tokens{}_masks.pkl'.format(i), 'wb') as pkl:
            pickle.dump(tokens_masks_list, pkl)
        del tokens_masks_list

        with open(path + 'preds{}_masks.pkl'.format(i), 'wb') as pkl:
            pickle.dump(preds_masks_list, pkl)
        del preds_masks_list

        with open(path + 'triple{}_pars_list.pkl'.format(i), 'wb') as pkl:
            pickle.dump(triple_pars_list, pkl)
        del triple_pars_list

        i += 1


def encode(encoder=None):
    if encoder == None:
        ENCODER_PATH = 'model/encoder_bpe_40000.json'
        BPE_PATH = 'model/vocab_40000.bpe'
        encoder = TextEncoder(ENCODER_PATH, BPE_PATH)

    tokens = encoder(get_paragraphs(), verbose=False)
    with open('Data/tokens.pkl', 'wb') as pkl:
        pickle.dump(tokens, pkl)


def get_validation():
    with open('Data/tokens.pkl', 'rb') as pkl:
        tokens = pickle.load(pkl)
    with open('Data/masks.pkl', 'rb') as pkl:
        masks = pickle.load(pkl)

    n = len(tokens) // 10
    return tokens[-n:], masks[-n:]


def merge(n_batch, p, m1, m2, mt):
    max_len = max(list(map(lambda x: len(x), p)))
    tokens = np.zeros((n_batch, max_len, 3), dtype=np.int32)
    masks1 = np.zeros((n_batch, max_len), dtype=np.int32)
    masks2 = np.zeros((n_batch, max_len), dtype=np.int32)

    for j in range(n_batch):
        tokens[j, : len(p[j]), :] = p[j]
        masks1[j, : len(p[j])] = m1[j]
        masks2[j, : len(p[j])] = m2[j]

    return tokens, masks1, masks2, mt


def iter_data(n_batch, n_epochs=None, n_files=7, path="./Data/", train=True):
    if train:
        for epoch in range(n_epochs):
            for b in range(n_files-1):
                print(">>>>> File {}".format(b))

                with open(path + 'triple{}_pars_list.pkl'.format(b), 'rb') as pkl:
                    triple_pars_list = pickle.load(pkl)
                    

                with open(path + 'tokens{}_masks.pkl'.format(b), 'rb') as pkl:
                    tokens_mask_list = pickle.load(pkl)

                with open(path + 'preds{}_masks.pkl'.format(b), 'rb') as pkl:
                    preds_mask_list = pickle.load(pkl)

                with open(path + 'metadata{}.pkl'.format(b), 'rb') as pkl:
                    metadata_list = pickle.load(pkl)

                n = len(metadata_list)
                pmmm = list(zip(triple_pars_list, tokens_mask_list, preds_mask_list, metadata_list))
                shuffle(pmmm)
                triple_pars_list, tokens_mask_list, preds_mask_list, metadata_list = zip(*pmmm)

                for i in range(0, n, n_batch):
                    if i + n_batch > n:
                        break
                    m1 = tokens_mask_list[i: i + n_batch]
                    m2 = preds_mask_list[i: i + n_batch]
                    mt = metadata_list[i: i + n_batch]
                    p = triple_pars_list[i: i + n_batch]
                    yield merge(n_batch, p, m1, m2, mt)

    else:
        with open(path + 'triple{}_pars_list.pkl'.format(n_files - 1), 'rb') as pkl:
            triple_pars_list = pickle.load(pkl)

        with open(path + 'tokens{}_masks.pkl'.format(n_files - 1), 'rb') as pkl:
            tokens_mask_list = pickle.load(pkl)

        with open(path + 'preds{}_masks.pkl'.format(n_files - 1), 'rb') as pkl:
            preds_mask_list = pickle.load(pkl)

        with open('Data/metadata{}.pkl'.format(n_files - 1), 'rb') as pkl:
            metadata_list = pickle.load(pkl)

        n = 2 * len(metadata_list) // 3
        triple_pars_list = triple_pars_list[-n:]
        tokens_mask_list = tokens_mask_list[-n:]
        preds_mask_list = preds_mask_list[-n:]
        metadata_list = metadata_list[-n:]

        for i in range(0, n, n_batch):
            if i + n_batch > n:
                break

            m1 = tokens_mask_list[i: i + n_batch]
            m2 = preds_mask_list[i: i + n_batch]
            mt = metadata_list[i: i + n_batch]
            p = triple_pars_list[i: i + n_batch]
            yield merge(n_batch, p, m1, m2, mt)
            
def iter_data_m(n_batch, n_epochs=None, n_files=7, path="./Data/",train=True):
    if train:
        for epoch in range(n_epochs):
            for b in range(n_files):
                print(">>>>> File {}".format(b))

               
                with open(path + 'triple{}_pars_list.pkl'.format(b), 'rb') as pkl:
                    triple_pars_list = pickle.load(pkl)
                    
                max_len = max(list(map(lambda x: len(x), triple_pars_list)))
                print('max_len', max_len)
                        
                # print('...')
                # print(type(triple_pars_list[0]))
                # print(triple_pars_list[0].dtype)
                # print(triple_pars_list[0].reshape([-1]))
                # print('----')
                # st=triple_pars_list[0].tostring()
                # d=np.fromstring(st, dtype=np.int32)
                # print(np.reshape(d,[-1,3]))
            

                with open(path + 'tokens{}_masks.pkl'.format(b), 'rb') as pkl:
                    tokens_mask_list = pickle.load(pkl)
                    
                # print(triple_pars_list[0].shape, triple_pars_list[1].shape)
                # print(tokens_mask_list[0].shape, tokens_mask_list[1].shape)
                    
                # # print(type(tokens_mask_list[0]))
                # print(tokens_mask_list[0])

                with open(path + 'preds{}_masks.pkl'.format(b), 'rb') as pkl:
                    preds_mask_list = pickle.load(pkl)
                    
                # print(type(preds_mask_list[0]))
                # print('preds_mask:',preds_mask_list[0])

                with open(path + 'metadata{}.pkl'.format(b), 'rb') as pkl:
                    metadata_list = pickle.load(pkl)
                    
                    
                # print(metadata_list[0],type(metadata_list[0]))
                    
                # print(len(triple_pars_list), len(tokens_mask_list), len(preds_mask_list))
                # print('...')
                
                l=[]
                for m in metadata_list:
                    l.append(m[0])
                print(set(l))


                n = len(metadata_list)
                pmmm = list(zip(triple_pars_list, tokens_mask_list, preds_mask_list, metadata_list))
                shuffle(pmmm)
                triple_pars_list, tokens_mask_list, preds_mask_list, metadata_list = zip(*pmmm)
                                                
                def get_example(tp, tm, pm, md):
                    
                    def _int64_feature(value):
                        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
                
                    # Create a dictionary with above lists individually wrapped in Feature
                    tp=tp+1
                    tm=tm+1
                    pm=pm+1
                    example=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'triple': _int64_feature(list(tp.reshape([-1]).astype(np.int64))),
                                'tokens_mask': _int64_feature(list(tm.reshape([-1]).astype(np.int64))),
                                'preds_mask': _int64_feature(list(pm.reshape([-1]).astype(np.int64))),
                                'book_id': _int64_feature([md[0]]),
                                'counter': _int64_feature([md[1]]),
                                'file_id': _int64_feature([md[2]])
                            }
                            )
                        )
                    return example
                    
                l=len(triple_pars_list)-(len(triple_pars_list)%n_batch)
                
                file_name='./Data/interpretation/no_segment/'+str(b)+'.tfrecord'
                with tf.python_io.TFRecordWriter(file_name) as tfwriter:
                    # Iterate through all records
                #     for i in range(l):
                        if i%1000==0:
                            print('i:',i)
                        example = get_example(triple_pars_list[i], tokens_mask_list[i], preds_mask_list[i], metadata_list[i])

                #     # Append each example into tfrecord
                        tfwriter.write(example.SerializeToString())
                    
                # tfwriter.close()
                    
                    

def gelu(x):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    """
    Swish tends to work better than ReLU on deeper models across a number of challenging data sets.
    For further information:
    medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820

    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the swish activation applied.
    """
    return x * tf.nn.sigmoid(x)


def dropout(input_tensor, dropout_prob, train):
    """
      Perform dropout.
      Args:
        input_tensor: input tensor.
        dropout_prob: the probability of dropping out a value

      Returns:
        A version of `input_tensor` with dropout applied.
    """
    if not train or dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def top_k_sampling(logits, k=25, temperature=0.8):
    'k must be greater than 0'

    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)

    logits = logits / temperature

    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)


def argmax(logits):
    return tf.argmax(logits)


def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
    indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
    t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
    to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
    logits = tf.where(
        to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )

    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)


def sampling(logits, temperature=0.8):
    logits = logits / temperature
    sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
    return tf.reduce_sum(sample)
