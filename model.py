import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from utils import shape_list, gelu, swish, dropout

act_fns = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}


class Norm(Model):
    """
    n_state = shape_list(x)[-1]
    """

    def __init__(self, name, n_state, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_state = n_state

    def build(self, input_shape):
        self.g = self.add_weight(name='g', shape=[self.n_state], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(1))
        self.b = self.add_weight(name="b", shape=[self.n_state], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(0))
        super(Norm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self._norm(inputs, self.g, self.b, axis=[-1])

    def _norm(self, x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x * g + b
        return x

    def compute_output_shape(self, input_shape):
        return super(Norm, self).compute_output_shape(input_shape)


class Conv1D(Model):

    def __init__(self, name, nx, nf, rf, **kwargs):
        super().__init__(name, **kwargs)
        self.nx = nx
        self.nf = nf
        self.rf = rf

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[self.rf, self.nx, self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.random_normal(stddev=0.02))
        self.b = self.add_weight(name="b", shape=[self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(0))
        super(Conv1D, self).build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        if self.rf == 1:
            c = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.nx]), tf.reshape(self.w, [-1, self.nf])) + self.b,
                           shape_list(inputs)[:-1] + [self.nf])
        else:
            c = tf.nn.conv1d(value=inputs, filters=self.w, stride=1, padding='VALID') + self.b
        return c

    def compute_output_shape(self, input_shape):
        return super(Conv1D, self).compute_output_shape(input_shape)


class Attention(Model):
    """
    nx = shape_list(x)[-1]
    where x in inputs args of call
    """

    def __init__(self, name, nx, n_state, n_head, attn_pdrop, resid_pdrop, train, scale=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nx = nx
        self.n_state = n_state
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.train = train
        self.scale = scale
        self.conv1d_c = Conv1D(name='c_attn', nx=self.nx, nf=self.n_state * 3, rf=1)
        self.conv1d_a = Conv1D(name='c_proj', nx=self.nx, nf=self.n_state, rf=1)

    def call(self, inputs):
        c = self.conv1d_c(inputs)
        q, k, v = tf.split(c, 3, 2)
        q = self.split_heads(q, self.n_head)
        k = self.split_heads(k, self.n_head, k=True)
        v = self.split_heads(v, self.n_head)
        a = self._attn(q, k, v)
        a = self.merge_heads(a)
        a = self.conv1d_a(a)
        a = dropout(a, self.resid_pdrop, self.train)
        return a

    def split_states(self, x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1] + [n, m // n]
        return tf.reshape(x, new_x_shape)

    def merge_states(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x, n, k=False):
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])

    def merge_heads(self, x):
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(self, w):
        n = shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w * b + -1e9 * (1 - b)
        return w

    def _attn(self, q, k, v):
        w = tf.matmul(q, k)
        if self.scale:
            n_state = shape_list(v)[-1]
            w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
        w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w)
        w = dropout(w, self.attn_pdrop, self.train)
        a = tf.matmul(w, v)
        return a

    def compute_output_shape(self, input_shape):
        return super(Attention, self).compute_output_shape(input_shape)


class MLP(Model):
    def __init__(self, name, n_embd, n_state, afn, resid_pdrop, train):
        """
        The multilayer perceptron is a class of feedforward.
        This module can be used as a one-dimensional convolutional neural network
        or as a fully-connected neural network.
        """
        super().__init__(name=name)
        self.n_embd = n_embd
        self.n_state = n_state
        self.act = act_fns[afn]
        self.resid_pdrop = resid_pdrop
        self.train = train
        self.conv_fc = Conv1D("c_fc", self.n_embd, self.n_state, 1)
        self.conv_proj = Conv1D("c_proj", self.n_state, self.n_embd, 1)

    def call(self, inputs):
        hidden1 = self.act(self.conv_fc(inputs))
        hidden2 = self.conv_proj(hidden1)
        hidden2 = dropout(hidden2, self.resid_pdrop, self.train)
        return hidden2


class Block(Model):
    def __init__(self, name, n_vocab, n_embd, n_head, attn_pdrop, resid_pdrop, afn, train, scale):
        """
          The Transformer block is the core of the model.
          It contains attention layer, layer normalization and multilayer perceptron (i.e. feedforward)
        """

        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.train = train
        self.afn = afn
        self.scale = scale
        self.attn = Attention("/attn", self.n_embd, self.n_embd, self.n_head, self.attn_pdrop, self.resid_pdrop,
                              self.train, self.scale)
        self.norm1 = Norm("/ln_1", self.n_embd)
        self.mlp = MLP("/mlp", self.n_embd, 4 * self.n_embd, self.afn, self.resid_pdrop, self.train)
        self.norm2 = Norm("/ln_2", self.n_embd)

    def call(self, inputs):
        a = self.attn(inputs)
        n = self.norm1(inputs + a)
        m = self.mlp(n)
        h = self.norm2(n + m)
        return h


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, name, n_vocab, n_ctx=512, n_special=3, n_segment=3,
                 n_embd=768, stddev=0.02, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_special = n_special
        self.n_segment = n_segment
        self.n_embd = n_embd
        self.stddev = stddev

    def build(self, input_shape):
        self.we = self.add_weight(name="we",
                                  shape=(self.n_ctx + self.n_vocab + self.n_special + self.n_segment, self.n_embd),
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=self.stddev))
        super().build(input_shape=input_shape)

    def call(self, inputs):
        return tf.reduce_sum(tf.gather(self.we, inputs), 2)


class Transformer(Model):
    def __init__(self, name, n_vocab, n_ctx=512, n_special=1, n_segment=3,
                 n_embd=768, n_layer=12, n_head=12,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1,
                 afn="gelu", train=False, scale=False):
        """
          This is the transformer model in
          'https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf'
          fine-tuned for language-model.
          Args:
            name: The name of the model
            n_vocab: Size of the vocabulary
            n_ctx: Size of the context
            n_special: Number of special tokens
            n_segment: Number of different segments
            n_embd: Embeddings dimension
            n_layer: Number of the transformer blocks
            n_head: Number of attention heads
            embd_pdrop: The dropout probability for embedding layers
            attn_pdrop: The dropout probability for attention layers
            resid_pdrop: The dropout probability for attention results
            afn: The non-linear activation function in MLP
            train: It is a boolean which is true for training model, false for eval model (to control dropout)
            scale: It is a boolean which is true when attention weights are scaled
        """

        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_special = n_special
        self.n_segment = n_segment
        self.n_embd = n_embd
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.afn = afn
        self.train = train
        self.scale = scale
        self.embed = EmbeddingLayer("embedding", n_vocab, n_ctx, n_special, n_segment, n_embd)

        self.transformer_stack = []
        for layer in range(n_layer):
            self.transformer_stack.append(
                Block("h", n_vocab, n_embd, n_head, attn_pdrop, resid_pdrop, afn, train, scale))

    def transform(self, hidden):
        for block in self.transformer_stack:
            hidden = block(hidden)
        return hidden

    def call(self, inputs):
        """
        Args:
            inputs: it is a list of the ID and positions of the tokens and their mask.
                    tokens shape = (batch size, context length, 3 (IDs and positions and segments))
                    masks shape = (batch size, context length)

        Returns:
            logits: shape = (batch size, context length, vocab size)
            losses: shape = ()
        """

        tokens, masks1, masks2 = inputs[0], inputs[1], inputs[2]
        embedding = self.embed(tokens)
        self.embed.we = dropout(self.embed.we, self.embd_pdrop, self.train)
        hidden = self.transform(embedding)
        hidden = tf.reshape(tf.boolean_mask(hidden, masks2), [-1, self.n_embd])
        tokens = tf.reshape(tf.boolean_mask(tokens[:, :, 0], masks1), [-1])
        logits = tf.reshape(tf.matmul(hidden, self.embed.we[:self.n_vocab + self.n_special, :], transpose_b=True),
                            [-1, self.n_vocab + self.n_special])
        eps = 1e-100
        labels = tf.one_hot(tokens, self.n_vocab + self.n_special, 1 - (self.n_vocab - 1) * eps, eps)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=labels)
        loss = tf.reduce_mean(losses)
        return logits, loss

class Transformer_MLT(Model):
    def __init__(self, name, n_vocab, n_ctx=512, n_special=2, n_segment=3,
                 n_batch = 1, n_choice = 2, n_embd=768, n_layer=12, n_head=12,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, clf_pdrop=0.05,
                 afn="gelu", train=False, scale=False):
        """
          Multi-task learner transformer with segment embedding.
          Args:
            name: The name of the model
            n_vocab: Size of the vocabulary
            n_ctx: Size of the context
            n_special: Number of special tokens
            n_segment: Number of different segments
            n_embd: Embeddings dimension
            n_layer: Number of the transformer blocks
            n_head: Number of attention heads
            embd_pdrop: The dropout probability for embedding layers
            attn_pdrop: The dropout probability for attention layers
            resid_pdrop: The dropout probability for attention results
            afn: The non-linear activation function in MLP
            train: It is a boolean which is true for training model, false for eval model (to control dropout)
            scale: It is a boolean which is true when attention weights are scaled
        """

        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_special = n_special
        self.n_segment = n_segment
        self.n_batch = n_batch
        self.n_choice = n_choice
        self.n_embd = n_embd
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.clf_pdrop = clf_pdrop
        self.afn = afn
        self.train = train
        self.scale = scale
        self.embed = EmbeddingLayer("embedding", n_vocab, n_ctx, n_special, n_segment, n_embd)
        self.classifier = keras.layers.Dense(1, input_shape = (n_embd,), name = 'classifier',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          bias_initializer=tf.random_normal_initializer(stddev=0.03))

        self.transformer_stack = []
        for layer in range(n_layer):
            self.transformer_stack.append(
                Block("h", n_vocab, n_embd, n_head, attn_pdrop, resid_pdrop, afn, train, scale))

    def transform(self, hidden):
        for block in self.transformer_stack:
            hidden = block(hidden)
        return hidden

    def lm(self, hidden, tokens, masks1, masks2):
        hidden = tf.reshape(tf.boolean_mask(hidden, masks2), [-1, self.n_embd])
        lm_tokens = tf.reshape(tf.boolean_mask(tokens[:, :, 0], masks1), [-1])
        lm_logits = tf.reshape(tf.matmul(hidden, self.embed.we[:self.n_vocab, :], transpose_b=True),
                               [-1, self.n_vocab])
        eps = 1e-100
        labels = tf.one_hot(lm_tokens, self.n_vocab, 1 - (self.n_vocab - 1) * eps, eps)
        lm_losses = tf.nn.softmax_cross_entropy_with_logits(logits=lm_logits, labels=labels)
        lm_loss = tf.reduce_mean(lm_losses)
        return lm_logits, lm_loss

    def clf(self, hidden, clf_ids, labels):
        clf_hidden = tf.reshape(tf.gather_nd(hidden, clf_ids), [-1, self.n_embd])
        clf_logits = self.classifier(clf_hidden)
        clf_logits = dropout(clf_logits, self.clf_pdrop, self.train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])
        eps = 1e-100
        labels = tf.one_hot(labels, 2, 1 - eps, eps)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits(logits=clf_logits, labels=labels)
        clf_loss = tf.reduce_mean(clf_losses)
        return clf_loss


    def call(self, inputs):
        """
        Args:
            inputs: it is a list of the tokens, masks, indices of clf tokens, and labels
                    tokens shape = (number of choices * batch size, context length, 3)
                    masks1 is the mask of the second paragraphs of the tokens
                           shape = (number of choices * batch size, context length)
                    masks2 is the mask of the second paragraphs of the predictions
                           shape = (number of choices * batch size, context length)
                    clf_ids is the list of indices of clf tokens
                           shape = (number of choices * batch size)
                    labels shape = (number of choices * batch size)

        Returns:
            lm_logits shape = (batch size, seq length, vocab size)
            lm_losses shape = ()
            clf_losses shape = ()
        """

        tokens, masks1, masks2, clf_ids, labels = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        embedding = self.embed(tokens)
        self.embed.we = dropout(self.embed.we, self.embd_pdrop, self.train)
        hidden = self.transform(embedding)

        lm_logits, lm_loss = self.lm(hidden, tokens, masks1, masks2)
        clf_loss = self.clf(hidden, clf_ids, labels)

        return lm_logits, lm_loss, clf_loss
