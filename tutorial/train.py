#!/usr/bin/env python
# coding: utf-8

# # BERT Pre-training and Fine-tuning

# ## Preparation
# 
# First, let's import necessary modules.

# Note that utils.py includes some Blocks defined in the previous transformer notebook

# In[4]:


get_ipython().system('pip install d2l -q')
get_ipython().system('pip install gluonnlp -q')


# In[5]:


get_ipython().system('pip uninstall mxnet-cu100mkl -y')
get_ipython().system('pip install mxnet-cu100mkl -q')


# In[16]:


import d2l
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp
from utils import train_loop, predict_sentiment


# ## BERT Fine-tuning (Sentiment Analysis)

# In this section, we fine-tune the BERT Base model for sentiment analysis on the IMDB dataset.
# 
# ### BERT for Sentence Classification
# 
# Let's first take
# a look at the BERT model
# architecture for single sentence classification below:

# Here the model takes a sentences and pools the representation of the first token in the sequence.
# Note that the original BERT model was trained for a masked language model and next-sentence prediction tasks, which includes layers for language model decoding and
# classification. These layers will not be used for fine-tuning sentence classification.

# ### Get Pre-train BERT Model

# We can load the pre-trained BERT fairly easily using the model API in GluonNLP, which returns the vocabulary along with the model. We include the pooler layer of the pre-trained model by setting `use_pooler` to `True`.
# The list of pre-trained BERT models available in GluonNLP can be found [here](../../model_zoo/bert/index.rst).

# In[7]:


ctx = d2l.try_all_gpus()
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                            dataset_name='book_corpus_wiki_en_uncased',
                                            pretrained=True, ctx=ctx,
                                            use_decoder=False, use_classifier=False)
print(bert_base)


# ### Model and Loss for Fine-tuning

# Now that we have loaded the BERT model, we only need to attach an additional layer for classification.
# The `BERTClassifier` class uses a BERT base model to encode sentence representation, followed by a `nn.Dense` layer for classification. We only need to initialize the classification layer. The encoding layers are already initialized with pre-trained weights.

# In[8]:


class BERTClassifier(gluon.nn.Block):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        # extra layer used for classification
        self.classifier = gluon.nn.Dense(num_classes)

    def forward(self, inputs, segment_types, seq_len):
        seq_encoding, cls_encoding = self.bert(inputs, segment_types, seq_len)
        return self.classifier(cls_encoding)

loss_fn = gluon.loss.SoftmaxCELoss()
net = BERTClassifier(bert_base, 2)
net.classifier.initialize(ctx=ctx)


# ## Data Preprocessing
# 
# To use the pre-trained BERT model, we need to:
# - tokenize the inputs into word pieces,
# - insert [CLS] at the beginning of a sentence, 
# - insert [SEP] at the end of a sentence, and
# - generate segment ids

# ### BERT-specific Transformations

# We again use the IMDB dataset, but for this time, downloading using the GluonNLP data API. We then use the transform API to transform the raw scores to positive labels and negative labels. 
# To process sentences with BERT-style '[CLS]', '[SEP]' tokens, you can use `data.BERTSentenceTransform` API.

# In[9]:


train_dataset_raw = nlp.data.IMDB('train')
test_dataset_raw = nlp.data.IMDB('test')

tokenizer = nlp.data.BERTTokenizer(vocabulary)

def transform_fn(data):
    text, label = data
    # Transform label into position / negative
    label = 1 if label >= 5 else 0
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128,
                                               pad=False, pair=False)
    data, length, segment_type = transform([text])
    data = data.astype('float32')
    length = length.astype('float32')
    segment_type = segment_type.astype('float32')
    return data, length, segment_type, label


# In[10]:


train_dataset = train_dataset_raw.transform(transform_fn)
test_dataset = test_dataset_raw.transform(transform_fn)

print(vocabulary)
print('Index for [CLS] = ', vocabulary['[CLS]'])
print('Index for [SEP] = ', vocabulary['[SEP]'])

data, length, segment_type, label = train_dataset[0]
print('words = ', data.astype('int32'))


# ### Batchify and Data Loader

# In[11]:


padding_id = vocabulary[vocabulary.padding_token]
batchify_fn = nlp.data.batchify.Tuple(
        # words: the first dimension is the batch dimension
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        # valid length
        nlp.data.batchify.Stack(),
        # segment type : the first dimension is the batch dimension
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        # label
        nlp.data.batchify.Stack(np.float32))

batch_size = 32 * len(ctx)
train_data = gluon.data.DataLoader(train_dataset,
                                   batchify_fn=batchify_fn, shuffle=True,
                                   batch_size=batch_size, num_workers=4)
test_data = gluon.data.DataLoader(test_dataset,
                                  batchify_fn=batchify_fn,
                                  shuffle=False, batch_size=batch_size, num_workers=4)


# ### Training Loop

# Now we have all the pieces to put together, and we can finally start fine-tuning the
# model with a few epochs.

# In[14]:


num_epoch = 1
lr = 0.00005
train_loop(net, train_data, test_data, num_epoch, lr, ctx, loss_fn)


# ### Prediction

# In[17]:


predict_sentiment(net, ctx, vocabulary, tokenizer, 'this movie is so great')


# ## Conclusion
# 
# In this tutorial, we showed how to fine-tune sentiment analysis model with pre-trained BERT parameters. In GluonNLP, this can be done with such few, simple steps. All we did was apply a BERT-style data transformation to pre-process the data, automatically download the pre-trained model, and feed the transformed data into the model, all within 50 lines of code!
# 
# For more fine-tuning scripts, visit the [BERT model zoo webpage](http://gluon-nlp.mxnet.io/model_zoo/bert/index.html).
# 
# ## References
# 
# [1] Devlin, Jacob, et al. "Bert:
# Pre-training of deep
# bidirectional transformers for language understanding."
# arXiv preprint
# arXiv:1810.04805 (2018).
# 
# [2] Dolan, William B., and Chris
# Brockett.
# "Automatically constructing a corpus of sentential paraphrases."
# Proceedings of
# the Third International Workshop on Paraphrasing (IWP2005). 2005.
# 
# [3] Peters,
# Matthew E., et al. "Deep contextualized word representations." arXiv
# preprint
# arXiv:1802.05365 (2018).
# 
# [4] Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).
# 
# For fine-tuning, we only need to initialize the last classifier layer from scratch. The other layers are already initialized from the pre-trained model weights.


