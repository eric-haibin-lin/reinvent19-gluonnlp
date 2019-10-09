import random
import math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import gluonnlp as nlp

def fit(net, train_data, test_data, num_epoch, lr, ctx, loss_fn):
    trainer = gluon.Trainer(net.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'wd':0.01},
                            update_on_kvstore=False)
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']
    grad_clip = 1
    
    num_warmup_steps = 50
    step_num = 0
    num_train_steps = len(train_data) * num_epoch

    for epoch in range(num_epoch):
        accuracy = mx.metric.Accuracy()
        running_loss = 0
        for i, (inputs, seq_lens, token_types, labels) in enumerate(train_data):
            step_num += 1
    
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)
            inputs = gluon.utils.split_and_load(inputs, ctx)
            seq_lens = gluon.utils.split_and_load(seq_lens, ctx)
            token_types = gluon.utils.split_and_load(token_types, ctx)
            labels = gluon.utils.split_and_load(labels, ctx)

            losses = []
            preds = [] 
            with mx.autograd.record():
                for inp, seq_len, token_type, label in zip(inputs, seq_lens, token_types, labels):
                    out = net(inp, token_type, seq_len)
                    loss = loss_fn(out, label.astype('float32'))
                    losses.append(loss)
                    preds.append(out)
            mx.autograd.backward(losses)
            for l in losses:
                running_loss += l.mean().asscalar() / len(losses)
            # Gradient clipping
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(1)
    
            accuracy.update(labels, preds)
            if i % 25 == 0:
                print("Batch {}, Train Acc {}, Train Loss {}".format(i, accuracy.get()[1], running_loss/(i+1)))
        print("Epoch {}, Train Acc {}, Train Loss {}".format(epoch, accuracy.get()[1], running_loss/(i+1)))
        evaluate(test_data, ctx, net)

def evaluate(test_data, ctx, net):
    accuracy = 0
    for i, (inputs, seq_lens, token_types, labels) in enumerate(test_data):
        inputs = gluon.utils.split_and_load(inputs, ctx)
        seq_lens = gluon.utils.split_and_load(seq_lens, ctx)
        token_types = gluon.utils.split_and_load(token_types, ctx)
        labels = gluon.utils.split_and_load(labels, ctx)
        for inp, seq_len, token_type, label in zip(inputs, seq_lens, token_types, labels):
            out = net(inp, token_type, seq_len)
            accuracy += (out.argmax(axis=1).squeeze() == label).mean().copyto(mx.cpu()) / len(ctx)
        accuracy.wait_to_read()
    print("Test Acc {},".format(accuracy.asscalar()/(i+1)))

def predict_sentiment(net, ctx, vocabulary, bert_tokenizer, sentence):
    ctx = ctx[0] if isinstance(ctx, list) else ctx
    max_len = 128
    padding_id = vocabulary[vocabulary.padding_token]

    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_len, pad=False, pair=False)
    dataset = gluon.data.SimpleDataset([[sentence]])
    dataset = dataset.transform(transform)
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id),
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=padding_id))
    predict_data = gluon.data.DataLoader(dataset, batchify_fn=batchify_fn,
                                         batch_size=1)

    for i, (inputs, seq_len, token_types) in enumerate(predict_data):
        inputs = mx.nd.array(inputs).as_in_context(ctx)
        token_types = mx.nd.array(token_types).as_in_context(ctx)
        seq_len = mx.nd.array(seq_len, dtype='float32').as_in_context(ctx)
        out = net(inputs, token_types, seq_len)
        label = nd.argmax(out, axis=1)
        return 'positive' if label.asscalar() == 1 else 'negative'

def get_dataloader(batch_size, vocabulary, train_dataset, test_dataset):
    padding_id = vocabulary[vocabulary.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0, pad_val=padding_id), # words
            nlp.data.batchify.Stack(), # valid length
            nlp.data.batchify.Pad(axis=0, pad_val=0), # segment type
            nlp.data.batchify.Stack(np.float32)) # label

    train_data = mx.gluon.data.DataLoader(train_dataset,
                                   batchify_fn=batchify_fn, shuffle=True,
                                   batch_size=batch_size, num_workers=4)
    test_data = mx.gluon.data.DataLoader(test_dataset,
                                  batchify_fn=batchify_fn,
                                  shuffle=False, batch_size=batch_size, num_workers=4)
    return train_data, test_data