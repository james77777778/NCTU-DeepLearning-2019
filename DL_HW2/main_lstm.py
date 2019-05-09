import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchtext import data

from model import LSTM


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch,
    i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.x
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.y)
        acc = binary_accuracy(predictions, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.x
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.y)
            acc = binary_accuracy(predictions, batch.y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


'''
# only use with jupyter notebook in vscode
if 'DL_HW2' not in os.getcwd():
    os.chdir(os.getcwd()+'/DL_HW2')
'''
# create needed folder
needed_folder = ['results', 'models']
for f in needed_folder:
    if not os.path.exists(f):
        os.makedirs(f)
# load data
TITLE = data.Field(
    tokenize="spacy", fix_length=10, pad_token="0", include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
fields = {'title': ('x', TITLE), 'label': ('y', LABEL)}
train_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.json', test='test.json',
    format='json', fields=fields
)
# build vocab
st = time.time()
MAX_VOCAB_SIZE = 25000
TITLE.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
et = time.time()
print("Build Vocab: {:.5f} secs".format(et-st))
# print(TITLE.vocab.freqs.most_common(20))
# print(LABEL.vocab.stoi)

# params
BATCH_SIZE = 64
INPUT_DIM = len(TITLE.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYER = 1
BIDIRECTIONAL = True
DROPOUT = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# setup iterator, model, opti, crit
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda data: len(data.x),
    sort_within_batch=True,
    device=device)
train_iterator.shuffle = True
test_iterator.shuffle = False
model = LSTM(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
    N_LAYER, BIDIRECTIONAL, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss().to(device)
# train
N_EPOCHS = 150
best_test_loss = float('inf')
train_record = {"acc": [], "loss": []}
test_record = {"acc": [], "loss": []}
for epoch in range(N_EPOCHS):
    st = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    et = time.time()
    if test_loss < best_test_loss:
        best_valid_loss = test_loss
        torch.save(model.state_dict(), 'models/paper-model.pt')
    print(f'Epoch: {epoch+1:02} | Epoch Time: {et-st}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
    train_record["acc"].append(train_acc*100.)
    train_record["loss"].append(train_loss)
    test_record["acc"].append(test_acc*100.)
    test_record["loss"].append(test_loss)
# plot results
class_name = "LSTM"
plt.figure()
plt.ylim(0.0, 5.0)
plt.title('training curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(train_record["loss"], label='train')
ymin = min(train_record["loss"])
xmin = train_record["loss"].index(ymin)
plt.annotate("{:.3f}".format(ymin), xy=(xmin, ymin))
plt.plot(test_record["loss"], label='test')
ymin = min(test_record["loss"])
xmin = test_record["loss"].index(ymin)
plt.annotate("{:.3f}".format(ymin), xy=(xmin, ymin))
plt.legend(loc='upper left')
plt.savefig('results/{}_loss.png'.format(class_name))
plt.figure()
plt.ylim(0.0, 105.0)
plt.yticks(np.arange(0.0, 110.0, 10.0))
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.plot(train_record['acc'], label='train')
plt.plot(test_record['acc'], label='test')
ymax = max(test_record['acc'])
xmax = test_record['acc'].index(ymax)
plt.annotate("(epoch={}){:.2f}".format(xmax, ymax), xy=(xmax, ymax))
plt.legend(loc='upper left')
plt.savefig('results/{}_acc.png'.format(class_name))
