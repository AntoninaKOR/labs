import torch
from torchmetrics import Accuracy
import torch.nn as nn
from sklearn.model_selection import train_test_split
from keras.datasets import imdb, boston_housing, reuters
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from torchmetrics.regression import MeanAbsoluteError
device = 'cuda'if torch.cuda.is_available() else 'cpu'
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
class Classificator(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(10000,64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, 1),
                                 nn.Sigmoid()
   )
  def forward(self, x):
    return self.model(x)
class ReferenceClassificator(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(10000, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, 1),
                                 nn.Sigmoid()
    )
  def forward(self, x):
    return self.model(x)
def train(model, criterion, optimizer, loss_fn, training_loader, validation_loader, num_epochs=100, show_study=False):
    print("Training...")
    if show_study:
        print('\n')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies =[]
    for j, epoch in enumerate(range(num_epochs)):
        acc = running_loss = accv = running_vloss =   0.
        for i, data in enumerate(training_loader):
            x, y = data
            preds = model(x)
            loss = loss_fn(preds.view(y.size()), y)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (preds.size()[-1]==1):
                acc += criterion(preds.view(y.size()), y)
            else:
                acc += criterion(preds.argmax(1), y.argmax(1))
        avg_loss = running_loss / (i + 1)
        acc_total = acc / (i + 1)
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                x, y = vdata
                preds = model(x)
                vloss = loss_fn(preds.view(y.size()), y)
                running_vloss += vloss
                if (preds.size()[-1]==1):
                    accv += criterion(preds.view(y.size()), y)
                else:
                    accv += criterion(preds.argmax(1), y.argmax(1))
        acc_vtotal = accv / (i + 1)
        avg_vloss = running_vloss / (i + 1)
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)
        train_accuracies.append(acc_total)
        val_accuracies.append(acc_vtotal)
        if show_study:
            print (f'Epoch {j}:  Accuracy on  train set : {acc_total}, Accuracy on val set : {acc_vtotal}, Loss on train set : {avg_loss}, Loss on val set : {avg_vloss}')
    return train_losses, val_losses, train_accuracies, val_accuracies
def plot_train_val(i, train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    ax1.plot(train_losses, label = 'train')
    ax1.plot(val_losses, label = 'val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(train_accuracies, label = 'train')
    ax2.plot(val_accuracies, label = 'val')
    ax2.set_title("Accuracy")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.show()
def step_1():
    accuracy = Accuracy(task = 'binary')
    loss = nn.BCELoss()
    (train_data, train_labels), (test_data, y_test) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    x_train = torch.from_numpy(x_train).type(torch.float32)
    x_test = torch.from_numpy(x_test).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)
    train_labels = torch.from_numpy(train_labels).type(torch.float32)
    x_train, x_val, y_train, y_val = train_test_split( x_train, train_labels, test_size=0.2, random_state=42)
    training_set = torch.utils.data.TensorDataset(x_train, y_train)
    validation_set = torch.utils.data.TensorDataset(x_val, y_val)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=512, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=512, shuffle=False)
    loss_ref= nn.BCELoss()
    ref = ReferenceClassificator()
    optimizer_ref = torch.optim.Adagrad(params=ref.parameters(), lr=0.001)
    _ = train(ref, accuracy, optimizer_ref, loss_ref, training_loader, validation_loader, num_epochs=20)
    print('\n')
    model = Classificator().to(device)
    optimizer = torch.optim.Adagrad(params=model.parameters(),lr=0.001)
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, accuracy, optimizer, loss, training_loader, validation_loader, num_epochs=20)
    plot_train_val(1, train_losses, val_losses, train_accuracies, val_accuracies)
    preds =model(x_test)
    preds_ref = ref(x_test)
    print('Results on test data', f"Accuracy of model on test data: {accuracy(preds.view(y_test.size()), y_test)}, Accuracy of basic model on test data: {accuracy(preds_ref.view(y_test.size()), y_test)} ")
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
class Multiclassificator(torch.nn.Module):
  def __init__(self, n_classes=46):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(10000,32),
                                 nn.Tanh(),
                                 nn.Linear(32, 4),
                                 nn.LeakyReLU(),
                                 nn.Linear(4, n_classes),
                                 nn.Softmax(dim=1)
    )
  def forward(self, x):
    return self.model(x)
class ReferenceMulticlassificator(torch.nn.Module):
  def __init__(self, n_classes=46):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(10000, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, n_classes),
                                 nn.Softmax(dim=1)
    )
  def forward(self, x):
    return self.model(x)
def step_2():
    accuracy = Accuracy(task = 'multiclass', num_classes=46)
    loss = nn.CrossEntropyLoss()
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = to_one_hot(train_labels)
    y_test = test_labels
    x_train = torch.from_numpy(x_train).type(torch.float32)
    x_test = torch.from_numpy(x_test).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, random_state=42)
    training_set = torch.utils.data.TensorDataset(x_train, y_train)
    validation_set = torch.utils.data.TensorDataset(x_val, y_val)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=512, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=512, shuffle=False)
    loss_ref= nn.CrossEntropyLoss()
    ref = ReferenceMulticlassificator()
    optimizer_ref = torch.optim.Adagrad(params=ref.parameters(), lr=0.01)
    _ = train(ref, accuracy, optimizer_ref, loss_ref, training_loader, validation_loader, num_epochs=50)
    print('\n')
    model = Multiclassificator().to(device)
    optimizer = torch.optim.Adagrad(params=model.parameters(),lr=0.01)
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, accuracy, optimizer, loss, training_loader, validation_loader, num_epochs=50)
    plot_train_val(1, train_losses, val_losses, train_accuracies, val_accuracies)
    preds =model(x_test)
    preds_ref = ref(x_test)
    print('Results on test data',f"Accuracy of model on test data: {accuracy(preds.argmax(1), y_test)}, Accuracy of basic model on test data: {accuracy(preds_ref.argmax(1), y_test)} ")
class Regression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(13,32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32, 4),
                                 nn.Tanh(),
                                 nn.Linear(4, 1)
    )
  def forward(self, x):
    return self.model(x)
class ReferenceRegression(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential( nn.Linear(13, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1)
    )
  def forward(self, x):
    return self.model(x)
def train_cross_val(model, criterion, optimizer, loss_fn, x_train, y_train, num_epochs=100, k=8, batch_size=100, epochs=30, show_study=False):
    print("Training...")
    if show_study:
        print('\n')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    kfold = KFold(n_splits=k)
    for fold, (train_index, val_index) in enumerate(kfold.split(x_train, y_train)):
        total_acc = total_vacc = total_vloss = total_loss = 0.
        x_train_fold = x_train[train_index]
        x_val_fold = x_train[val_index]
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]
        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        val = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
        val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
        for j, epoch in enumerate(range(epochs)):
            acc_epoch = loss_epoch = vloss_epoch = vacc_epoch = 0
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = loss_fn(preds.view(y_batch.size()), y_batch)
                loss.backward()
                optimizer.step()
                acc_epoch += criterion(preds.view(y_batch.size()), y_batch).item()
                loss_epoch +=loss.item()
            for batch_index, (x, y) in enumerate(val_loader):
                with torch.no_grad():
                    preds = model(x)
                    loss = loss_fn(preds.view(y.size()), y)
                    vacc_epoch += criterion(preds.view(y.size()), y).item()
                    vloss_epoch +=loss.item()
            total_acc = acc_epoch/ ((batch_index+1)*batch_size)
            total_vacc = vacc_epoch/ ((batch_index+1)*batch_size)
            total_loss = loss_epoch/ ((batch_index+1)*batch_size)
            total_vloss = vloss_epoch/ ((batch_index+1)*batch_size)
            if show_study:
                print (f'Fold {fold} Epoch {j}:  Accuracy on  train set : {acc_epoch/ ((batch_index+1)*batch_size)}, Accuracy on val set : {vacc_epoch/ ((batch_index+1)*batch_size)}, Loss on train set : {loss_epoch/ ((batch_index+1)*batch_size)}, Loss on val set : {vloss_epoch/ ((batch_index+1)*batch_size)}')
            train_losses.append(total_loss)
            val_losses.append(total_vloss)
            train_accuracies.append(total_acc)
            val_accuracies.append(total_vacc)
        print('\n')
    return train_losses, val_losses, train_accuracies, val_accuracies
def step_3():
    metric = MeanAbsoluteError()
    loss = nn.MSELoss()
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std
    x_train = torch.from_numpy(train_data).type(torch.float32)
    x_test = torch.from_numpy(test_data).type(torch.float32)
    y_test = torch.from_numpy(test_targets).type(torch.float32)
    y_train = torch.from_numpy(train_targets).type(torch.float32)
    loss_ref= nn.MSELoss()
    ref = ReferenceRegression().to(device)
    optimizer_ref = torch.optim.Adagrad(params=ref.parameters(), lr=0.01)
    _ = train_cross_val(ref, metric, optimizer_ref, loss_ref, x_train, y_train, num_epochs=30, k=3)
    print('\n')
    model = Regression().to(device)
    optimizer = torch.optim.Adagrad(params=model.parameters(),lr=0.01)
    train_losses, val_losses, train_accuracies, val_accuracies = train_cross_val(model, metric, optimizer, loss, x_train, y_train, num_epochs=30, k=3)
    plot_train_val(1, train_losses, val_losses, train_accuracies, val_accuracies)
    preds =model(x_test)
    preds_ref = ref(x_test)
    print('Results on test data',f"Accuracy of model on test data: {metric(preds.view(y_test.size()), y_test)}, Accuracy of basic model on test data: {metric(preds_ref.view(y_test.size()), y_test)} ")
def main():
  print('Step 1')
  step_1()
  print('Step 2')
  step_2()
  print('Step 3')
  step_3()
if __name__ == "__main__":
  main()