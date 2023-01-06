import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests_hw4 import test_prediction, test_generation
from tqdm import tqdm

device = f'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: define other hyperparameters here
NUM_EPOCHS = 20
BATCH_SIZE = 128
EMB_DIM = 512
HIDDEN_SIZE = 512

os.chdir('HW4P1/hw4/')

# load all that we need
dataset = np.load('../dataset/wiki.train.npy', allow_pickle=True)
devset = np.load('../dataset/wiki.valid.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')

run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)

# data loader
class DataLoaderForLanguageModeling(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset#[:1] # TODO
        self.batch_size = batch_size # TODO
        self.shuffle = shuffle # TODO 

        self.seq_length = 70
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        """
            You may implement some of the techniques in https://arxiv.org/pdf/1708.02182.pdf
            example: Variable length backpropagation sequences (Section 4.1)
        """
        # 1. Randomly shuffle all the articles from the WikiText-2 dataset.
        if self.shuffle:
            np.random.shuffle(self.dataset)
        
        # 2. Concatenate all text in one long string.
        self.dataset = np.concatenate(self.dataset, axis=0)
        
        # 3. Group the sequences into batches.
        # seq_len로 데이터 자르기
        if self.dataset.size % self.seq_length != 0:
            self.dataset = self.dataset[:-(self.dataset.size % self.seq_length)]
            # self.dataset = np.concatenate((self.dataset, np.zeros(self.seq_length - self.dataset.size % self.seq_length)), axis=0)
        
        self.dataset.shape = (self.dataset.size//self.seq_length, self.seq_length)
        dataset_target = np.roll(self.dataset, -1)

        # 4. Run a loop that returns a tuple of (input, label) on every iteration with yield.
        for i in range((self.dataset.shape[0]//self.batch_size)+1):
            inputs = self.dataset[i*self.batch_size:(i+1)*self.batch_size, :]
            targets = dataset_target[i*self.batch_size:(i+1)*self.batch_size, :]
            yield (inputs, targets)

class LockedDropout(torch.nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`.

    Args:
        dropout (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, dropout=0.35):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        """
        Args:
            x (`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): 
                Input to apply dropout.
        """
        if not self.training or not self.dropout:
            return x
        
        # T, B, C -> B, T, C
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """ 
    **Thank you** to Sales Force for their initial implementation. Here is
    their `License <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`.
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    x = torch.nn.functional.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return x

# model
class Model(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_size:int):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=3, batch_first=True,
                            dropout=0.2)
        self.locked_dropout = LockedDropout(0.2)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # weight tying
        # self.linear.weight = self.embedding.weight 
        

    def forward(self, x): # 2, 3
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        # x = self.embedding(x) # 2, 3, 64
        x = embedded_dropout(self.embedding, x, 0.2)

        #x = self.locked_dropout(x)
        x, _ = self.lstm(x) # 2, 3, 128
        # x = self.locked_dropout(x)
        x = self.linear(x) # 2, 3, 33278

        return x


class TestLanguageModel:
    def predict(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        model.eval()
        predictions = []

        inp = torch.LongTensor(inp).to(device) 
        output = model(inp)
        output = output[:, -1, :]
        output = output.squeeze(1)
        predictions = output.detach().cpu().numpy()

        return predictions
        
    def generate(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        model.eval()
        generated = []
        generated_logits = []

        inp = torch.LongTensor(inp).to(device)
        output = model(inp)
        generated_logits.append(output.detach().cpu().numpy())

        # for i in range(forward):
        #     output = model(output)
        #     generated_logits.append(output.detach().cpu().numpy())

        # generated_logits = np.concatenate(generated_logits, axis=1)
        # generated = np.argmax(generated_logits, axis=2)

        return generated_logits


# model trainer
class Trainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        # feel free to define a learning rate scheduler as well if you want
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, amsgrad=True, weight_decay=1e-6)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train() # set to training mode
        batch_bar = tqdm(total=len(self.loader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
            batch_bar.set_postfix(
                loss = f"{epoch_loss/ (batch_num+1):.4f}",
                lr = f"{self.optimizer.param_groups[0]['lr']}"
            )
            batch_bar.update()

        epoch_loss = epoch_loss / (batch_num + 1)
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
            
            :return 
                    (float) loss value
        """
        # 1. Zero the gradients
        self.optimizer.zero_grad()

        # 2. Forward pass
        inputs = torch.LongTensor(inputs).to(device)
        targets = torch.LongTensor(targets).to(device)
        
        outputs = self.model(inputs) # 2, 3, len(vocab)
        outputs = outputs.view(-1, outputs.size(2))

        # 3. Compute loss
        loss = self.criterion(outputs, targets.view(-1))

        # 4. Backward pass
        loss.backward()

        # 5. Update weights
        self.optimizer.step()

        return loss.item()

    
    def test(self):
        # don't change these
        # Predictions
        self.model.eval() # set to eval mode

        predictions = TestLanguageModel.predict(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)

        nll = test_prediction(predictions, fixtures_pred['out'])
        self.val_losses.append(nll)

        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        self.epochs += 1

        # generate predictions for test data
        predictions_test = TestLanguageModel.predict(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)

        # # Generation
        # generated_logits = TestLanguageModel.generate(fixtures_gen, 10, self.model) # generated predictions for 10 words
        # generated_logits_test = TestLanguageModel.generate(fixtures_gen_test, 10, self.model)
        
        # generated = test_generation(fixtures_gen, generated_logits, vocab)
        # generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        
        # self.generated.append(generated)
        # self.generated_test.append(generated_test)
        # self.generated_logits.append(generated_logits)
        # self.generated_logits_test.append(generated_logits_test)

        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        # np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        # np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        # with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
        #     fw.write(self.generated[-1])
        # with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
        #     fw.write(self.generated_test[-1])



loader = DataLoaderForLanguageModeling(
    dataset=dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

# sanity check
for i in loader:
    # print(i)
    break

model = Model(len(vocab), embedding_dim=EMB_DIM, hidden_size=HIDDEN_SIZE)
model = model.to(device)

trainer = Trainer(
    model=model, 
    loader=loader, 
    max_epochs=NUM_EPOCHS, 
    run_id=run_id
)

best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
        trainer.save()

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.savefig("results.png")

# see generated output
# print (trainer.generated[-1]) # get last generated output