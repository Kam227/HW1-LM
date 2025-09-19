import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import Vocab, read_data
import time
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
	def __init__(self, vocab, dims):
		super().__init__()
		self.vocab = vocab
		self.dims = dims
		"""	TODO: Initialize LSTM weights/layers."""
		self.embedding = nn.Embedding(len(vocab.sym2num), dims)
		self.Wx = nn.Linear(dims, 4 * dims, bias=True)
		self.Wh = nn.Linear(dims, 4 * dims, bias=False)
		self.fc = nn.Linear(dims, len(vocab.sym2num))

	def start(self):
		h = torch.zeros(1, self.dims, device=device)
		c = torch.zeros(1, self.dims, device=device)
		return (h, c)

	def step(self, state, idx):
		"""	TODO: Pass idx through the layers of the model. 
			Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
		h, c = state
		idx = torch.tensor([idx], device=device)
		x = self.embedding(idx)
  
		gates = self.Wx(x) + self.Wh(h)
		i, f, o, g = gates.chunk(4, dim=1)
  
		i = torch.sigmoid(i)
		f = torch.sigmoid(f)
		o = torch.sigmoid(o)
		g = torch.tanh(g)
  
		new_c = f * c + i * g
		new_h = o * torch.tanh(new_c)

		logits = self.fc(new_h)
		log_probs = torch.log_softmax(logits, dim=-1)

		return (new_h, new_c), log_probs.squeeze(0)

	def predict(self, state, idx):
		"""	TODO: Obtain the updated state and log probabilities after calling self.step(). 
			Return the updated state and the most likely next symbol."""
		(new_h, new_c), log_probs = self.step(state, idx)
		next_idx = torch.argmax(log_probs).item()
  
		return (new_h, new_c), next_idx

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: This function is identical to fit() from part2.py. 
			The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		criterion = nn.NLLLoss()
  
		for epoch in range(epochs):
			self.train()
			random.shuffle(data)
			total_loss = 0.0
			total_chars = 0
   
			for sentence in data:
				state = self.start()
				state = (state[0].detach(), state[1].detach())
				optimizer.zero_grad()
				loss = 0.0
    
				for i in range(1, len(sentence)):
					prev_char = self.vocab.sym2num.get(sentence[i - 1], self.vocab.sym2num['<UNK>'])
					curr_char = self.vocab.sym2num.get(sentence[i], self.vocab.sym2num['<UNK>'])
     
					state, log_probs = self.step(state, prev_char)
     
					log_probs = log_probs.view(1, -1)
					target = torch.tensor([curr_char], device=device).view(1)
					char_loss = criterion(log_probs, target)
     
					loss += char_loss
					total_chars += 1
     
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
				optimizer.step()
    
				total_loss += loss.item()
    
			avg_loss = total_loss / total_chars
			print(f"Epoch {epoch+1}/{epochs}, Avg Loss per char: {avg_loss:.4f}")

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy.
			The code may be identitcal to evaluate() from part2.py."""
		self.eval()
		correct = 0
		total = 0

		with torch.no_grad():
			for sentence in data:
				state = self.start()
     
				for i in range(1, len(sentence)):
					prev_char = self.vocab.sym2num.get(sentence[i - 1], self.vocab.sym2num['<UNK>'])
					curr_char = self.vocab.sym2num.get(sentence[i], self.vocab.sym2num['<UNK>'])

					state, pred_idx = self.predict(state, prev_char)
					if pred_idx == curr_char:
						correct += 1
					total += 1
		return correct / total if total > 0 else 0.0

if __name__ == '__main__':
	
	vocab = Vocab()
	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')

	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')

	for sent in train_data:
		vocab.update(sent)
	model = LSTMModel(vocab, dims=128).to(device)
	model.fit(train_data)
	torch.save({
		'model_state_dict': model.state_dict(),
		'vocab': model.vocab,
		'dims': model.dims
	}, 'lstm_model.pth')

	"""Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
	# checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
	# vocab = checkpoint['vocab']
	# dims = checkpoint['dims']
	# model = RNNModel(vocab, dims).to(device)
	# model.load_state_dict(checkpoint['model_state_dict'])
	"""
	
	model.eval()

	print(model.evaluate(val_data), model.evaluate(test_data))

	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			idx = vocab.numberize(char)
			state, _ = model.predict(state, idx)
		idx = vocab.numberize(x[-1])
		for _ in range(100):
			state, idx = model.predict(state, idx)
			sym = vocab.denumberize(idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.
		print(''.join(x))
