import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from utils import Vocab, read_data
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to use gpu on kaggle or colab

class RNNModel(nn.Module):
	def __init__(self, vocab, dims):
		super().__init__()
		self.vocab = vocab
		self.dims = dims
		"""	TODO: Initialize RNN weights/layers."""
		self.embedding = nn.Embedding(len(vocab.sym2num), dims)
		self.Wx = nn.Linear(dims, dims, bias=False)
		self.Wh = nn.Linear(dims, dims, bias=True) 
		self.fc = nn.Linear(dims, len(vocab.sym2num))

	def start(self):
		return torch.zeros(1, self.dims, device=device)

	def step(self, h, idx):
		"""	TODO: Pass idx through the layers of the model. Return the updated hidden state (h) and log probabilities."""
		idx = torch.tensor([[idx]], device=device)
		embed = self.embedding(idx)
		embed = embed.squeeze(1)
		h = torch.tanh(self.Wx(embed) + self.Wh(h))
		logits = self.fc(h)
		log_probs = torch.log_softmax(logits, dim=-1)

		return h, log_probs.squeeze(0)

	def predict(self, h, idx):
		"""	TODO: Obtain the updated hidden state and log probabilities after calling self.step(). 
			Return the updated hidden state and the most likely next symbol."""
		new_h, log_probs = self.step(h, idx)
		next_idx = torch.argmax(log_probs).item()
  
		return new_h, next_idx

	def fit(self, data, lr=0.001, epochs=10):
		"""	TODO: Fill in the code using PyTorch functions and other functions from part2.py and utils.py.
			Most steps will only be 1 line of code. You may write it in the space below the step."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		# 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
		criterion = nn.NLLLoss()
		# 3. Loop through the specified number of epochs.
		for epoch in range(epochs):
		#	 1. Put the model into training mode using `self.train()`.
			self.train()
		#	 2. Shuffle the training data using random.shuffle().
			random.shuffle(data)
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of characters (`total_chars`).
			total_loss = 0.0
			total_chars = 0
		#	 4. Loop over each sentence in the training data.
			for sentence in data:
		#	 	 1. Initialize the hidden state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph with `.detach()`.
				h = self.start().to(device).detach()
		#	 	 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
				optimizer.zero_grad()
		#	 	 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
				loss = 0.0
		#	 	 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
				for i in range(1, len(sentence)):
		#	 	 	1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
					prev_char = self.vocab.sym2num.get(sentence[i - 1], self.vocab.sym2num['<UNK>'])
					curr_char = self.vocab.sym2num.get(sentence[i], self.vocab.sym2num['<UNK>'])
		#			2. Call self.step() to get the next hidden state and log probabilities over the vocabulary given the previous character.
					h, log_probs = self.step(h, prev_char)
		#			3. See if this matches the actual current character (numberized). Do so by computing the loss with the nn.NLLLoss() loss initialized above. 
		#			   * The first argument is the updated log probabilities returned from self.step(). You'll need to reshape it to `(1, V)` using `.view(1, -1)`.
		#			   * The second argument is the current numberized character. It will need to be wrapped in a tensor with `device=device`. Reshape this to `(1,)` using `.view(1)`.
					log_probs = log_probs.view(1, -1)
					target = torch.tensor([curr_char], device=device).view(1)
					char_loss = criterion(log_probs, target)
		#			4. Add this this character loss value to `loss`.
					loss += char_loss
		#			5. Increment `total_chars` by 1.
					total_chars += 1
		#	 	 5. After processing the full sentence, call `loss.backward()` to compute gradients.
				loss.backward()
		#		 6. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
		#		 7. Call `optimizer.step()` to update the model parameters using the computed gradients.
				optimizer.step()
		#		 8. Add `loss.item()` to `total_loss`.
				total_loss += loss.item()
		#	5. Compute the average loss per character by dividing `total_loss / total_chars`.
			avg_loss = total_loss / total_chars
		#	6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch. Average loss per character should always decrease epoch to epoch and drop from about 3 to 1.2 over the 10 epochs.
			print(f"Epoch {epoch+1}/{epochs}, Avg Loss per char: {avg_loss:.4f}")
   
	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy."""
		self.eval()
		correct = 0
		total = 0

		with torch.no_grad():
			for sentence in data:
				h = self.start().to(device)
     
				for i in range(1, len(sentence)):
					prev_char = self.vocab.sym2num.get(sentence[i - 1], self.vocab.sym2num['<UNK>'])
					curr_char = self.vocab.sym2num.get(sentence[i], self.vocab.sym2num['<UNK>'])

					h, pred_idx = self.predict(h, prev_char)
					if pred_idx == curr_char:
						correct += 1
					total += 1
		return correct / total if total > 0 else 0.0

if __name__ == '__main__':
	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')

	vocab = Vocab()
	"""TODO: Populate vocabulary with all possible characters/symbols in the training data, including '<BOS>', '<EOS>', and '<UNK>'."""
	for sequence in train_data:
		for token in sequence:
			vocab.add(token)

	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')
 
	model = RNNModel(vocab, dims=128).to(device)
	model.fit(train_data)

	torch.save({
		'model_state_dict': model.state_dict(),
		'vocab': model.vocab,
		'dims': model.dims
	}, 'rnn_model.pth')

	"""Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
	# checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
	# vocab = checkpoint['vocab']
	# dims = checkpoint['dims']
	# model = RNNModel(vocab, dims).to(device)
	# model.load_state_dict(checkpoint['model_state_dict'])
	"""

	model.eval()

	print(model.evaluate(val_data), model.evaluate(test_data))

	"""Generate the next 100 characters for the free response questions."""
	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			idx = vocab.numberize(char)
			state, _ = model.step(state, idx)
		idx = vocab.numberize(x[-1])
		for _ in range(100):
			state, idx = model.predict(state, idx)
			sym = vocab.denumberize(idx)
			x += sym # My predict() returns the denumberized symbol. Yours may work differently; change the code as needed.
		print(''.join(x))
