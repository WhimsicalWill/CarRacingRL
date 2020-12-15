import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import gym

from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Net(nn.Module)

	def __init__(self):
		super(Net, self).__init__()
		self.base = nn.Sequential(
			nn.Conv2d(4, 8, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(8, 16, 3, stride=2),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, stride=2), 
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, stride=1),
			nn.ReLU()) # output shape (256, 1, 1)
		self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
		self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
		self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
		self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
		self.apply(self._weights_init)

	@staticmethod
	def _weights_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
			nn.init.constant_(m.bias, 0.1)

	def forward(self, x)
		x = self.base(x)
		x = x.view(-1, 256)
		v = self.v(x)
		x = self.fc(x)
		alpha = self.alpha_head(x) + 1
		beta = self.beta_head(x) + 1

		return (alpha, beta), v


transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])

class Agent():
	clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
    	self.training_step = 0
    	self.net = Net()
    	self.buffer = np.empty(self.buffer_capacity, dtype=transition)
    	self.counter = 0

    	self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
    	# deal with datatype of state and transform it
    	state = torch.from_numpy(state)
    	with torch.no_grad():
    		alpha, beta = self.net(state)[0]
		dist = Beta(alpha, beta)
		action = dist.sample() # sampled action in interval (0, 1)
		a_logp = dist.log_prob(action).sum(dim=1) # add the log probability densities of the 3-stack

		# question: when optimizer.step applied, how to deal with multiple variable adjustments leading to prob increase
		# only one gradient, so the step is some scalar-multiple of this (and beneath clip size?)
		# in order to not stray > epsilon, one should take multiple steps of minibatch SGD, (with early stopping when > epsilon?)

		action = action.squeeze().numpy()
		a_logp = a_logp.item()
		return action, a_logp

	def save_param(self):
		torch.save(self.net.state_dict(), './param/ppo_net_params.pkl')

	def store(self, transition):
		self.buffer[self.counter] = transition
		self.counter += 1
		if self.counter == self.buffer_capacity:
			self.counter = 0
			return True
		else:
			return False

	def update(self):
		self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double)
        a = torch.tensor(self.buffer['a'], dtype=torch.double)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

         for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()

                # intuition says to do this step differently
                # i.e. compute loss using minibatches and take multiple SGD steps

                # new insight: the shape of the objective function is fundamental in limiting
                # how the parameters theta don't move to a region where L > 1 + epsilon
                # because the norm of the gradient near the 'ceiling' approaches 0, we don't move far into the territory
                # this works with multiple SGD steps, but unclear how a step of grad * lr works

                # in an update, theta_k is constant so we are always moving in the same space
                # what happens if we move with too big of a gradient?
                # then the grad = 0, and we have finished early

                # epsilon is relevant for each individual action, so if its not yet there,
                # each action takes a gradient step closer to the ceiling

                # ppo just limits the adjustments of each action under the policy (given state)
                # objective must be maxed for each action

                # when adjusting theta for another transition, a different ratio can be > epsilon
                # this is fine, as long as the optimizer does not act greedily w.r.t this
                
                self.optimizer.step()
