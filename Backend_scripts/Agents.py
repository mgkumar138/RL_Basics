import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


class SARSA_Q_AC_Agent:
    ''' Tabular SARSA, Q, Actor Critic agents,  Sutton Barto '''
    def __init__(self, nstates, nactions, gamma=0.9, lr=0.25, vlr=0.5, epsdecay=0.99, agentt='Qeps'):
        self.nstates = nstates
        self.nactions = nactions
        self.value_table = np.zeros(nstates)
        self.q_table = np.zeros([nstates, nactions])
        self.epsilon = 1
        self.epsdecay = epsdecay
        self.gamma = gamma
        self.vlr = vlr
        self.lr = lr
        self.atype = agentt

    def get_action(self, state, upeps=True):
        if self.atype[-3:] == 'eps':
            if upeps:
                self.epsilon *= self.epsdecay
            if np.random.uniform() > self.epsilon:
                # exploit with greedy action
                action = np.argmax(self.q_table[state])
            else:
                # explore with random action
                action = np.random.random_integers(0,self.nactions-1)
        else:
            action = np.random.choice(np.arange(self.nactions), p=softmax(self.q_table[state]))
        return action

    def learn(self, state, action, reward, newstate, newaction):
        if self.atype[:5] == 'SARSA':
            # SARSA
            td = reward + self.gamma * self.q_table[newstate, newaction] - self.q_table[state, action]
        elif self.atype[:1] == 'Q':
            # Q learning
            td = reward + self.gamma * np.max(self.q_table[newstate]) - self.q_table[state, action]
        else:
            # actor critic: use critic to compute TD and update critic and actor separately
            td = reward + self.gamma * self.value_table[newstate] - self.value_table[state]
            self.value_table[state] += self.vlr * td
        self.q_table[state,action] += self.lr * td


class Q_MB_Agent:
    ''' Tabular Dyna-Q agent, Sutton Barto '''
    def __init__(self, nstates, nactions, nreplay,gamma=0.9, lr=0.25, vlr=0.5, epsdecay=0.99, agentt='Qeps'):
        self.nstates = nstates
        self.nactions = nactions
        self.value_table = np.zeros(nstates)
        self.q_table = np.zeros([nstates, nactions])
        self.epsilon = 1
        self.epsdecay = epsdecay
        self.gamma = gamma
        self.vlr = vlr
        self.lr = lr
        self.atype = agentt
        self.world_model = np.zeros([nstates, nactions, 2])  # store state & reward value
        self.nreplay = nreplay

    def get_action(self, state, upeps=True):
        if self.atype[-3:] == 'eps':
            if upeps:
                self.epsilon *= self.epsdecay
            if np.random.uniform() > self.epsilon:
                # exploit with greedy action
                action = np.argmax(self.q_table[state])
            else:
                # explore with random action
                action = np.random.random_integers(0,self.nactions-1)
        else:
            action = np.random.choice(np.arange(self.nactions), p=softmax(self.q_table[state]))
        return action

    def get_world_states(self, state, action):
        newstate, reward = self.world_model[state, action]
        return int(newstate), reward

    def learn(self, state, action, reward, newstate, newaction):
        if self.atype[:5] == 'SARSA':
            # SARSA
            td = reward + self.gamma * self.q_table[newstate, newaction] - self.q_table[state, action]
        elif self.atype[:1] == 'Q':
            # Q learning
            td = reward + self.gamma * np.max(self.q_table[newstate]) - self.q_table[state, action]
        else:
            # actor critic: use critic to compute TD and update critic and actor separately
            td = reward + self.gamma * self.value_table[newstate] - self.value_table[state]
            self.value_table[state] += self.lr * td
        self.q_table[state,action] += self.lr * td

        # learn world model
        self.world_model[state,action] = np.array([newstate, reward])

    def replay(self):
        for n in range(int(self.nreplay)):
            # randomly sample state
            sampstate = np.random.random_integers(0,self.nstates-1)
            # choose action based on sampled state
            sampaction = self.get_action(sampstate,upeps=False)
            # use world model to predict next state given sampled state & action
            newsampstate, sampreward = self.get_world_states(sampstate, sampaction)
            # choose newaction based on new sampled state
            newsampaction = self.get_action(newsampstate,upeps=False)

            # learn through replay
            self.learn(sampstate,sampaction, sampreward, newsampstate, newsampaction)


class Q_SR_Agent:
    ''' Tabular successor representation Dayan 1993, Gardner 2018
    https://github.com/awjuliani/successor_examples/blob/master/SR-SARSA.ipynb
    https://github.com/mphgardner/TDSR/blob/master/linearTDSR.m
    '''
    def __init__(self, nstates, nactions, gamma=0.9, rlr=0.25, mlr=0.5, epsdecay=0.99, agentt='Qeps'):
        self.nstates = nstates
        self.nactions = nactions
        self.Istate = np.eye(nstates)
        self.reward_table = np.zeros(nstates)
        self.sr_table = np.zeros([nstates, nstates, nactions])
        self.value_table = np.zeros(nstates)
        self.epsilon = 1
        self.epsdecay = epsdecay
        self.gamma = gamma
        self.mlr = mlr
        self.rlr = rlr
        self.atype = agentt

    def get_action(self, state, upeps=True):
        qsa = np.matmul(self.sr_table[state].T, self.reward_table)
        if self.atype[-3:] == 'eps':
            if upeps:
                self.epsilon *= self.epsdecay
            if np.random.uniform() > self.epsilon:
                # exploit with greedy action
                action = np.argmax(qsa)
            else:
                # explore with random action
                action = np.random.random_integers(0,self.nactions-1)
        else:
            action = np.random.choice(np.arange(self.nactions), p=softmax(qsa))
        return action

    def learn(self, state, action, reward, newstate, newaction):
        # tdm = self.Istate[state] + self.gamma * self.sr_table[newstate,:,newaction] - self.sr_table[state,:,action]
        # self.sr_table[state, action] += self.mlr * tdm
        tdr = reward - self.reward_table[state]
        self.reward_table[state] += self.rlr * tdr

        if self.atype[:5] == 'SARSA':
            # SARSA
            tdm = self.Istate[state] + self.gamma * self.sr_table[newstate,:,newaction] - self.sr_table[state,:,action]
        elif self.atype[:1] == 'Q':
            # Q learning
            tdm = self.Istate[state] + self.gamma * np.max(self.sr_table[newstate],axis=1) - self.sr_table[state, :,action]
        else:
            # actor critic: use critic to compute TD and update critic and actor separately
            td = reward + self.gamma * self.value_table[newstate] - self.value_table[state]
            self.value_table[state] += self.rlr * td

        self.sr_table[state, :, action] += self.mlr * tdm
