import numpy as np
import matplotlib.pyplot as plt
from Backend_scripts.Environments import LinearEnv
from Backend_scripts.Agents import SARSA_Q_AC_Agent


def run_expt(epochs, agentt='Qeps',lr=0.25,vlr=0.5,gamma=0.9, epsdecay=0.999):
    # init env
    env = LinearEnv(nstates=5)

    # init agent
    qagent = SARSA_Q_AC_Agent(nstates=env.nstates, nactions=env.nactions, lr=lr,gamma=gamma,agentt=agentt,epsdecay=epsdecay,vlr=vlr)
    trackt = []

    # begin training
    for e in range(epochs):
        state, reward, done = env.reset()

        while not done:
            # choose action given state
            action = qagent.get_action(state, upeps=True)

            # environment dynamics decides next state & reward
            newstate, reward, done = env.step(action)

            # choose action given newstate for SARSA
            newaction = qagent.get_action(newstate, upeps=False)

            # agent learns value and policy based on env dynamics
            qagent.learn(state=state, action=action, reward=reward, newstate=newstate, newaction=newaction)

            # set new state to be current state
            state = newstate

            if done:
                trackt.append(env.t)
                break

        # if e % 5 == 0:
        #     print(e, np.round(qagent.q_table,1))

    return np.array(trackt), qagent.q_table


if __name__ == '__main__':
    btstp = 10
    epochs = 20
    lr = 0.25
    vlr = 0.5
    epsdecay = 0.99
    gamma = 0.99

    exptname = '{}lr_{}vlr_{}gamma_{}eps'.format(lr, vlr, gamma,epsdecay)
    latency = np.zeros([3, btstp, epochs])
    agenttable = []
    allagents = ['SARSAeps','Qeps', 'ACstoch']

    for a, agentt in enumerate(allagents):
        avgQtable = 0

        for b in range(btstp):
            latency[a, b], q_table = run_expt(epochs,agentt=agentt, lr=lr,gamma=gamma, epsdecay=epsdecay,vlr=vlr)
            avgQtable += q_table
        agenttable.append(avgQtable / btstp) # get mean q_table values

    f = plt.figure()
    f.text(0.01,0.01,exptname)
    for a in range(len(allagents)):
        plt.subplot(221)
        plt.errorbar(x=np.arange(epochs), y=np.mean(latency[a],axis=0),yerr=np.std(latency[a],axis=0)/np.sqrt(btstp))
        plt.title('Latency')
        plt.legend(allagents)
        plt.xlabel('Epoch')
    for a in range(len(allagents)):
        plt.subplot(2,2,2+a)
        plt.title('{} policy'.format(allagents[a]))
        plt.imshow(agenttable[a],aspect='auto')
        plt.colorbar()
    plt.show()



