import numpy as np
import matplotlib.pyplot as plt
from Backend_scripts.Environments import LinearEnv
from Backend_scripts.Agents import Q_MB_Agent


def run_expt(epochs, agentt='Qeps',lr=0.25,gamma=0.9, nreplay=0, epsdecay=0.99):
    # init env
    env = LinearEnv(nstates=10)

    # init agent
    qagent = Q_MB_Agent(nstates=env.nstates, nactions=env.nactions, lr=lr,
                        gamma=gamma,agentt=agentt, nreplay=nreplay, epsdecay=epsdecay)
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
                qagent.replay()
                trackt.append(env.t)
                break

        # if e % 5 == 0:
        #     print(e, np.round(qagent.q_table,1))

    return np.array(trackt), qagent.q_table


if __name__ == '__main__':
    btstp = 20
    epochs = 20
    lr = 0.5
    gamma = 0.99
    epsdecay = 0.999

    exptname = '{}lr_{}gamma_{}eps'.format(lr, gamma, epsdecay)
    latency = np.zeros([3, btstp, epochs])
    agenttable = []
    allreplay = np.linspace(0,50,3)

    for a, nreplay in enumerate(allreplay):
        avgQtable = 0

        for b in range(btstp):
            latency[a, b], q_table = run_expt(epochs,agentt='Qeps', lr=lr,gamma=gamma, nreplay=nreplay, epsdecay=epsdecay)
            avgQtable += q_table
        agenttable.append(avgQtable / btstp) # get mean q_table values

    f = plt.figure()
    f.text(0.01,0.01,exptname)
    for a in range(len(allreplay)):
        plt.subplot(221)
        plt.errorbar(x=np.arange(epochs), y=np.mean(latency[a],axis=0),yerr=np.std(latency[a],axis=0)/np.sqrt(btstp))
        plt.title('Latency')
        plt.legend(allreplay)
        plt.xlabel('Epoch')
    for a in range(len(allreplay)):
        plt.subplot(2,2,2+a)
        plt.title('R{} policy'.format(allreplay[a]))
        plt.imshow(agenttable[a],aspect='auto')
        plt.colorbar()
    plt.show()



