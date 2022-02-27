import numpy as np
import matplotlib.pyplot as plt
from Backend_scripts.Environments import SimpleGrid
from Backend_scripts.Agents import SARSA_Q_AC_Agent



def run_expt(epochs, agentt='Qeps',lr=0.25,vlr=0.5,gamma=0.9, epsdecay=0.999,x=5,y=4, rloc=4,obs=[2,7,12], stpos=0):
    # init env
    env = SimpleGrid(x=x, y=y, rloc=rloc, stpos=stpos, obs=obs)

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
    btstp = 20
    epochs = 100
    lr = 0.25
    vlr = 0.5
    epsdecay = 0.99
    gamma = 0.95
    x,y = 5,4
    stpos = 0
    rloc = 4
    obs = [2,7, 12]

    exptname = '{}lr_{}vlr_{}gamma_{}eps'.format(lr, vlr, gamma,epsdecay)
    latency = np.zeros([3, btstp, epochs])
    agenttable = []
    allagents = ['SARSAeps','Qeps','ACstoch']

    for a, agentt in enumerate(allagents):
        avgQtable = 0

        for b in range(btstp):
            latency[a, b], q_table = run_expt(epochs,agentt=agentt, lr=lr,gamma=gamma, epsdecay=epsdecay,vlr=vlr,
                                              x=x,y=y, rloc=rloc,obs=obs, stpos=stpos)
            avgQtable += q_table
        agenttable.append(avgQtable / btstp) # get mean q_table values

    actions = ['U','R','D','L']

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
        polstr = []
        pol = np.argmax(agenttable[a], axis=1)
        for i in pol:
            polstr.append(actions[i])
        val = np.mean(agenttable[a],axis=1).reshape(y,x)
        plt.imshow(val,aspect='auto')
        plt.colorbar()
        i = 0
        for ix in range(x):
            for iy in range(y):
                plt.text(ix, iy, polstr[i], va='center', ha='center')
                i+=1
    plt.show()



