import numpy as np
import matplotlib.pyplot as plt
from Backend_scripts.Environments import LinearEnv
from Backend_scripts.Agents import Q_SR_Agent


def run_expt(epochs, agentt='Qeps',mlr=0.25,rlr=0.5,gamma=0.9,epsdecay=0.99):
    # init env
    env = LinearEnv(nstates=10)

    # init agent
    qagent = Q_SR_Agent(nstates=env.nstates, nactions=env.nactions, mlr=mlr,rlr=rlr,
                        gamma=gamma,agentt=agentt, epsdecay=epsdecay)
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

    return np.array(trackt), qagent.sr_table


if __name__ == '__main__':
    btstp = 5
    epochs = 20
    mlr = 0.25
    rlr = 0.5
    epsdecay = 0.99

    exptname = '{}mlr_{}rlr_{}eps'.format(mlr, rlr,epsdecay)
    latency = np.zeros([3, btstp, epochs])
    agenttable = []
    allgamma = np.linspace(0.1,0.99,3)

    for a, gamma in enumerate(allgamma):
        avgSRtable = 0

        for b in range(btstp):
            latency[a, b], sr_table = run_expt(epochs,agentt='Qeps', mlr=mlr,rlr=rlr,gamma=gamma, epsdecay=epsdecay)
            avgSRtable += sr_table
        agenttable.append(avgSRtable / btstp) # get mean q_table values

    f = plt.figure()
    f.text(0.01,0.01,exptname)
    for a in range(len(allgamma)):
        plt.subplot(221)
        plt.errorbar(x=np.arange(epochs), y=np.mean(latency[a],axis=0),yerr=np.std(latency[a],axis=0)/np.sqrt(btstp))
        plt.title('Latency')
        plt.legend(allgamma)
        plt.xlabel('Epoch')
    for a in range(len(allgamma)):
        plt.subplot(2,2,2+a)
        plt.title('R{} gamma'.format(allgamma[a]))
        plt.imshow(np.mean(agenttable[a],axis=1),aspect='auto')
        plt.colorbar()
    plt.show()



