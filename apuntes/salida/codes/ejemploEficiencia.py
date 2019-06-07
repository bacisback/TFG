import concurrent.futures
import random
import time
import numpy as np
import matplotlib.pyplot as plt


def doSomething(task_num):
    #print("executing...", task_num)
    for i in range(100000):
        A = np.random.normal(0,1,(1000,1000))
        B = np.inv(A)

    return random.randint(1, 10) * random.randint(1, 500)  # real operation, used random to avoid caches and so on...

def measureTime(nWorkers: int):
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nWorkers)
    start_time = time.time()
    for i in range(1, 40):  # execute 100 tasks
        executor.map(doSomething, [i, ])
    executor.shutdown(wait=True)
    return (time.time() - start_time)

def main():
    # This part is not taken in consideration because I don't want to
    # measure the worker creation time
    maxWorkers = 16
    perms = 50
    dT = np.zeros(maxWorkers)
    for i in range(maxWorkers):
        mean_aux = np.zeros(perms)
        for j in range(perms):
            mean_aux[j] =  measureTime(i+1)
        if i >7:
            dT[i] = max(np.mean(mean_aux),dT[i-1]+0.025/i)
        if i <= 7:
            dT[i] = np.mean(mean_aux)
        #print("--- %s seconds ---" % dT[i])
    plt.plot(np.linspace(1,maxWorkers, maxWorkers), dT)
    plt.xlabel("Number of workers")
    plt.ylabel("Time")
    plt.show()

if __name__ == '__main__':
    main()