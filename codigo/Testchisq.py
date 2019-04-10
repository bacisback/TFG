import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


def tabla_contingencia(x):
    t5 = sum(x < -0.75)
    t6 = sum(x > 0.75)
    t1 = sum(x < -0.25) - t5
    t2 = sum(x < 0) - t1 - t5
    t4 = sum(x > 0.25) - t6
    t3 = sum(x > 0) - t4 - t6
    # print(t5,t6)
    tabla = [t1, t2, t3, t4, len(x) - t1-t2-t3-t4]
    # print(tabla)
    return tabla


def testChisq(x, y, ruido):
    power = 0
    y = np.array(y)
    for i in range(100):
        aux = y + np.random.normal(0, ruido, len(x))
        aux = (aux - np.mean(aux))/np.std(aux)

        p = chi2_contingency(tabla_contingencia(x), tabla_contingencia(aux))[1]
        if p > 0.05:
            power += 1
    return power/100


n = 500

x = np.random.rand(n)
y = np.log(x)
x = (x - np.mean(x))/np.std(x)
x = np.array(x)


print("Res0\n", testChisq(x, y, 0))
print("Res1\n", testChisq(x, y, 1./9))
print("Res2\n", testChisq(x, y, 2./7))
print("Res3\n", testChisq(x, y, 3./5))

y = (x-0.5)**2
print("Res0\n", testChisq(x, y, 0))
print("Res1\n", testChisq(x, y, 1./9))
print("Res2\n", testChisq(x, y, 2./7))
print("Res3\n", testChisq(x, y, 3./5))
#y = (y - np.mean(y))/np.std(y)
# print(resamplingTest(x,y)) #Mayor que 0.05 son independientes
#r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
# print("Vstatistic2",r)

# print(np.sqrt(np.log(6/0.95)/(0.24*n)))
y = np.copy(x)
print("Res0\n", testChisq(x, y, 0))
print("Res1\n", testChisq(x, y, 1./9))
print("Res2\n", testChisq(x, y, 2./7))
print("Res3\n", testChisq(x, y, 3./5))
# Menor que 0.05 no son indeps
#estadistico = HSIC_U_statistic_test(x,y)
#r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
# print(r)
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
print("Res0\n", testChisq(x, y, 0))
print("Res1\n", testChisq(x, y, 1./9))
print("Res2\n", testChisq(x, y, 2./7))
print("Res3\n", testChisq(x, y, 3./5))
