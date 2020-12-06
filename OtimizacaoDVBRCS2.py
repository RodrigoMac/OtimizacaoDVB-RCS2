''' *********************** IMPORTANTE: ***********************

    VARIAVEIS DE ANALISE DESTACADAS ENTRE COMENTARIOS COM ASPAS
    
'''

import time
import numpy as np
import matplotlib.pyplot as plt


def Bounded(N, efficiency, weight, c, priority, priorchance):
    
#    print()
#    print("c")
#    print(c)
    remaining_c = c
    profit = 0
    
    allocation = np.zeros(N).astype(int) #QUANTOS TIMESLOTS CADA USUARIO VAI RECEBER
        
    sorted_j = np.argsort(efficiency)
    sorted_j = np.flip(sorted_j) #USUARIOS ORDENADOS POR EFICIENCIA DECRESCENTE
#    print("Usuários ordenados por eficiência: ")
#    print(sorted_j) 
#    print()
    
    temp_efficiency = efficiency.copy()
    temp_weight = weight.copy()
    temp_priority = priority.copy()
    
    for j in range(N):
        
        temp_efficiency[j] = efficiency[sorted_j[j]]
        temp_weight[j] = weight[sorted_j[j]]
        temp_priority[j] = priority[sorted_j[j]]
        
#    print("Eficiencias em ordem decrescente: ")
#    print(efficiency)
#    print("Pesos em ordem decrescente de eficiência: ")
#    print(weight)
#    print()
        
    #LAÇO DE ALOCAÇÃO GREEDY DOS USUÁRIOS
    for j in range(N):
        
        if temp_priority[j] < priorchance:
            if temp_weight[j] <= remaining_c:
                allocation[j] = temp_weight[j]
                profit += temp_weight[j]*temp_efficiency[j]
                remaining_c -= temp_weight[j]
                
            else:
                allocation[j] = remaining_c
                profit += remaining_c*temp_efficiency[j]
                remaining_c = 0
            
#    print()
#    print("remaining_c")
#    print(remaining_c)
    allocation[0] += remaining_c
    remaining_c = 0
#    print("remaining_c")
#    print(remaining_c)
#    print()
    profit += remaining_c*temp_efficiency[0]
    
#    print("total timeslots usados")
#    print(np.sum(allocation))
#    print()
              
#    print("Timeslots recebidos por cada usuário em ordem de eficiência: ")
#    print(allocation)
#    print("Usuários ordenados por eficiência: ")
#    print(sorted_j) 
#    print("Profit da solução: ")
#    print(profit)
#    print("Total de timeslots usados: ")
#    print(np.sum(allocation))
    
    return allocation, sorted_j, profit;




'''
        FUNÇÃO PRINCIPAL
'''
start_time = time.time()

# SYMBOL LIST

# SigPow       # SIGNAL POWER IN DB
# BWsf         # SUPERFRAME TOTAL BANDWIDTH
# Tsf          # SUPERFRAME TOTAL DURATION
# K            # NUMBER OF DRA SCHEMES
# Ttsk         # TIME SLOT DURATION FOR DRA SCHEME K 
# Nk           # NUMBER OF TIME SLOTS PER CHANNEL FOR DRA SCHEME K 
# Rk           # DRA SCHEME K SYMBOL RATE
# GBk          # DRA SCHEME K GUARD BAND
# Bk           # DRA SCHEME K BURST LENGTH IN SYMBOLS
# GSk          # DRA SCHEME K NUMBER OF GUARD SYMBOLS
# Lk           # NUMBER OF RCST'S
# Ckl          # AMOUNT OF RESOURCE REQUESTS FROM RCST'S WITH DRA SCHEME K
# Dk           # PAYLOAD SIZE OF A BURST
# alpha        # ROLL OFF FACTOR

# INITIALIZATION

''' ***** NUMERO DE ITERACOES PARA OTIMIZACAO DE USUARIOS ***** '''
ix = 1001
''' ***** NUMERO DE ITERACOES PARA OTIMIZACAO DOS CANAIS  ***** '''
iy = 50
''' *********************************************************** '''

K = 5 #QUANTIDADE DE ESQUEMAS DRA, NÃO MEXER

Jy_mem_sum = np.zeros(iy)
Jx_mem_sum = np.zeros([ix, K])
profit_mem_sum = np.zeros([ix, K])


''' ***** NUMERO DE ITERACOES PARA CALCULO DE MÉDIA ***** '''
reps = 10 # ITERATIONS FOR AVERAGE
''' ***************************************************** '''
   
for z in range(reps):

    #WaveID = [13, 14, 15, 16, 17]
    #Modulation = [QPSK, QPSK, QPSK, QPSK, QPSK]
    
    BWsf  = 3700 # [kHz]        
    Tsf   = 1000 # [ms]       
    K     = 5 # int 
    Rk    = [128, 192, 288, 432, 648] # [ksym/s] or [sym/ms]
    Bk    = 1616 # sym
    GSk   = 4 # sym  
    
    ''' ***** QUANTIDADE DE USUARIOS POR ESQUEMA DRA ***** '''
    Lk    = np.ceil(np.random.poisson(lam = 30, size = K)) #EXCESSO DE PAYLOAD
#    Lk    = np.ceil(np.random.poisson(lam = 50, size = K)) #EXCESSO DE PAYLOAD
#    Lk    = np.ceil(np.random.poisson(lam = 150, size = K)) #EXCESSO DE USUARIOS
#    Lk    = np.ceil(np.random.poisson(lam = 100, size = K))
    ''' ************************************************** '''
    
    stop = np.amax(Lk).astype(int)
    #Ckl   = np.ceil(np.random.uniform(0, 1000, (stop, K))) 
    Dk    = [123, 188, 264, 298, 333] # bytes      
    alpha = 0.2 

    ''' *** TAXA DE TRANSFERENCIA DE CANAIS ENTRE ESQUEMAS DRA POR ITERACAO DO FAIRNESS *** '''    
    beta  = 1
    ''' *** TAXA DE TRANSFERENCIA DE TIMESLOTS ENTRE USUARIOS POR ITERACAO DO FAIRNESS  *** '''
    gamma = [1, 1, 1, 1, 1]
    ''' *********************************************************************************** '''
    
    Jmax = 0
    profit = np.zeros(K).astype(int)
    
    Jy_mem = np.zeros(iy)
    Jx_mem = np.zeros([ix, K])    
    profit_mem = np.zeros([ix, K])
    usedcap = np.zeros(K)
    
    ''' ***** PORCENTAGEM DE USUÁRIOS PRIORITARIOS ***** '''
    prior_chance = 1.1 # Chance do usuário ser prioritário = 1 - prior_chance
    ''' ************************************************ '''
    
    
    # ALGORITMO FAIRNESS
    
    # 1
    
    BWk = [78, 156, 312, 624, 1248] #kHz
       
#    m = [32, 16, 8, 8, 2]
#    m = [32, 16, 8, 4, 2]
#    m = [192, 0, 0, 0, 0]
#    m = [160, 0, 0, 0, 0]
    m = [256, 0, 0, 0, 0] #20MHz de banda
#    m = [48, 24, 12, 6, 4] #20MHz de banda
    
    # 2
    
    #BWk = np.multiply(Rk,(1 + alpha)) + GBk # [kHz]
    
    Ttsk = np.divide(np.add(Bk, GSk),Rk) # [ms]
    
    Nk = Tsf//Ttsk 
    Nk = Nk.astype(int)
    
    if np.amax(Lk) > np.amax(Nk):
        stop = np.amax(Nk).astype(int)
        
    excluidos = np.zeros(K)
#    
#    for a in range(K):
#        
#            if Lk[a] > Nk[a]:
#                excluidos[a] = Lk[a] - Nk[a]
#                Lk[a] = Nk[a]
    #3 
    
    ''' ***** PAYLOAD EXIGIDA POR USUARIO ***** '''
    Ckl   = np.ceil(np.random.uniform(0, 70000, (stop, K))) #EXCESSO DE PAYLOAD
#    Ckl   = np.ceil(np.random.uniform(0, 45000, (stop, K))) #EXCESSO DE PAYLOAD
#    Ckl   = np.ceil(np.random.uniform(0, 15000, (stop, K))) #EXCESSO DE USUARIOS
#    Ckl   = np.ceil(np.random.uniform(0, 20000, (stop, K)))
    ''' *************************************** '''
    
    SigPow = np.random.random_integers(1, 100, (stop, K))
    Priority = np.random.rand(stop, K)
    
    Lk = Lk.astype(int)
    Ckl = Ckl.astype(int)
    
    for i in range(K):
        for j in range(stop):
            if j >= Lk[i]:
                Ckl[j, i] = 0
                SigPow[j, i] = 0
    
    Xkl_req = np.ceil(Ckl/Dk)
    
    for a in range(K):
        for b in range(Lk[a]):
            if Xkl_req[b, a] == 1:
                Xkl_req[b, a] += 1
                
    
#    xklsum1 = np.sum(Xkl_req[:,0])
#    xklsum2 = np.sum(Xkl_req[:,1])
#    xklsum3 = np.sum(Xkl_req[:,2])
#    xklsum4 = np.sum(Xkl_req[:,3])
#    xklsum5 = np.sum(Xkl_req[:,4])
#    
#    np.savetxt('xklreq1.txt', Xkl_req[:,0], delimiter = '    ')
#    np.savetxt('xklreq2.txt', Xkl_req[:,1], delimiter = '    ')
#    np.savetxt('xklreq3.txt', Xkl_req[:,2], delimiter = '    ')
#    np.savetxt('xklreq4.txt', Xkl_req[:,3], delimiter = '    ')
#    np.savetxt('xklreq5.txt', Xkl_req[:,4], delimiter = '    ')
    
    #4
    
    Yk_req = np.zeros(K).astype(int)
    
    for i in range(K):
        for j in range(stop):
            Yk_req[i] += Xkl_req[j,i] 
    
    #5
    
    Yk_norm = np.zeros(K).astype(int)
    
#    Yk_norm = np.multiply(m, Nk)//Yk_req
    Yk_norm = np.true_divide(np.multiply(m, Nk), Yk_req)

    #6
    
    for i in range(iy):
        kmax = np.amax(Yk_norm)
        i_kmax = np.where(Yk_norm == np.amax(Yk_norm))
        
        kmin = np.amin(Yk_norm)
        i_kmin = np.where(Yk_norm == np.amin(Yk_norm))
        
        i_kmax = np.asscalar(np.amax(i_kmax[0]))
        i_kmin = np.asscalar(np.amin(i_kmin[0]))

        del_kmax = np.lcm(BWk[i_kmax], BWk[i_kmin])/BWk[i_kmax]
        del_kmin = np.lcm(BWk[i_kmax], BWk[i_kmin])/BWk[i_kmin]
        
        if beta*del_kmax > m[i_kmax]:
            if i_kmax > i_kmin:
                beta_aux = m[i_kmax]
            else:
                beta_aux = np.floor(m[i_kmax]/del_kmax)
#            print()
#            print(m[i_kmax])
#            print(i_kmax)
#            print(i_kmin)
#            print(del_kmax)
#            print(del_kmin)
#            print("m[i_kmin]")
#            print(m[i_kmin])
            m[i_kmin] = int(m[i_kmin] + beta_aux*del_kmin)
            m[i_kmax] = int(m[i_kmax] - beta_aux*del_kmax)
#            print(m[i_kmin])
#            print(m[i_kmax])
#            print()
            
        else:
            m[i_kmax] = int(m[i_kmax] - beta*del_kmax)
            m[i_kmin] = int(m[i_kmin] + beta*del_kmin)


#        Yk_norm = np.multiply(m, Nk)//Yk_req
        Yk_norm = np.true_divide(np.multiply(m, Nk), Yk_req)
        
        Jy = (np.sum(Yk_norm))**2/(K*np.sum(np.multiply(Yk_norm, Yk_norm)))
        
        Jy_mem[i] = Jy
        
        if Jy > Jmax:
            Jmax = Jy
            m_out = m
        
    ''' ***** GAMMA DINAMICO ***** '''
    gamma = np.floor(np.divide(np.multiply(m_out,Nk),Lk))
    print("gamma")
    print(gamma)
    gamma = np.round(gamma*0.1)
    for i in range(K):
        if gamma[i] == 0:
            gamma[i] = 1   
    print(gamma)
    ''' ************************** '''
    
    #7
    
    Xk = np.zeros([stop,K])#.astype(int)

    limit = np.multiply(m_out, Nk)

    Jmax = 0
    
        # ALGORITMO PRIORITY
        
    for j in range(K):
        
        for i in range(Lk[j]):
            
            if Priority[i, j] >= prior_chance:
                Priority[i, j] = 1
                usedcap[j] += Xkl_req[i, j]
            
            else:
                Priority[i, j] = 0
                
#        print()
#        print("Xk priority")
#        print(Xk)
#        print()
    
    # FIM ALGORITMO PRIORITY
    
    
        # ALGORITMO BOUNDED
    
    for j in range(K):
#        print()
#        print("limit")
#        print(limit[j])
#        print("usedcap")
#        print(usedcap[j])
#        print()
        result = Bounded(Lk[j], SigPow[:, j], Xkl_req[:, j], limit[j] - usedcap[j], Priority[:, j], prior_chance)
        
        allocation = result[0]
        sorted_j = result[1]
        profit[j] = result[2]
        
        for i in range(Lk[j]):
            temp_x = allocation[np.where(sorted_j == i)]
            Xk[i, j] = temp_x[0]
#            if Priority[i, j] >= prior_chance:
#                Xk[i, j] = Xkl_req[i, j]
                
            
#        print()    
#        print("Quantidade de timeslots recebidos por cada usuário (não ordenado por eficiência): ")
#        print(Xk[:, j])
#        print(np.sum(Xk[:, j]))
#        print()
#        print("Profit: ")
#        print(profit)
#        print(np.sum(np.multiply(Xk[:, j], SigPow[:, j])))
#        print()
#        print(SigPow[:, j])
#        print() 
#        print() 
    
    profit_max = profit.copy()
#    print(np.sum(np.multiply(Xk, SigPow)))
    print("m_out")
    print(m_out)
    # FIM ALGORITMO BOUNDED
    
    # ALGORITMO PRIORITY
        
    for j in range(K):
        
        for i in range(Lk[j]):
            
            if Priority[i, j] >= prior_chance:
                Xk[i, j] = Xkl_req[i, j]

        profit[j] = np.sum(np.multiply(Xk[:, j], SigPow[:, j]))
                
#        print()
#        print("Xk priority")
#        print(Xk)
#        print()
    
    # FIM ALGORITMO PRIORITY
    
    profit_max = profit.copy()
    
#    print()
#    print("Xk")
#    print(np.sum(Xk[:, 0]))
#    print(np.sum(Xk[:, 1]))
#    print(np.sum(Xk[:, 2]))
#    print(np.sum(Xk[:, 3]))
#    print(np.sum(Xk[:, 4]))
#    print()
    
    for k in range(K):
        
        for l in range(stop):
            #if l < Lk[k]:
                #Xk[l,k] = limit[k]//Lk[k]
                
            if l >= Lk[k]:
                Xk[l,k] = -np.inf
                Priority[l,k] = 0
        
        #Xkl_norm = Xk//Xkl_req
        Xkl_norm = np.true_divide(Xk, Xkl_req)

        for i in range(ix):
        
            for a in range(stop):
                for b in range(K):
                    if np.isnan(Xkl_norm[a,b]) or Priority[a,b] >= prior_chance:
                        Xkl_norm[a,b] = -np.inf
            
            lmax = np.amax(Xkl_norm[:,k])
            i_lmax = np.where(Xkl_norm[:,k] == lmax)
            i_lmax = np.asscalar(np.amax(i_lmax[0]))
            
            for a in range(stop):
                for b in range(K):
                    if np.isinf(Xkl_norm[a,b]) or Priority[a,b] >= prior_chance:
                        Xkl_norm[a,b] = np.inf    
            
            lmin = np.amin(Xkl_norm[:,k])
            i_lmin = np.where(Xkl_norm[:,k] == lmin)
            i_lmin = np.asscalar(np.amin(i_lmin[0]))
            
            for a in range(stop):
                for b in range(K):
                    if np.isinf(Xkl_norm[a,b]) or Priority[a,b] >= prior_chance:
                        Xkl_norm[a,b] = np.nan

#            print("Xkl_norm")
#            print(Xkl_norm[:, k])
#            print()
#            
#            print("i_lmax")
#            print(i_lmax)
#            print("i_lmin")
#            print(i_lmin)
#            print()
#                 
#            print("Xk[i_lmax, k]")
#            print(Xk[i_lmax, k])
#            print()
#            print("Xk[i_lmin, k]")
#            print(Xk[i_lmin, k])
#            print()
#            print("Xk")
#            print(Xk[:, k])
#            print()

            if gamma[k] > Xk[i_lmax, k]:
                Xk[i_lmin, k] = Xk[i_lmin, k] + Xk[i_lmax, k]
                Xk[i_lmax, k] = 0
            
            else:
                Xk[i_lmax, k] = Xk[i_lmax, k] - gamma[k]
                Xk[i_lmin, k] = Xk[i_lmin, k] + gamma[k]
        
            
            Xk_usable = Xk.copy()
            
            for a in range(stop):
                for b in range(K):
                    if np.isinf(Xk_usable[a,b]) or np.isnan(Xk_usable[a,b]):
                        Xk_usable[a,b] = 0
                        
            #print("Xk tot")
            #print(np.sum(Xk_usable[:,k]))
            #print()
                    

            #Xkl_norm = Xk//Xkl_req
            
            '''TESTE'''
            #########################################################
            Xkl_norm = np.true_divide(Xk_usable, Xkl_req)
            #########################################################
#            print("Xkl_norm")
#            print(Xkl_norm[:, k])
#            print()
            
            for a in range(stop):
                for b in range(K):
                    if np.isnan(Xkl_norm[a,b]):
                        Xkl_norm[a,b] = 0
                        
            Xkl_norm_aux = Xkl_norm[:,k]
           
            Jx = (np.sum(Xkl_norm_aux))**2/(Lk[k]*np.sum(np.multiply(Xkl_norm_aux, Xkl_norm_aux)))
            
            if np.isnan(Jx):
                Jx = 0
            
            Jx_mem[i,k] = Jx
            profit_mem[i,k] = np.sum(np.multiply(Xk_usable[:, k], SigPow[:, k]))
            
            if(Jx > Jmax):
                Jmax = Jx
                x_out = Xk
            
            '''TESTE'''
            ###########################################################
            Xkl_norm = np.true_divide(Xk, Xkl_req)
            
            for a in range(stop):
                for b in range(K):
                    if np.isnan(Xkl_norm[a,b]):
                        Xkl_norm[a,b] = 0
            ###########################################################    
                
    Jy_mem_sum = np.add(Jy_mem_sum, Jy_mem)
    Jx_mem_sum = np.add(Jx_mem_sum, Jx_mem)
    profit_mem_sum = np.add(profit_mem_sum, profit_mem)
        
Jy_mem_sum = np.true_divide(Jy_mem_sum, reps)
Jx_mem_sum = np.true_divide(Jx_mem_sum, reps)
profit_mem_sum = np.true_divide(profit_mem_sum, reps)

#9

for k in range(K):
        
        for l in range(stop):
                
            if l >= Lk[k]:
                x_out[l,k] = 0

t = np.arange(0, iy, 1)

for a in range(iy):
    if np.isnan(Jy_mem_sum[a]):
        Jy_mem_sum[a] = 0

for a in range(iy):
    if a == 0:
        a = 0
        
    elif Jy_mem_sum[a-1] == np.max(Jy_mem_sum):
        Jy_mem_sum[a] = np.max(Jy_mem_sum)

np.savetxt('jy_mem_sum_b10.txt', Jy_mem_sum, delimiter = '    ')

a = plt.figure(1)
plt.plot(t, Jy_mem_sum, marker = 'o')
plt.axis([0, iy, 0, 1.01])
#plt.legend(loc = "lower right")
plt.xlabel("Nº de iterações")
plt.ylabel("J(x)")
plt.title("β = 2")
#a.show()

t = np.arange(0, ix, 1)

for a in range(K):
    for b in range(ix):
        if np.isnan(Jx_mem_sum[b,a]):
            Jx_mem_sum[b,a] = 0

for a in range(K):
    for b in range(ix):
        if np.isnan(profit_mem[b,a]):
            profit_mem[b,a] = 0

for a in range(K):
    
    for b in range(ix):
        if b == 0:
            b = 0
        
        elif Jx_mem_sum[(b-1),a] == np.max(Jx_mem_sum[:,a]):
            Jx_mem_sum[b,a] = np.max(Jx_mem_sum[:,a])
            profit_mem_sum[b,a] = np.min(profit_mem_sum[:,a])
                        
b = plt.figure(2)
plt.plot(t, Jx_mem_sum[:,0], label = 'DRA 1', marker = 'v', markevery = 25)
#plt.plot(t, np.divide(profit_mem[:,0], profit_max[0]), label = 'profit DRA 1', marker = 'v', markevery = 25)
plt.plot(t, Jx_mem_sum[:,1], label = 'DRA 2', marker = '^', markevery = 25)
#plt.plot(t, np.divide(profit_mem[:,1], profit_max[1]), label = 'profit DRA 2', marker = '^', markevery = 25)
plt.plot(t, Jx_mem_sum[:,2], label = 'DRA 3', marker = 's', markevery = 25)
#plt.plot(t, np.divide(profit_mem[:,2], profit_max[2]), label = 'profit DRA 3', marker = 's', markevery = 25)
plt.plot(t, Jx_mem_sum[:,3], label = 'DRA 4', marker = 'o', markevery = 25)
#plt.plot(t, np.divide(profit_mem[:,3], profit_max[3]), label = 'profit DRA 4', marker = 'o', markevery = 25)
plt.plot(t, Jx_mem_sum[:,4], label = 'DRA 5', marker = 'd', markevery = 25)
#plt.plot(t, np.divide(profit_mem[:,4], profit_max[4]), label = 'profit DRA 5', marker = 'd', markevery = 25)
plt.xlabel("Nº de iterações")
plt.ylabel("J(x)")
plt.title("γ = 1")
plt.legend(loc = "lower right")
plt.axis([0, ix, 0, 1.01])

profit_max[0] = np.max(profit_mem_sum[:,0])
profit_max[1] = np.max(profit_mem_sum[:,1])
profit_max[2] = np.max(profit_mem_sum[:,2])
profit_max[3] = np.max(profit_mem_sum[:,3])
profit_max[4] = np.max(profit_mem_sum[:,4])

c = plt.figure(3)
plt.plot(t, np.divide(profit_mem_sum[:,0], profit_max[0]), label = 'profit DRA 1', marker = 'v', markevery = 25)
plt.plot(t, np.divide(profit_mem_sum[:,1], profit_max[1]), label = 'profit DRA 2', marker = '^', markevery = 25)
plt.plot(t, np.divide(profit_mem_sum[:,2], profit_max[2]), label = 'profit DRA 3', marker = 's', markevery = 25)
plt.plot(t, np.divide(profit_mem_sum[:,3], profit_max[3]), label = 'profit DRA 4', marker = 'o', markevery = 25)
plt.plot(t, np.divide(profit_mem_sum[:,4], profit_max[4]), label = 'profit DRA 5', marker = 'd', markevery = 25)
plt.xlabel("Nº de iterações")
plt.ylabel("P(x)")
plt.title("γ = 1")
plt.legend(loc = "lower right")
plt.axis([0, ix, 0, 1.01])

#np.savetxt('m.txt', m_out, delimiter = '    ')

sum1 = np.sum(x_out[:,0])
sum2 = np.sum(x_out[:,1])
sum3 = np.sum(x_out[:,2])
sum4 = np.sum(x_out[:,3])
sum5 = np.sum(x_out[:,4])

sumtot = [sum1, sum2, sum3, sum4, sum5]

memo = np.ceil(Ckl/Dk)

memo1 = np.sum(memo[:,0])
memo2 = np.sum(memo[:,1])
memo3 = np.sum(memo[:,2])
memo4 = np.sum(memo[:,3])
memo5 = np.sum(memo[:,4])

memotot = [memo1, memo2, memo3, memo4, memo5]

cons = np.multiply(sumtot, m_out)

tot = np.multiply(limit, m_out)

percent = np.divide(sumtot, limit)

menor = np.zeros(K)
maior = np.zeros(K)
igual = np.zeros(K)

for a in range(stop):
    for b in range(K):
        
        if x_out[a,b] < Xkl_req[a,b] and a < Lk[b]:
            menor[b] += 1
            
        if x_out[a,b] > Xkl_req[a,b] and a < Lk[b]:
            maior[b] += 1
        
        if x_out[a,b] == Xkl_req[a,b] and a < Lk[b]:
            igual[b] += 1
        
end_time = time.time()

print()
print('Demanda em bytes de cada terminal:')
print(Ckl)
print()
print('Quantidade de timeslots pedidos por cada terminal:')
print(Xkl_req)
print()
print('Quantidade de timeslots fornecidos para cada terminal:')
print(x_out)
print()
print('Razão final oferta/demanda de timeslots por terminal:')
print(Xkl_norm)
print()
print('Total de timeslots disponíveis por esquema DRA:')
print(limit)
print()
print('Soma total de timeslots pedidos por cada esquema DRA:')
print(memotot)
print()
print('Soma total de timeslots fornecidos para cada esquema DRA:')
print(sumtot)
print()
print('Quantidade de timeslots disponiveis por terminal do esquema DRA:')
print(np.floor(np.divide(np.multiply(m_out,Nk),Lk)))
print()
print('Quantidade final de canais por esquema DRA:')
print(m_out)
print()
print('Produto timeslots consumidos x canais:')
print(cons)
print()
print('Produto timeslots disponiveis x canais:')
print(tot)
print()
print('Porcentagem de timeslots consumidos:')
print(percent)
print()
print('Produto nº de canal x banda:')
print(np.multiply(m_out, BWk))
print('Somatório:')
print(np.sum(np.multiply(m_out, BWk)))
print()

print('Número de terminais por esquema DRA:')
print(Lk)
print()
print('Usuários que recebem menos:')
print(menor)
print()
print('Usuários que recebem mais:')
print(maior)
print()
print('Usuários que recebem exato:')
print(igual)
print()
print('Usuários excluidos:')
print(excluidos)
print()

ts_loss = np.copy(x_out)

ts_loss[:, 0] = np.subtract(x_out[:, 0], Xkl_req[:, 0])
ts_loss[:, 1] = np.subtract(x_out[:, 1], Xkl_req[:, 1])
ts_loss[:, 2] = np.subtract(x_out[:, 2], Xkl_req[:, 2])
ts_loss[:, 3] = np.subtract(x_out[:, 3], Xkl_req[:, 3])
ts_loss[:, 4] = np.subtract(x_out[:, 4], Xkl_req[:, 4])

print("X_out - Xkl_req")
print(np.clip(ts_loss[:, 0], 0, 999))
print(np.sum(np.clip(ts_loss[:, 0], 0, 999)))
print()
print(np.clip(ts_loss[:, 1], 0, 999))
print(np.sum(np.clip(ts_loss[:, 1], 0, 999)))
print()
print(np.clip(ts_loss[:, 2], 0, 999))
print(np.sum(np.clip(ts_loss[:, 2], 0, 999)))
print()
print(np.clip(ts_loss[:, 3], 0, 999))
print(np.sum(np.clip(ts_loss[:, 3], 0, 999)))
print()
print(np.clip(ts_loss[:, 4], 0, 999))
print(np.sum(np.clip(ts_loss[:, 4], 0, 999)))

print()
print("Usuários prioritários (linha) por esquema DRA (coluna)")
print(Priority)
print("Total de timeslots consumidos por usuários prioritarios por esquema DRA")
print(usedcap)

print()
print("Tempo de execução: ")
print(end_time - start_time)
