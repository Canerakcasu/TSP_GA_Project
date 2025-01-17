import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

def parse_tsp(filename):
    lines=open(filename).read().strip().splitlines()
    data=[]
    readcoords=False
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            readcoords=True
            continue
        if line.startswith("EOF"):
            break
        if readcoords:
            parts=line.split()
            data.append([int(parts[0]),float(parts[1]),float(parts[2])])
    df=pd.DataFrame(data,columns=["id","x","y"])
    return df

def distance(city1,city2):
    return np.sqrt((city1["x"]-city2["x"])**2+(city1["y"]-city2["y"])**2)

def total_distance(df,solution):
    dist=0
    for i in range(len(solution)-1):
        dist+=distance(df.iloc[solution[i]-1],df.iloc[solution[i+1]-1])
    dist+=distance(df.iloc[solution[-1]-1],df.iloc[solution[0]-1])
    return dist

def random_solution(df):
    route=list(df["id"])
    random.shuffle(route)
    return route

def greedy_solution(df):
    ids=list(df["id"])
    bestdist=float("inf")
    bestsol=None
    for start in ids:
        remaining=[c for c in ids if c!=start]
        route=[start]
        current=start
        while remaining:
            nxt=min(remaining,key=lambda x:distance(df.iloc[current-1],df.iloc[x-1]))
            route.append(nxt)
            remaining.remove(nxt)
            current=nxt
        d=total_distance(df,route)
        if d<bestdist:
            bestdist=d
            bestsol=route
    return bestsol

def create_population(df,pop_size,include_greedy=False):
    pop=[]
    for i in range(pop_size):
        if include_greedy and i==0:
            pop.append(greedy_solution(df))
        else:
            pop.append(random_solution(df))
    return pop

def evaluate_population(df,pop):
    scores=[total_distance(df,sol) for sol in pop]
    return scores

def selection_tournament(df,pop,k):
    selected=random.sample(pop,k)
    best=min(selected,key=lambda x:total_distance(df,x))
    return best

def crossover_pmx(parent1,parent2):
    size=len(parent1)
    p1=random.randint(0,size-1)
    p2=random.randint(p1,size-1)
    child=[None]*size
    child[p1:p2]=parent1[p1:p2]
    for x in parent2[p1:p2]:
        if x in child:
            continue
        idx=parent2.index(parent1[parent2.index(x)])
        while p1<=idx<p2:
            idx=parent2.index(parent1[idx])
        child[idx]=x
    for i in range(size):
        if child[i] is None:
            child[i]=parent2[i]
    return child

def mutate_swap(sol,prob):
    for i in range(len(sol)):
        if random.random()<prob:
            j=random.randint(0,len(sol)-1)
            sol[i],sol[j]=sol[j],sol[i]
    return sol

def new_generation(df,pop,pop_size,pmut,pxover,k):
    newpop=[]
    while len(newpop)<pop_size:
        p1=selection_tournament(df,pop,k)
        p2=selection_tournament(df,pop,k)
        if random.random()<pxover:
            child=crossover_pmx(p1,p2)
        else:
            child=p1[:]
        child=mutate_swap(child,pmut)
        newpop.append(child)
    return newpop

def run_ga(df,pop_size,epochs,pmut,pxover,k):
    pop=create_population(df,pop_size,True)
    best_sol=None
    best_fit=float("inf")
    x_vals=[]
    y_vals=[]
    for gen in range(epochs):
        scores=evaluate_population(df,pop)
        fit=min(scores)
        if fit<best_fit:
            best_fit=fit
            best_sol=pop[scores.index(fit)][:]
        print(gen, fit)  # <-- PROGRESS LINE ADDED HERE
        x_vals.append(gen)
        y_vals.append(fit)
        pop=new_generation(df,pop,pop_size,pmut,pxover,k)
    return best_sol,best_fit,x_vals,y_vals

def main():
    if len(sys.argv)<2:
        print("python main.py <tsp_file>")
        sys.exit()
    filename=sys.argv[1]
    df=parse_tsp(filename)
    rsol=random_solution(df)
    gsol=greedy_solution(df)
    rdist=total_distance(df,rsol)
    gdist=total_distance(df,gsol)
    pop_size=50
    epochs=200
    pmut=0.1
    pxover=0.8
    k=3
    best_sol,best_fit,x_vals,y_vals=run_ga(df,pop_size,epochs,pmut,pxover,k)
    plt.plot(x_vals,y_vals,label="GA Best Distance")
    plt.axhline(y=rdist,color="r",linestyle="--",label="Random Distance")
    plt.axhline(y=gdist,color="g",linestyle="-.",label="Greedy Distance")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("TSP GA Results")
    plt.savefig("result.png")
    plt.show()
    print(best_fit)

if __name__=="__main__":
    main()
