import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os


def greedy_solution(df):
    """
    Implements a greedy algorithm to find an initial TSP solution.
    """
    cities = df["id"].tolist()
    current_city = cities[0]  # Start from the first city
    route = [current_city]
    unvisited_cities = set(cities[1:])

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance(df.iloc[current_city - 1], df.iloc[city - 1]))
        route.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    return route






def parse_tsp(filename):
    lines = open(filename).read().strip().splitlines()
    data = []
    readcoords = False
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            readcoords = True
            continue
        if line.startswith("EOF"):
            break
        if readcoords:
            parts = line.split()
            data.append([int(parts[0]), float(parts[1]), float(parts[2])])
    df = pd.DataFrame(data, columns=["id", "x", "y"])
    return df


def distance(city1, city2):
    return np.sqrt((city1["x"] - city2["x"]) ** 2 + (city1["y"] - city2["y"]) ** 2)


def total_distance(df, solution):
    return sum(distance(df.iloc[solution[i] - 1], df.iloc[solution[i + 1] - 1])
               for i in range(len(solution) - 1)) + distance(df.iloc[solution[-1] - 1], df.iloc[solution[0] - 1])


def random_solution(df):
    route = list(df["id"])
    random.shuffle(route)
    return route


def create_population(df, pop_size):
    return [random_solution(df) for _ in range(pop_size)]


def evaluate_population(df, pop):
    return [total_distance(df, sol) for sol in pop]


def selection_tournament(df, pop, k):
    selected = random.sample(pop, k)
    return min(selected, key=lambda x: total_distance(df, x))


def crossover_ox(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    hole = set(parent1[a:b])
    child = [None] * size
    child[a:b] = parent1[a:b]

    current = b
    for gene in parent2[b:] + parent2[:b]:
        if gene not in hole:
            if current >= size:
                current = 0
            child[current] = gene
            current += 1
    return child


def mutate_inversion(sol, prob):
    if random.random() < prob:
        i, j = sorted(random.sample(range(len(sol)), 2))
        sol[i:j + 1] = sol[i:j + 1][::-1]  # Gerçek ters çevirme işlemi
    return sol


def new_generation(df, pop, pop_size, pmut, pxover, k):
    elite_count = 5  # İlk 5 en iyi bireyi koru
    elite_solutions = sorted(pop, key=lambda x: total_distance(df, x))[:elite_count]

    newpop = []
    while len(newpop) < (pop_size - elite_count):
        p1 = selection_tournament(df, pop, k)
        p2 = selection_tournament(df, pop, k)
        if random.random() < pxover:
            child = crossover_ox(p1, p2)
        else:
            child = p1.copy()
        child = mutate_inversion(child, pmut)
        newpop.append(child)

    newpop.extend(elite_solutions)
    return newpop


def run_ga(df, pop_size, epochs, pmut, pxover, k, patience=50):
    pop = create_population(df, pop_size)
    best_sol = None
    best_fit = float("inf")
    x_vals, y_vals = [], []
    stagnant_epochs = 0

    for gen in range(epochs):
        scores = evaluate_population(df, pop)
        current_best = min(scores)

        if current_best < best_fit:
            best_fit = current_best
            best_sol = pop[scores.index(current_best)].copy()
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        x_vals.append(gen)
        y_vals.append(best_fit)

        print(f"Gen {gen:3d}: Best Distance = {best_fit:.2f}")

        if stagnant_epochs >= patience:
            print(f"Early stopping at generation {gen}")
            break

        pop = new_generation(df, pop, pop_size, pmut, pxover, k)

        if gen % 10 == 0:
            pop.extend([random_solution(df) for _ in range(10)])

    return best_sol, best_fit, x_vals, y_vals


def plot_results(x_vals, y_vals, gdist, pop_size, epochs, pmut, pxover, k, best_fit):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="GA Best Distance", color="blue", linewidth=2)
    plt.axhline(y=gdist, color="green", linestyle="--", label="Greedy Distance")

    # Grafik başlığına test edilen parametreleri ekle
    plt.title(f"Genetic Algorithm Progress\nPop: {pop_size}, Epochs: {epochs}, pmut: {pmut}, pxover: {pxover}, k: {k}",
              fontsize=12, fontweight="bold")

    # En iyi bulunan sonucu grafikte göster
    plt.text(len(x_vals) * 0.8, min(y_vals) + 50, f"Final Best: {best_fit:.2f}", fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Kaydetme
    output_filename = f"results_pop{pop_size}_epochs{epochs}_pmut{pmut}_pxover{pxover}_k{k}_gdist{gdist:.2f}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", output_filename))
    print(f"Results saved as {output_filename}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <tsp_file>")
        sys.exit()

    filename = sys.argv[1]
    df = parse_tsp(filename)

    # Parametreler
    pop_size = 300
    epochs = 150
    pmut = 0.5  # Mutation probability
    pxover = 0.7
    k = 5

    # Referans değerler
    greedy_route = greedy_solution(df)
    gdist = total_distance(df, greedy_route)

    # GA çalıştır
    best_sol, best_fit, x_vals, y_vals = run_ga(df, pop_size, epochs, pmut, pxover, k)

    # Sonuçları görselleştir
    plot_results(x_vals, y_vals, gdist, pop_size, epochs, pmut, pxover, k, best_fit)

    print(f"\nFinal Results:")
    print(f"Greed   y Distance: {gdist:.2f}")
    print(f"GA Best Distance: {best_fit:.2f}")


if __name__ == "__main__":
    main()
