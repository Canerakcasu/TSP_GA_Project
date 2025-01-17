
# TSP_GA_Project

A Genetic Algorithm (GA) approach to solve the classic Traveling Salesman Problem (TSP).  
This project demonstrates how to encode TSP routes into a genetic representation and apply GA operators (selection, crossover, and mutation) to iteratively find better solutions.

## Table of Contents
1. Project Overview
2. How to Run
3. File Descriptions
4. Requirements
5. Example Results
6. License

## 1. Project Overview
The Traveling Salesman Problem (TSP) is a combinatorial optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city. This project uses a Genetic Algorithm to approach the problem, providing approximate solutions efficiently.

## 2. How to Run
To run the project, follow these steps:
1. Clone this repository or download the project files:
   ```
   git clone https://github.com/Canerakcasu/TSP_GA_Project.git
   ```
2. Navigate to the project directory:
   ```
   cd TSP_GA_Project
   ```
3. Ensure all dependencies are installed (see "Requirements" below).
4. Run the `main.py` file to execute the Genetic Algorithm:
   ```
   python main.py
   ```

## 3. File Descriptions
- `main.py`: Main Python script to execute the Genetic Algorithm.
- `.tsp` files: Sample input files containing city coordinates.
- `result.png`: A visualization of the best TSP route found by the algorithm.
- `readme.txt`: Documentation of the project.
- `test.tsp`: A test file for verifying the algorithm.

## 4. Requirements
- Python 3.8 or higher
- Libraries:
  - `matplotlib`
  - `numpy`

To install the required libraries, run:
```
pip install -r requirements.txt
```

## 5. Example Results
The algorithm generates a visualization of the best TSP route found (`result.png`). You can modify the input `.tsp` files to test the algorithm on different datasets.

## 6. License
This project is open-source and available under the MIT License.
