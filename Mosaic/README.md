# Mosaic Solver

This repository contains a Python script `mosaic_solver.py` that solves Mosaic Logic Puzzles using different search algorithms: Depth-First Search (DFS), Breadth-First Search (BFS), and A* Search.

## Usage

### Requirements
Ensure you have Python installed on your system.

### Running the Script
To execute the script, use the following command:

```sh
python mosaic_solver.py <testcase> <algorithm>
```

#### Arguments:
- `<testcase>`: The name of the test case file (without extension) located in the `testcases/` directory. Default: `tc1`
- `<algorithm>`: The search algorithm to use. Options: `DFS`, `BFS`, `A*`. Default: `DFS`

#### Example Usage:
```sh
python mosaic_solver.py tc4 BFS
```
This will solve `testcases/tc4.txt` using the BFS algorithm.

## Implementation Details
- **Depth-First Search (DFS):** Explores each path deeply before backtracking.
- **Breadth-First Search (BFS):** Expands nodes level by level.
- **A* Search:** Uses heuristics to optimize the solution search.

## File Structure
```
|-- mosaic_solver.py  # Main script
|-- testcases/
    |-- tc1.txt  # Example test case
    |-- tc4.txt  # Another test case
|-- README.md  # Documentation
```

## Notes
- Ensure your test case files are properly formatted.
- Modify the `Mosaic` class if needed to support additional puzzle constraints.


