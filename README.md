# 🧮 Numerical Methods: Solving Systems of Linear Equations

Direct and iterative solvers for **SLAE** (systems of linear algebraic equations) in **Python**: Gauss elimination with row pivoting and the Jacobi method, with a detailed step-by-step output of every stage.

## 1. Project Introduction

This project is a lab work on numerical linear algebra. It shows how to solve `Ax = b` by Gauss and Jacobi algorithms and verifies the results against NumPy, printing all intermediate matrices so the solution can be followed and checked step by step.

## 2. Project Overview and Features

| Method | Type | What it does |
|---|---|---|
| `gauss_with_row_pivoting` | Direct | Gauss elimination with main-element selection by rows, using permutation matrices `P_k` and elimination matrices `M_k`, followed by back substitution |
| `jacobi` | Iterative | Jacobi method with a configurable initial guess `x0` and stopping criterion `‖x^k − x^(k−1)‖∞ ≤ ε` |

**Features**

- Check that `det A ≠ 0` before applying a method
- Step-by-step output: `P_k`, `Ã_k`, `M_k`, `A_k` for each Gauss step
- Printed iteration formulas and per-iteration values for Jacobi
- Determinant computed from the pivots and the permutation sign
- Cross-check with `numpy.linalg.solve`
- Condition number `cond(A)` in the infinity norm
- Output formatted with a decimal comma (`fmt`, `fmt_mat`, `fmt_vec` helpers)

## 3. Tech Stack, APIs, and Other Resources

- Python 3.8+
- NumPy
- No external APIs

## 4. Getting Started: Setup and Running Instructions

```bash
git clone https://github.com/yuliia71/linear-systems-solver.git
cd linear-systems-solver
pip install numpy
python Labaig.py
```

1. Edit the test data in the `if __name__ == "__main__":` block: matrices `A_gauss`, `A_jacobi`, vectors `b_gauss`, `b_jacobi`.
2. For Jacobi, set the accuracy `eps_jacobi` and the initial approximation `my_x0` (or pass `x0=None` to start from zero).
3. Run the script and read the report in the console.

> ⚠️ **Note:** the Jacobi method is guaranteed to converge only for matrices with strict diagonal dominance, so check this condition when using your own data.
