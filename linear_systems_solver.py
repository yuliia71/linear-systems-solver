import numpy as np


def fmt(x, d=6):
    return f"{x:.{d}f}".replace('.', ',')


def fmt_mat(M, d=4):
    lines = []
    for row in M:
        lines.append("  ".join(f"{v: .{d}f}".replace('.', ',') for v in row))
    return "\n".join(lines)


def fmt_vec(v, d=6):
    return "(" + "; ".join(fmt(x, d) for x in v) + ")"

# 1. МЕТОД ГАУССА З ВИБОРОМ ГОЛОВНОГО ЕЛЕМЕНТА ПО РЯДКАХ

def gauss_with_row_pivoting(A, b):
    print("\n Метод Гаусса з вибором головного елемента по рядках\n")

    n = len(b)
    print("Вихідна розширена матриця A0 | b:")
    print(fmt_mat(np.column_stack([A, b])))
    print()

    detA_pre = np.linalg.det(A)
    if abs(detA_pre) < 1e-12:
        print(f"detA = {fmt(detA_pre)}  (= 0, матриця вироджена, метод застосувати не можемо)")
        return None
    print(f"detA = {fmt(detA_pre)}  (!= 0, матриця невироджена, можемо застосувати метод)")
    print()

    Acur = A.copy()
    bcur = b.copy()
    col_order = list(range(n))
    sign = 1
    pivots = []

    for k in range(n):
        row_k = Acur[k, k:]
        l_rel = np.argmax(np.abs(row_k))
        l = k + l_rel
        print(f"Крок {k + 1}")
        print(f"Рядок {k + 1}: {Acur[k, k:]}, головний елемент на позиції "
              f"стовпця {l + 1} (значення {Acur[k, l]:.6f})")

        Pk = np.eye(n)
        if l != k:
            Pk[[k, l]] = Pk[[l, k]]
            sign *= -1
            col_order[k], col_order[l] = col_order[l], col_order[k]
        print("P_k =")
        print(fmt_mat(Pk, d=0))

        Acur = Acur @ Pk
        print("Ã_k = A_(k-1) * P_k | b:")
        print(fmt_mat(np.column_stack([Acur, bcur])))

        Mk = np.eye(n)
        pivot = Acur[k, k]
        pivots.append(pivot)
        Mk[k, k] = 1 / pivot
        for i in range(k + 1, n):
            Mk[i, k] = -Acur[i, k] / pivot
        print("M_k =")
        print(fmt_mat(Mk))

        Acur = Mk @ Acur
        bcur = Mk @ bcur
        print("A_k = M_k * Ã_k | b:")
        print(fmt_mat(np.column_stack([Acur, bcur])))
        print()

    print("Зворотний хід Гаусса")
    y = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(Acur[i, j] * y[j] for j in range(i + 1, n))
        y[i] = (bcur[i] - s) / Acur[i, i]
        print(f"y_{i + 1} = {fmt(y[i])}")

    x1 = np.zeros(n)
    for pos, orig in enumerate(col_order):
        x1[orig] = y[pos]

    print()
    print("Розв'язок 1 (метод Гаусса з вибором по рядках):", [fmt(v) for v in x1])

    detA = sign
    for p in pivots:
        detA *= p
    print(f"detA = {fmt(detA)}")

    x2 = np.linalg.solve(A, b)
    print("Розв'язок 2:", [fmt(v) for v in x2])
    print(f"Число обумовленості cond(A) = {fmt(np.linalg.cond(A, p = np.inf))}")
    print()
    return x1

# 2. МЕТОД ЯКОБІ

def jacobi(A, b, eps, x0=None):
    
    print("\n Метод Якобі\n")

    n = len(b)
    print("Матриця системи A | b:")
    for row, bi in zip(A, b):
        print("  " + "  ".join(fmt(v) for v in row) + "  |  " + fmt(bi))
    print()

    detA_pre = np.linalg.det(A)
    if abs(detA_pre) < 1e-12:
        print(f"detA = {fmt(detA_pre)}  (= 0, матриця вироджена, метод не можемо застосувати)")
        return None
    print(f"detA = {fmt(detA_pre)}  (!= 0, матриця невироджена, можемо застосувати метод)")
    print()

    print("Побудуємо ітераційний процес:")
    for i in range(n):
        formula_parts = []
        for j in range(n):
            if j != i:
                coef = -A[i, j]
                formula_parts.append(f"{fmt(coef, 1)} * x_{j+1}^k")
        print(f"x_{i+1}^(k+1) = (1 / {fmt(A[i, i], 1)}) * ({' + '.join(formula_parts)} + {fmt(b[i], 2)})")
    print()

    # Визначаємо початкове наближення
    if x0 is None:
        x_prev = np.zeros(n)
        print(f"Оберемо довільне початкове наближення: x^0 = (0, " + ", ".join(["0"] * (n-1)) + ")^T.")
    else:
        x_prev = np.array(x0, dtype=float)
        print(f"Задане початкове наближення: x^0 = {fmt_vec(x_prev)}^T.")
    print()

    k = 0
    while True:
        k += 1
        print(f"Ітерація {k}.")
        
        x_next = np.zeros(n)
        for i in range(n):
            s1 = sum(A[i, j] * x_prev[j] for j in range(i))
            s2 = sum(A[i, j] * x_prev[j] for j in range(i + 1, n))
            x_next[i] = (-s1 - s2 + b[i]) / A[i, i]
            print(f"x_{i+1}^{k} = (1 / {fmt(A[i, i], 1)}) * ({fmt(-s1 - s2, 4)} + {fmt(b[i], 4)}) = {fmt(x_next[i], 4)};")

        diff = x_next - x_prev
        norm = np.max(np.abs(diff))
        
        print("Перевіримо умову припинення:")
        vec_diff_str = "; ".join(fmt(d, 4) for d in diff)
        vec_next_str = fmt_vec(x_next, 4)
        vec_prev_str = fmt_vec(x_prev, 4)
        
        if norm > eps:
            print(f"||x^{k} - x^{{k-1}}|| = ||{vec_next_str} - {vec_prev_str}||_inf = {fmt(norm, 4)} > ε.")
        else:
            print(f"||x^{k} - x^{{k-1}}|| = ||({vec_diff_str})^T||_inf = {fmt(norm, 4)} <= ε.")

        x_prev = x_next
        print()
        if norm <= eps:
            break

    print(f"Розв'язок системи з точністю {eps}: {fmt_vec(x_prev, 4)}^T")
    print()
    x_exact = np.linalg.solve(A, b)
    print(f"Розв'язок 2: {fmt_vec(x_exact)}")
    print(f"Число обумовленості cond(A) = {fmt(np.linalg.cond(A, p = np.inf))}")
    print()
    return x_prev


# ЗАПУСК


if __name__ == "__main__":
    A_gauss = np.array([[1., 3., 2., 1.],
                         [-1., 1., 2., -1.],
                         [3., 3., 1., -3.],
                         [2., 4., 1., -2.]])
    b_gauss = np.array([1., -1., 5., 4.])
    gauss_with_row_pivoting(A_gauss, b_gauss)

    print("\n" * 2)

    A_jacobi = np.array([[6., 3., 2., -1],
                          [-6., -8., 1., -1.],
                          [-1., 4., -10., 5.],
                          [1., -1., -1., 4.]])
    b_jacobi = np.array([10., 11., 12., 13.])
    eps_jacobi = 0.01
    my_x0 = [5., -5.5, -4., 0.] 
    
    jacobi(A_jacobi, b_jacobi, eps_jacobi, x0=my_x0)