import numpy as np
import sys
import argparse

EPS = 1e-9
MAX_IT = 1000

def lu_decompose(A):
    A = A.copy().astype(float)
    n = A.shape[0]
    P = np.arange(n)
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        piv = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[piv, k]) < EPS:
            raise np.linalg.LinAlgError
        if piv != k:
            U[[k, piv], :] = U[[piv, k], :]
            L[[k, piv], :k] = L[[piv, k], :k]
            P[[k, piv]] = P[[piv, k]]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    y = np.zeros(n)
    b = b[P]
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def PrimalSimplex(c, A, b):
    m, n = A.shape
    basis = list(range(n - m, n))
    nbasis = [j for j in range(n) if j not in basis]
    B = A[:, basis]
    P, L, U = lu_decompose(B)
    for _ in range(MAX_IT):
        xB = lu_solve(P, L, U, b)
        cB = c[basis]
        y = np.linalg.solve(B.T, cB)
        rN = c[nbasis] - A[:, nbasis].T @ y
        if np.all(rN <= EPS):
            x = np.zeros(n)
            for i, bi in enumerate(basis):
                x[bi] = xB[i]
            return "optimal", x, c @ x
        pos_idx = np.where(rN > EPS)[0]
        entering_pos = pos_idx[np.argmin([nbasis[i] for i in pos_idx])] if pos_idx.size > 0 else -1
        if entering_pos == -1 or rN[entering_pos] <= EPS:
            return "optimal", None, None
        entering = nbasis[entering_pos]
        a_j = A[:, entering]
        d = lu_solve(P, L, U, a_j)
        if np.all(d <= EPS):
            return "unbounded", None, None
        ratios = [xB[i] / d[i] if d[i] > EPS else np.inf for i in range(m)]
        theta = min(ratios)
        cand_rows = [i for i, r in enumerate(ratios) if abs(r - theta) <= 1e-12]
        if len(cand_rows) == 1:
            leaving_row = cand_rows[0]
        else:
            leaving_row = min(cand_rows, key=lambda i: basis[i])
        leaving = basis[leaving_row]
        B[:, leaving_row] = a_j
        P, L, U = lu_decompose(B)
        basis[leaving_row] = entering
        nbasis[entering_pos] = leaving
    return "infeasible", None, None

def Phase1(c, A, b):
    m, n = A.shape
    A_slack = np.hstack([A, np.eye(m)])
    A_art = np.hstack([A_slack, -np.ones((m, 1))])
    c_art = np.concatenate([np.zeros(n + m), [-1]])
    basis = list(range(n, n + m))
    status, x, obj = PrimalSimplex(c_art, A_art, b)
    if status != "optimal" or obj < -EPS:
        return "infeasible", None, None
    A_ext = A_slack
    c_ext = np.concatenate([c, np.zeros(m)])
    return PrimalSimplex(c_ext, A_ext, b)

def Solve(c, A, b):
    if np.all(b >= -EPS):
        A_ext = np.hstack([A, np.eye(len(b))])
        c_ext = np.concatenate([c, np.zeros(len(b))])
        return PrimalSimplex(c_ext, A_ext, b)
    return Phase1(c, A, b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    with open(args.filename) as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A, b = [], []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
    A = np.array(A)
    b = np.array(b)
    status, x, obj = Solve(c, A, b)
    if status == "optimal":
        print("optimal")
        print(" ".join(f"{v:.10g}" for v in x[:len(c)]))
        print(f"{obj:.10g}")
    elif status == "unbounded":
        print("unbounded")
    else:
        print("infeasible")

if __name__ == "__main__":
    main()
