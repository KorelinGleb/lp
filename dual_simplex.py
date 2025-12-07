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

def dual_simplex_no_phase1(c, A, b, U=1e10):
    m, n = A.shape
    
    A_full = np.hstack([A, np.eye(m)])
    c_full = np.concatenate([c, np.zeros(m)])
    n_full = n + m
    
    basis = list(range(n, n_full))
    nbasis = list(range(n))
    
    B = A_full[:, basis].copy()
    try:
        P, L, U_decomp = lu_decompose(B)
    except np.linalg.LinAlgError:
        return "infeasible", None, None
    
    U_vec = np.full(n_full, U)
    
    cB = c_full[basis]
    y = np.linalg.solve(B.T, cB)
    N_mat = A_full[:, nbasis]
    c_bar_N = c_full[nbasis] - N_mat.T @ y
    z_star_N = -c_bar_N
    
    x_star_N = np.zeros(len(nbasis))
    d = np.ones(len(nbasis))
    
    for j_idx, j in enumerate(nbasis):
        if c_bar_N[j_idx] > EPS:
            x_star_N[j_idx] = U_vec[j]
            d[j_idx] = -1
        else:
            x_star_N[j_idx] = 0.0
            d[j_idx] = +1
    
    r = d * c_bar_N
    
    x_star_B = lu_solve(P, L, U_decomp, b - N_mat @ x_star_N)
    
    for iteration in range(MAX_IT):
        if np.all(x_star_B >= -EPS) and np.all(r <= EPS):
            x = np.zeros(n_full)
            for i, bi in enumerate(basis):
                x[bi] = x_star_B[i]
            for j_idx, j in enumerate(nbasis):
                x[j] = x_star_N[j_idx]
            
            for j_idx, j in enumerate(nbasis):
                if j < n:
                    if abs(x_star_N[j_idx] - U_vec[j]) < EPS:
                        if d[j_idx] == -1:
                            if c_bar_N[j_idx] > EPS:
                                return "unbounded", None, None
            
            return "optimal", x[:n], c @ x[:n]
        
        violating_rows = [i for i in range(m) if x_star_B[i] < -EPS]
        if not violating_rows:
            if np.all(r <= EPS):
                x = np.zeros(n_full)
                for i, bi in enumerate(basis):
                    x[bi] = x_star_B[i]
                for j_idx, j in enumerate(nbasis):
                    x[j] = x_star_N[j_idx]
                return "optimal", x[:n], c @ x[:n]
            else:
                return "infeasible", None, None
        
        min_val = min(x_star_B[i] for i in violating_rows)
        tied = [i for i in violating_rows if abs(x_star_B[i] - min_val) <= 1e-12]
        i_row = min(tied, key=lambda r: basis[r])
        
        a_hat_row = np.array([lu_solve(P, L, U_decomp, A_full[:, j])[i_row] for j in nbasis])
        
        sigma = -a_hat_row * d
        
        candidate_indices = [j_idx for j_idx in range(len(nbasis)) if sigma[j_idx] > EPS]
        
        if not candidate_indices:
            return "infeasible", None, None
        
        ratios = [-r[j_idx] / sigma[j_idx] for j_idx in candidate_indices]
        t = min(ratios)
        
        if t < -EPS:
            return "infeasible", None, None
        
        tied_j = [candidate_indices[idx] for idx, ratio in enumerate(ratios) if abs(ratio - t) <= 1e-12]
        if len(tied_j) == 1:
            j_star_idx = tied_j[0]
        else:
            j_star_idx = min(tied_j, key=lambda idx: nbasis[idx])
        
        j_star = nbasis[j_star_idx]
        
        a_j_star = lu_solve(P, L, U_decomp, A_full[:, j_star])
        delta_xB = -a_j_star * d[j_star_idx]
        
        x_star_B = x_star_B - t * a_j_star * d[j_star_idx]
        r = r + t * sigma
        
        leaving = basis[i_row]
        basis[i_row] = j_star
        
        B[:, i_row] = A_full[:, j_star]
        try:
            P, L, U_decomp = lu_decompose(B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None
        
        nbasis[j_star_idx] = leaving
        
        x_star_N[j_star_idx] = 0.0
        d[j_star_idx] = +1
        
        cB = c_full[basis]
        y = np.linalg.solve(B.T, cB)
        N_mat = A_full[:, nbasis]
        c_bar_N = c_full[nbasis] - N_mat.T @ y
        r = d * c_bar_N
        
        x_star_B = lu_solve(P, L, U_decomp, b - N_mat @ x_star_N)
    
    return "infeasible", None, None

def solve(c, A, b):
    status, x, obj = dual_simplex_no_phase1(c, A, b)

    return status, x, obj

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
    status, x, obj = solve(c, A, b)
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
