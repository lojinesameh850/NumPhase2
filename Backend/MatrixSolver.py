import numpy as np
import sympy as sp
class MatrixSolver:
    import sympy as sp

    def alphabeticalSolution(self, matrix):

        # if len(matrix) != len(set(tuple(row) for row in matrix)):
        #     return "No Solution For Matrix"
    
        # Extract the number of variables from the matrix (assuming a square matrix)
        num_vars = len(matrix)

        # Define symbolic variables for each of the unknowns
        variables = sp.symbols(' '.join([chr(120 + i) for i in range(num_vars)]))  # This will create 'x', 'y', ...

        # Convert string coefficients to symbolic variables
        symbolic_matrix = [
            [sp.symbols(matrix[i][j]) if isinstance(matrix[i][j], str) else matrix[i][j] for j in range(len(matrix[i]))]
            for i in range(len(matrix))
        ]

        # Define the system of equations
        equations = []

        # Iterate through the matrix to create the equations
        for i in range(num_vars):
            equation = sum(symbolic_matrix[i][j] * variables[j] for j in range(num_vars))
            equations.append(sp.Eq(equation, symbolic_matrix[i][-1]))  # The last column is the right-hand side

        # Solve the system of equations
        solutions = sp.solve(equations, variables)

        return solutions

    def handle(self,matrix,b,operator, epsilon, its, x0, significant_digits):
        np.set_printoptions(precision=significant_digits, suppress=True)
        if (np.linalg.det(matrix) == 0):
            return "Matrix is singular , no unique solution exists"
        if operator == 1:
            try :
                x,result = self.Gauss_Elimination(matrix,b)
                return x , result
            except :
                error = self.Gauss_Elimination(matrix,b)
                print(error)
                return error
        elif operator == 2:
            try :
                x,result = self.Gauss_Jordan_Elimination(matrix,b)
                return x, result
            except :
                error = self.Gauss_Jordan_Elimination(matrix,b)
                return error
        elif operator == 3:
            try :
                x , iterations = self.Jacobi(matrix,b, epsilon, its, x0, mode=1)
                return x , iterations
            except :
                error = self.Jacobi(matrix,b, epsilon, its, x0, mode=1)
                return error
        elif operator == 4:
            try :
                x , iterations = self.GaussSeidel(matrix,b, epsilon, its, x0, mode=1)
                return x , iterations
            except :
                error = self.GaussSeidel(matrix,b, epsilon, its, x0, mode=1)
                return error
        elif operator == 5:
            try :
                x,result = self.LUDoolittlesForm(matrix,b)
                return x , result
            except :
                error = self.LUDoolittlesForm(matrix,b)
                return error             
        elif operator == 6:
            try :
                x , LU = self.LUCroutsForm(matrix,b)
                return x , LU
            except :
                error = self.LUCroutsForm(matrix,b)
                return error
        elif operator == 7:
            try :
                x,L = self.LUCholeskyForm(matrix,b)
                return x, L
            except :
                error = self.LUCholeskyForm(matrix,b)
                return error   
        
    def forward_Elimination(self,augmented_matrix,n):
        for i in range(n):
                # Find the row with the maximum absolute value in column i
                max_row = np.argmax(np.abs(augmented_matrix[i:n, i])) + i
                augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

                # Check if the pivot element is zero or very close to zero
                if np.abs(augmented_matrix[i, i]) < 1e-12:
                    return "Pivot element is zero or very small, cannot proceed."

                # Perform the row reduction
                for j in range(i + 1, n):
                    factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
        return augmented_matrix


    def forward_substitution(self, augmented_matrix, n):
        x = np.zeros(n)
        for i in range(n):
            if augmented_matrix[i, i] == 0:
                return "System has no unique solution."
            x[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]
            for j in range(i + 1, n):
                augmented_matrix[j, -1] -= augmented_matrix[j, i] * x[i]
        # Handle negative zero values
        x = np.where(x == -0.0, 0.0, x)

        return x, augmented_matrix       


    def backward_Elimination(self,augmented_matrix,n):
        for i in range(n-1, -1, -1):
                # Normalize the pivot row (make the pivot element equal to 1)
                augmented_matrix[i] /= augmented_matrix[i, i]
                # Eliminate the elements above the pivot
                for j in range(i-1, -1, -1):  # Go through rows above the pivot
                    factor = augmented_matrix[j, i]
                    augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
        return augmented_matrix



    def backward_substitution(self,augmented_matrix,n):
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if augmented_matrix[i, i] == 0:
                return "System has no unique solution."
            x[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]
            augmented_matrix[:i, -1] -= augmented_matrix[:i, i] * x[i]
        # Handle negative zero values
        x = np.where(x == -0.0, 0.0, x)
        return x , augmented_matrix
    

    def Gauss_Elimination(self, A: np.ndarray, B: np.ndarray):
        try:
            n = len(B)
            augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

            augmented_matrix = self.forward_Elimination(augmented_matrix,n)
            
            x, augmented_matrix = self.backward_substitution(augmented_matrix,n)
            return x, augmented_matrix
        except ZeroDivisionError as e:
            print("error occured")
            return str(e)
        


    def Gauss_Jordan_Elimination(self, A: np.ndarray, B: np.ndarray):
        try:
            n = len(B)
            augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

            augmented_matrix = self.forward_Elimination(augmented_matrix,n)

            augmented_matrix = self.backward_Elimination(augmented_matrix,n)
            
            # Handle negative zero values
            augmented_matrix = np.where(augmented_matrix == -0.0, 0.0, augmented_matrix)
            x = augmented_matrix[:,-1]
            return x, augmented_matrix
        except ZeroDivisionError as e:
            return str(e)

        
    def Jacobi(self, A: np.ndarray, b: np.ndarray, epsilon=1e-9, iterations=50, x=None, mode=2):
   
    # if 'mode' is negative it means Gauess Seidel is applied and not Jacobi
        try:
            max_its = iterations
            GaussSeidel = False
            if (mode < 0):
                GaussSeidel = True
                mode = abs(mode)

            # if (mode != 1 and mode != 2):
            #     raise ValueError("Unknown Mode choosen in Jacobi.")

            if (x is None or not isinstance(x, np.ndarray)):
                x = np.zeros(A.shape[0])
            else:
                x = np.array(x, dtype=float)

            D = np.diag(A)
            if np.any(D == 0):
                raise ValueError("Matrix contains zero diagonal elements, Jacobi method cannot proceed.")

            curr_it = 0
            while (True):
                x_new = x.copy()
                for i in range(A.shape[0]):
                    x_new[i] = b[i]
                    for j in range(A.shape[0]):
                        if (i == j):
                            continue
                        if (GaussSeidel): 
                            x_new[i] -= A[i][j] * x_new[j]
                        else: 
                            x_new[i] -= A[i][j] * x[j]
                    x_new[i] /= A[i][i]

            
                if (curr_it > max_its):
                    raise ValueError("Divergence occured, max iterations reached")
                    # return x_new, curr_it-1

                if (np.isinf(np.linalg.norm(np.dot(A, x) - b))):
                    if (GaussSeidel): print("Divergence occured in Gauss Seidel")
                    else: print("Divergence occured in Jacobi")
                    raise ValueError("Divergence occured, overflow happened")
                    # return x_new, curr_it-1

            
                condition = True
                for i in range(A.shape[0]):
                    if (not (abs(x[i] - x_new[i]) < epsilon)):
                        condition = False
                if (condition):
                    # print("Iterations", curr_it-1, x_new)
                    return x_new, curr_it-1


                x = x_new
                curr_it += 1
                
        except ZeroDivisionError as e:
            return str(e)
        except ValueError as e:
            return str(e)
            

    def GaussSeidel(self, A: np.ndarray, b: np.ndarray, epsilon=1e-9, iterations=50, x=None, mode=2):
        return self.Jacobi(A, b, epsilon, iterations, x, mode=-mode)


    
    def LUCroutsForm(self,A: np.ndarray, B: np.ndarray):
        L = np.zeros_like(A)
        U = np.zeros_like(A)
        L_and_U = np.zeros_like(A)
        n = len(B)
        try:
            for i in range(A.shape[0]):
                for j in range(i):
                    L[i][j] = A[i][j]
                    A[i] -= L[i][j] * U[j]
                L[i][i] = A[i][i] #i==j (diagonal)
                if L[i][i] != 0:
                    U[i] = A[i] / L[i][i]
                else:  # infinite number of solution or no solution
                    return "System has no unique solution or no solution."
            #solve the equation        
            augmented_L = np.hstack((L, B.reshape(-1, 1)))
            Y, augmented_L = self.forward_substitution(augmented_L,n)
            augmented_U = np.hstack((U, Y.reshape(-1, 1)))
            X, augmented_U = self.backward_substitution(augmented_U,n)
            for i in range(n):
                    for j in range (n):
                        if i <= j:
                            L_and_U[i][j] = U[i][j]
                        else:
                            L_and_U[i][j] = L[i][j]
            return X, L_and_U
        except ZeroDivisionError as e:
            return str(e)


    def LUCholeskyForm(self,A: np.ndarray, B: np.ndarray):
        L = np.zeros_like(A)
        try : 
            print(f"{np.linalg.cholesky(A)} is the answer")
        except np.linalg.LinAlgError:
            error = "matrix isnt positive definite"
            return error
        try:
            for i in range(A.shape[0]):
                for j in range(i):
                    s=np.sum(L[i][:j]*L[j][:j])
                    L[i][j]=(A[i][j]-s)/L[j][j]
                s=np.sum(L[i][:i]**2)
                L[i][i]=np.sqrt(A[i][i]-s)
            U=L.T   #U = L transpose
            n=len(B)
            #solve the equation     
            augmented_L = np.hstack((L, B.reshape(-1, 1)))
            Y, augmented_L = self.forward_substitution(augmented_L,n)
            augmented_U = np.hstack((U, Y.reshape(-1, 1)))
            X, augmented_U = self.backward_substitution(augmented_U,n)
            return X,L   
        except ZeroDivisionError as e:
            return str(e)     


    def LUDoolittlesForm(self,A: np.ndarray, B: np.ndarray):
        L = np.zeros_like(A, dtype=float)
        U = np.zeros_like(A, dtype=float)
        L_and_U = np.zeros_like(A, dtype=float)
        n = len(B)
        try:
            for i in range(n):
                max_row = np.argmax(np.abs(A[i:n, i])) + i
                A[[i, max_row]] = A[[max_row, i]]
                B[[i, max_row]] = B[[max_row, i]]
                L[i][i] = 1
           
            for i in range(n):
                # Compute Upper Triangular Matrix
                for j in range(i, n):
                    U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
                # Compute Lower Triangular Matrix
                for j in range(i + 1, n):
                    L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
                L[i, i] = 1 
                if U[i, i] == 0:
                    return "Matrix is singular, no unique solution."
            # Storing L and U in one matrix
            for i in range(n):
                for j in range (n):
                    if i <= j:
                        L_and_U[i][j] = U[i][j]
                    else:
                        L_and_U[i][j] = L[i][j]
                
            #solve the equation        
            Y = np.zeros(n)
            for i in range(n):
                Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
            X = np.zeros(n)
            for i in range(n - 1, -1, -1):
                X[i] = (Y[i] - np.dot(U[i, i + 1:], X[i + 1:])) / U[i, i]
            return X , L_and_U
        
        except ZeroDivisionError as e:
            return str(e) 