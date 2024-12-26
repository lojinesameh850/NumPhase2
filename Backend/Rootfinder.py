
import numpy as np
import sympy as sp
from sympy import symbols, diff, sympify
from sympy.parsing.sympy_parser import parse_expr
import time
class ConvergenceError(Exception):
    pass
class RootFinder:
    def bisectionMethod(self, f_expression, xl, xu, epsilon=1e-5, max_iterations=50):
        start = time.time()
        parsedF = parse_expr(f_expression, transformations="all")

        x = symbols('x')

        try:
            f = sympify(parsedF)
        except Exception as e:
            return f"Invalid function expression: {e}"
        f_xl = f.subs(x, xl)
        f_xu = f.subs(x, xu)
        if f_xl * f_xu > 0:
            raise ValueError("The function does not change sign over the given interval. The method cannot proceed.")
        
        iteration = 0
        xr = xl
        error = float('inf')
        
        # Lists to store iteration data
        roots = []
        errors = []
        
        while error > epsilon and iteration < max_iterations:
            xr_old = xr
            xr = (xl + xu) / 2
            f_xr = f.subs(x, xr)
            f_xl = f.subs(x, xl)
            error = abs((xr - xr_old) / xr) if xr != 0 else 0
            iteration += 1
            
            # Store the current root and error
            roots.append(float(xr))
            errors.append(float(error))
            
            if f_xr == 0:
                break
            if f_xl * f_xr < 0:
                xu = xr
            else:
                xl = xr
        
        if iteration == max_iterations:
            raise ConvergenceError("The method did not converge within the maximum number of iterations.")
        else:
            end = time.time()
            elapsed = end - start
            correct_sfs = -int(sp.log(error, 10).evalf()) if error > 0 else 0
            return {
                "root": xr,
                "iterations": iteration+1,
                "relative_error": error,
                "time_taken": elapsed,
                "significant_figures": correct_sfs,
                "all_roots": roots,
                "all_errors": errors,
            }

    def falsePositionMethod(self, f_expression, xl, xu, epsilon=1e-5, max_iterations=50):
        start = time.time()
        parsedF = parse_expr(f_expression, transformations="all")

        x = symbols('x')

        try:
            f = sympify(parsedF)
        except Exception as e:
            return f"Invalid function expression: {e}"
        f_xl = f.subs(x, xl)
        f_xu = f.subs(x, xu)
        if f_xl * f_xu > 0:
            raise ValueError("The function does not change sign over the given interval. The method cannot proceed.")
        
        iteration = 0
        xr = xl
        error = float('inf')
        
        # Lists to store iteration data
        roots = []
        errors = []
        
        while error > epsilon and iteration < max_iterations:
            xr_old = xr
            xr = xu - (f.subs(x, xu) * (xl - xu)) / (f.subs(x, xl) - f.subs(x, xu))
            f_xr = f.subs(x, xr)
            f_xl = f.subs(x, xl)
            error = abs((xr - xr_old) / xr) if xr != 0 else 0
            iteration += 1
            
            # Store the current root and error
            roots.append(float(xr))
            errors.append(float(error))
            
            if f_xr == 0:
                break
            if f_xl * f_xr < 0:
                xu = xr
            else:
                xl = xr
        
        if iteration == max_iterations:
            raise ConvergenceError("The method did not converge within the maximum number of iterations.")
        else:
            end = time.time()
            elapsed = end - start
            correct_sfs = -int(sp.log(error, 10).evalf()) if error > 0 else 0
            return {
                "root": float(xr),
                "iterations": iteration+1,
                "relative_error": float(error),
                "time_taken": elapsed,
                "significant_figures": correct_sfs,
                "all_roots": roots,
                "all_errors": errors,
            }

    def newtonRaphson(self, f, initialGuess, minRelativeError, MaxItretion):
        try:
            start = time.time()
            parsedF = parse_expr(f, transformations="all")

            x = symbols('x')
            finalF = sympify(parsedF)
            diffF = diff(finalF, x)

            xi = initialGuess
            
            # Lists to store iteration data
            roots = []
            errors = []
            
            for i in range(MaxItretion):
                f_value_at_point = finalF.subs(x, xi)
                diffF_at_point = diffF.subs(x, xi)
                xi2 = xi - (f_value_at_point / diffF_at_point)
                
                error = abs((xi2 - xi) / xi) if xi != 0 else 0
                roots.append(float(xi2))
                errors.append(float(error))
                
                if error <= minRelativeError:
                    end = time.time()
                    elapsed = end - start
                    correct_sfs = -int(sp.log(error, 10).evalf()) if error > 0 else 0
                    return {
                        "root": float(xi2),
                        "iterations": i + 1,
                        "relative_error": float(error),
                        "time_taken": elapsed,
                        "significant_figures": correct_sfs,
                        "all_roots": roots,
                        "all_errors": errors,
                    }
                xi = xi2

            raise ConvergenceError("The method did not converge within the maximum number of iterations.")
        except Exception as e:
            return f"Invalid function expression: {e}"

    def ModifiedNewtonRaphson(self, f, initialGuess, minRelativeError, MaxItretion):
        try:
            start = time.time()
            parsedF = parse_expr(f, transformations="all")

            x = symbols('x')
            finalF = sympify(parsedF)
            diffF = diff(finalF, x)
            doubleDiffF = diff(diffF, x)

            xi = initialGuess
            
            # Lists to store iteration data
            roots = []
            errors = []
            
            for i in range(MaxItretion):
                f_value_at_point = finalF.subs(x, xi)
                diffF_at_point = diffF.subs(x, xi)
                doubleDiffF_at_point = doubleDiffF.subs(x, xi)

                xi2 = xi - ((f_value_at_point * diffF_at_point) /
                            ((diffF_at_point)**2 - (f_value_at_point * doubleDiffF_at_point)))

                error = abs((xi2 - xi) / xi) if xi != 0 else 0
                roots.append(float(xi2))
                errors.append(float(error))
                
                if error <= minRelativeError:
                    end = time.time()
                    elapsed = end - start
                    correct_sfs = -int(sp.log(error, 10).evalf()) if error > 0 else 0
                    return {
                        "root": float(xi2),
                        "iterations": i + 1,
                        "relative_error": float(error),
                        "time_taken": elapsed,
                        "significant_figures": correct_sfs,
                        "all_roots": roots,
                        "all_errors": errors,
                    }
                xi = xi2

            raise ConvergenceError("The method did not converge within the maximum number of iterations.")
        except Exception as e:
            return f"Invalid function expression: {e}"

    def fixedPointMethod(self, g_exp, initial_guess, eps=1e-5, max_it=50):
        try:
            x = sp.symbols('x')
            g = sp.sympify(g_exp)
        except Exception as e:
            return {f"Invalid function: {e}"}

        try:
            start_time = time.time()
            current_guess = initial_guess
            iteration = 0
            
            # Lists to store iteration data
            roots = []
            errors = []

            while iteration < max_it:
                next_guess = float(g.subs(x, current_guess))
                absolute_error = abs(next_guess - current_guess)
                relative_error = (absolute_error / abs(next_guess)) if next_guess != 0 else 0

                roots.append(next_guess)
                errors.append(relative_error)

                if relative_error < eps:
                    break

                current_guess = next_guess
                iteration += 1

            execution_time = time.time() - start_time
            if iteration == max_it:
                return {"Maximum iterations reached without convergence."}
            
            correct_sfs = -int(sp.log(relative_error, 10).evalf()) if relative_error > 0 else 0

            return {
                "root": next_guess,
                "iterations": iteration+1,
                "relative_error": relative_error,
                "significant_figures": correct_sfs,
                "execution_time": execution_time,
                "all_roots": roots,
                "all_errors": errors,
            }
        except Exception as e:
            return {f"An error occurred during computation: {e}"}

    def secantMethod(self, f_expression, x0, x1, epsilon=1e-5, max_iterations=50):
        try:
            x = sp.symbols('x')
            f = sp.sympify(f_expression)
        except Exception as e:
            return {f"Invalid function expression: {e}"}

        try:
            start_time = time.time()
            iteration = 0
            
            # Lists to store iteration data
            roots = []
            errors = []

            while iteration < max_iterations:
                f_x0 = float(f.subs(x, x0))
                f_x1 = float(f.subs(x, x1))

                if abs(f_x1 - f_x0) < 1e-12:
                    return {"Division by zero or near-zero detected during the Secant Method."}

                x2 = x1 - ((f_x1 * (x0 - x1)) / (f_x0 - f_x1))
                absolute_error = abs(x2 - x1)
                relative_error = (absolute_error / abs(x2)) if x2 != 0 else 0

                roots.append(float(x2))
                errors.append(relative_error)

                if relative_error < epsilon:
                    break

                x0, x1 = x1, x2
                iteration += 1

            execution_time = time.time() - start_time
            if iteration == max_iterations:
                return {"Maximum iterations reached without convergence."}
            
            correct_sfs = -int(sp.log(relative_error, 10).evalf()) if relative_error > 0 else 0

            return {
                "root": x2,
                "iterations": iteration+1,
                "relative_error": relative_error,
                "significant_figures": correct_sfs,
                "execution_time": execution_time,
                "all_roots": roots,
                "all_errors": errors,
            }
        except Exception as e:
            return {f"An error occurred during computation: {e}"}
