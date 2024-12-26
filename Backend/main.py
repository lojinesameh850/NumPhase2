from calendar import c
from flask import Flask  , request
from flask.json import jsonify
import numpy as np
import time

# from torch import res
from MatrixSolver import MatrixSolver
from flask_cors import CORS

from Rootfinder import ConvergenceError, RootFinder

app = Flask(__name__)
CORS(app)
@app.route('/alphabetical',methods =["POST"])
def alpha():
    data = request.get_json()
    matrix = data.get("matrix")
    Solver = MatrixSolver()
    result = Solver.alphabeticalSolution(matrix)
    try:
        result_serializable = {str(key): str(value) for key, value in result.items()}
    except:
        if result == "No Solution For Matrix":
             return {"error" : result}
        else:
             result_serializable = result
    print(result_serializable)
    return result_serializable
@app.route('/', methods=["POST"])
def home():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        # Extract data from JSON
        augmented_matrix = data.get("matrix")
        if not augmented_matrix:
            return jsonify({"error": "Matrix data is missing"}), 400
        augmented_matrix = np.array(augmented_matrix, dtype=float)

        x0 = data.get("x0")
        x0 = np.array(x0)
        # print(x0)
        # return jsonify("hello")
        significant_digits = data.get("significant_digits")
        # mode = data.get("mode")
        its = data.get("its")
        epsilon = data.get("epsilon")
        operation = int(data.get("operation", 1)) 

        # Process the augmented matrix
        matrix = augmented_matrix[:, :-1]
        b = augmented_matrix[:, -1]
        Solver = MatrixSolver()
        start = time.time()
        try :
            if operation == 1 or operation == 2 or operation == 5 or operation == 6:
                    x , result = Solver.handle(matrix , b , operation , epsilon , its , x0 , significant_digits)
                    end = time.time()
                    elapsed =  end - start
                    elapsed = float(np.clip(elapsed , 1e-6, None))
                    print(x , result , elapsed , sep="\n")
                    return jsonify({"x" : x.tolist(),
                                    "result" : result.tolist(),
                                    "time_taken" : elapsed})
            elif operation == 3 or operation == 4:
                    x , iterations = Solver.handle(matrix , b , operation , epsilon , its , x0 , significant_digits)
                    end = time.time()
                    elapsed =  end - start
                    elapsed = float(np.clip(elapsed , 1e-6, None))
                    print(x , elapsed , sep="\n")
                    return jsonify({"x" : x.tolist(),
                                    "time_taken" : elapsed , 
                                    "iterations" : iterations,
                                    "time_taken" : elapsed})
            elif operation == 7:
                    x,L = Solver.handle(matrix , b , operation , epsilon , its , x0 , significant_digits)
                    end = time.time()
                    elapsed =  end - start
                    elapsed = float(np.clip(elapsed , 1e-6, None))
                    print(x , L , elapsed , sep="\n")
                    return jsonify({"x" : x.tolist(),
                                    "L" : L.tolist(),
                                    "time_taken" : elapsed})
        except:
            error = Solver.handle(matrix , b , operation , epsilon , its , x0 , significant_digits)
            return jsonify({"error" : error})
    except:
        return jsonify({"error" : "an error has occured"})
@app.route('/roots', methods=["POST"])
def root():
    data = request.get_json()
    print(request.get_json())

    function = data.get("function")
    operation = data.get("operation")
    significat_digits = data.get("significant_digits")
    max_its = data.get("max_its")
    x0 = data.get("x0")
    epsilon = data.get("epsilon")
    x1 = data.get("x1")
    xl = data.get("xl")
    xu = data.get("xu")
    Root = RootFinder()
    result = None
    if operation == 1:
        try:
            result = Root.bisectionMethod(function , xl , xu , epsilon , max_its)
            return jsonify({"result" : result})
        except ConvergenceError as e:
            return jsonify({"error" : str(e)})
        except ValueError as e:
            return jsonify({"error" : str(e)})
        except:
            result = str(Root.bisectionMethod(function , xl , xu , epsilon , max_its))
            return jsonify({"error" : result})
    elif operation == 2:
        try:
            result = Root.falsePositionMethod(function , xl , xu , epsilon , max_its)
            return jsonify({"result" : result})
        except ConvergenceError as e:
            return jsonify({"error" : str(e)})
        except ValueError as e:
            return jsonify({"error" : str(e)})
        except Exception as e:
            print(e)
            result = str(Root.falsePositionMethod(function , xl , xu , epsilon , max_its))
            print(result)
            return jsonify({"error" : result})
    elif operation == 3:
        try:
            result = Root.fixedPointMethod(function , x0 , epsilon , max_its)
            return jsonify({"result" : result})
        except:
            result = str(Root.fixedPointMethod(function , x0 , epsilon , max_its))
            return jsonify({"error" : result})
    elif operation == 4:
        try:
            result = Root.newtonRaphson(function , x0 , epsilon , max_its)
            return jsonify({"result" : result})
        except:
            result = str(Root.newtonRaphson(function , x0 , epsilon , max_its))
            return jsonify({"error" : result})
    elif operation == 5:
        try:
            result = Root.ModifiedNewtonRaphson(function , x0 , epsilon , max_its)
            return jsonify({"result" : result})
        except:
            result = str(Root.ModifiedNewtonRaphson(function , x0 , epsilon , max_its))
            return jsonify({"error" : result})
    elif operation == 6:
        try:
            result = Root.secantMethod(function , x0 , x1 , epsilon , max_its)
            return jsonify({"result" : result})
        except:
            result = str(Root.secantMethod(function , x0 , x1 , epsilon , max_its))
            return jsonify({"error" : result})
    
if __name__ == "__main__":
    app.run(debug=True)
