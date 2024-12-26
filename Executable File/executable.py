import subprocess as sp
import threading as th

def backend():
    try:
        sp.run(["python", r"Backend\main.py"])
    except:
        event.set()

def frontend():
    try:
        sp.run(["python", r"Frontend\main.py"])
    except:
        event.set()

if __name__ == "__main__":
    # Start the backend and frontend processes
    backend_process = th.Thread(target=backend)
    frontend_process = th.Thread(target=frontend)
    
    event = th.Event()

    # Start both processes
    backend_process.start()
    frontend_process.start()

    # Wait for both processes to finish
    backend_process.join()
    frontend_process.join()