#!/usr/bin/python3
import os
from subprocess import Popen, PIPE

def run_process_2(cmd, prefix=None, suffix=None, cwd = None,
                    use_pipe=False, use_shell = False):
    if cwd == None:
        cwd = os.getcwd()
    print("Running process with command: {}".format(cmd),flush=True)
    cmd_pcs = cmd.split()
    p = Popen(cmd_pcs, cwd = cwd, 
            stdout = PIPE if use_pipe else None,
            stderr = PIPE if use_pipe else None,
            shell = use_shell)
    return p

if __name__ == "__main__": 
    command = 'python inference.py' 
    all_process = {}
    p_id = 0 
    p = run_process_2(model)
    all_process["P{}".format(p_id)] = [p, True]
    p.wait()   

