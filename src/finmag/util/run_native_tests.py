if __name__ == "__main__":
    import os
    import sys
    from finmag.util.native_compiler import pipe_output

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../../native")
    res = pipe_output("make test")
    sys.exit(res)
