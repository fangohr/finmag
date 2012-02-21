#ifndef __FINMAG_UTIL_PYTHON_THREADING_H
#define __FINMAG_UTIL_PYTHON_THREADING_H

 namespace finmag { namespace util {
    // Releases the GIL
    class scoped_gil_release
    {
    public:
        scoped_gil_release(): thread_state(0)  {
            thread_state = PyEval_SaveThread();
        }

        ~scoped_gil_release() {
            PyEval_RestoreThread(thread_state);
            thread_state = 0;
        }

    private:
        PyThreadState * thread_state;
    };

    // Acquires the gil if necessary
    class scoped_gil_ensure
    {
    public:
        scoped_gil_ensure() { gstate = PyGILState_Ensure(); }

        ~scoped_gil_ensure() { PyGILState_Release(gstate); }

    private:
        PyGILState_STATE gstate;
    };

 }}

 #endif
