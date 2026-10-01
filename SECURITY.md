# Security

## Reporting a vulnerability

Report suspected vulnerabilities privately to the project maintainers rather
than in a public issue. Include the version, the platform, and the smallest
input that reproduces the problem. You will get an acknowledgement within a few
days and an assessment of severity and timeline.

## The threat model, stated plainly

Materia is a single-user local application. It binds a loopback HTTP endpoint on
127.0.0.1 to carry messages between its own interface and its own computational
core. That endpoint is not advertised, is not reachable from the network, and
closes with the window. There is no account system, no server and no telemetry.

### What is *not* a security boundary

**The restricted Python mode is a guard rail, not a sandbox.** CPython cannot be
sandboxed from inside: a determined script can reach interpreter internals
through object introspection regardless of what is removed from `builtins`.
Restricted mode blocks the common accidents, importing `os`, calling `open`,
`eval` or `exec`, and nothing more. Treat a script you are about to run the way
you would treat any other program you are about to run. The application states
this before it lets you switch to trusted mode, and records the switch in the
project history.

**Plug-ins run with the privileges of the application.** Loading a plug-in
executes its code. Materia shows what it is about to load and only searches
directories you configured. It does not sandbox plug-ins.

**Project files contain executable Python if you put it there.** A `.materia`
archive can carry saved scripts. Opening a project does not run them; running
one is an explicit action.

### What is in scope

* Path traversal or arbitrary file access through the HTTP layer.
* Any endpoint that executes code without an explicit user action.
* Binding to an interface other than loopback.
* Any outbound network connection. Materia makes none; if you find one, that is
  a vulnerability.
* Denial of service that a malformed project file, material definition or
  structure file can cause on load.

### What is out of scope

* Scripts the user chose to run, in either execution mode.
* Plug-ins the user chose to install.
* External solvers the user chose to install, and their own vulnerabilities.
* Resource exhaustion from a calculation the user deliberately requested.

## Supply chain

Runtime dependencies are NumPy and SciPy. The native window uses pywebview,
which wraps the operating system's own web-view component. Optional extras are
ASE and h5py. External solver adapters communicate with packages you install
yourself through their public interfaces or input and output files; no external
solver code is vendored or linked. `requirements-lock.txt` records the exact
versions used to produce the reported results.
