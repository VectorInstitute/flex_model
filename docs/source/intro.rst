Introduction
============
``FlexModel`` is a simple-to-use and robust wrapper for model surgery and
introspection. It provides a few powerful primitives for activation retrieval,
activation editing and auxillary module training. Users can define their own
``HookFunction`` instances, which provide a single-threaded runtime for
manipulating layer activations. The entire wrapper and hook runtime can be
used under arbitrary distributed model topologies.


Motivation
**********
Repositories for mechanistic interpretability have very built-out feature sets,
however they do not provide many utilities for using models that are
distributed in potentially complicated ways. This framework is intended to
provide these same utilities in a way which is robust to potentially
complicated distributed strategies.


Limitations
***********
Being a new framework, we currently lack many ease-of-use and high-level
features that more mature frameworks include. However, we hope that the
well-tested primitives currently exposed are powerful enough where these
features will be simple to implement.


Additionally, there is only support for up to 3-D distributed parallelism.
