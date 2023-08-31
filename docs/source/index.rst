.. FlexModel documentation master file, created by
   sphinx-quickstart on Wed Aug 30 14:56:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlexModel's documentation!
=====================================

FlexModel is a wrapper for Pytorch models which exposes powerful primitives
for model surgery and introspection.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   intro
   examples


.. toctree::
   :maxdepth: 1
   :caption: Python API:

   flex_model.core
   flex_model.distributed
   flex_model.traverse


TODOs
-----
* Add documentation for :code:`flex_model.traverse` :code:`flatten` and :code:`unflatten`
  functions.
* Support for backward hooks.
* Finish tests for main codebase classes and functions.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
