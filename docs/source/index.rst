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
   demos

Check-out our examples for single and multi-gpu, which use both megatron-style
layers and PyTorch's FSDP wrapper. Additionally, we have a demo which includes
code for induction head identification in Llama-2-70b.


.. toctree::
   :maxdepth: 1
   :caption: Key Python API:

   flex_model.core
   flex_model.distributed
   flex_model.traverse
