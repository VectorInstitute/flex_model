import logging
from typing import Any

import torch.nn as nn


logger = logging.getLogger(__name__)


class DummyModule(nn.Module):
    """Identity module used to expose activations.

    Can be placed in any `nn.Module` to artificially create an activation to
    be hooked onto. For instance, explicitly calling a module's `.forward()`
    method will not run forward hooks and therefore will not generate an
    activation. However, applying this module to the output of that will
    generate an activation which can be hooked onto.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Any) -> Any:
        return inputs
