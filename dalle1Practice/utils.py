import attr
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logit_laplace_eps: float = 0.1

@attr.s(eq=False)
class Conv2d(nn.Module):
	n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
	n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
	kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

	use_float16:   bool         = attr.ib(default=True)
	device:        torch.device = attr.ib(defa2ult=torch.device('cpu'))
	requires_grad: bool         = attr.ib(default=False)
	
    