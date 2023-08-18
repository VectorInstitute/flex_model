import torch
import torch.nn as nn


model = nn.Linear(5120, 6969).cuda().to(torch.bfloat16)
inputs = torch.randn(4, 512, 5120).cuda().to(torch.bfloat16)

output_0 = model(inputs)

chunks = inputs.chunk(2, dim=0)
output_1 = model(chunks[0])
output_2 = model(chunks[1])
output_3 = torch.cat((output_1, output_2), dim=0)

breakpoint()
