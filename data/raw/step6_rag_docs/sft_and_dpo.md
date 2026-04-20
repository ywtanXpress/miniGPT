# SFT and DPO

Supervised fine-tuning, or SFT, teaches a pretrained model to follow instructions.
The dataset usually contains an instruction, optional input context, and a target response.
The prompt is formatted in a consistent template, and the model is trained to predict the assistant response.

One useful detail is prefix-only supervision.
That means the loss is applied mainly to the assistant portion of the sequence instead of the entire prompt.
This keeps the objective focused on the part of the output the model is supposed to generate.

Direct Preference Optimization, or DPO, is a later alignment step.
Instead of training on a single target answer, DPO compares a chosen response with a rejected response for the same prompt.
The policy model is encouraged to score the chosen answer more highly than the rejected one, relative to a frozen reference model.

DPO is attractive because it avoids the extra reward-model training step used in some RLHF pipelines.
It is still important to remember that DPO does not magically add new factual knowledge.
It mostly changes the model's preference behavior, style, and ranking of candidate responses.

A practical takeaway is:
pretraining teaches language,
SFT teaches task formatting and instruction following,
and DPO nudges the model toward preferred responses.
