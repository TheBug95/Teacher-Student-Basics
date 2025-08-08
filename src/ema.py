import torch

@torch.no_grad()
def update_ema(student, teacher, alpha=0.99):
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(alpha).add_(s_p.data, alpha=1-alpha)
