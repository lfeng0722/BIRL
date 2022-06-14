import torch


def phi (M, a_dim, s_dim):
    phi_out = []

    for h in range(M):
        w1 = torch.randn((a_dim+s_dim,64), requires_grad=True,device='cuda:0')
        w2 = torch.randn((64, 64), requires_grad=True,device='cuda:0')
        w3 = torch.randn((64, 1), requires_grad=True,device='cuda:0')

        b1 = torch.randn((64), requires_grad=True,device='cuda:0')
        b2 = torch.randn((64), requires_grad=True,device='cuda:0')
        b3 = torch.randn((1), requires_grad=True,device='cuda:0')

        phi_0 =  w1,b1,w2,b2,w3,b3

        phi_out.append(phi_0)
        # phi_out=torch.stack(phi_0)

    return phi_out

