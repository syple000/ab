from math import *
import torch

# 验证用文件

def f(x1, x2):
    return (2*x1 + x2) * (x1 - x2) + x2/x1 + x2**(x1+x2+1) + log(x1+x2) + sin(x2-x1)/cos(x1+x2)

def ft():
    x1_ = torch.tensor([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [1, 2],
            [2, 1],
        ]
    ], dtype=float, requires_grad=True)
    x1 = x1_.inverse()
    x2_ = torch.tensor([
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        [
            [6, 5],
            [4, 3],
            [2, 1],
        ]
    ], dtype=float, requires_grad=True)
    x2 = x2_.reshape(2, 2, 3)
    ct = torch.tensor([
        [
            [1, 1, 1]
        ],
        [
            [1, 1, 1]
        ]
    ], dtype=float)
    item1 = x1.matmul(x2)
    item2 = item1.transpose(1, 2)
    item3 = item2.matmul(x2)
    print("item3: {}".format(item3))
    item4 = item3.matmul(ct.transpose(1, 2))
    item5 = item4.transpose(1, 2)
    item = item5.matmul(ct.transpose(1, 2))
    print("item: {}".format(item))
    xgrad = torch.autograd.grad(outputs=item.sum(), inputs=[x1_, x2_], create_graph=True)
    print("xgrad: {}".format(xgrad))
    xgrad_x1_2rd = torch.autograd.grad(outputs=xgrad[0].sum(), inputs=[x1_, x2_], create_graph=True)
    xgrad_x2_2rd = torch.autograd.grad(outputs=xgrad[1].sum(), inputs=[x1_, x2_], create_graph=True)
    print("xgrad_x1_2rd: {}".format(xgrad_x1_2rd))
    print("xgrad_x2_2rd: {}".format(xgrad_x2_2rd))

def ft2():
    x1 = torch.tensor([
        [1, 2],
        [3, 4]
    ], dtype=float, requires_grad=True)
    x2 = torch.tensor([2], dtype=float, requires_grad=True)
    item1 = x1 + x2
    item2 = item1 * x2
    item3 = item2 ** x2
    item4 = item3 / x2 
    item = item4 - x2
    print("item: {}".format(item))
    xgrad = torch.autograd.grad(outputs=item.sum(), inputs=[x1, x2], create_graph=True)
    print("xgrad: {}".format(xgrad))
    xgrad_x1_2rd = torch.autograd.grad(outputs=xgrad[0].sum(), inputs=[x1, x2], create_graph=True)
    xgrad_x2_2rd = torch.autograd.grad(outputs=xgrad[1].sum(), inputs=[x1, x2], create_graph=True)
    print("xgrad_x1_2rd: {}".format(xgrad_x1_2rd))
    print("xgrad_x2_2rd: {}".format(xgrad_x2_2rd))

def ft3():
    x1 = torch.tensor([
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        [
            [0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.7, 0.6, 0.5],
        ]
    ], dtype=float, requires_grad=True)
    item1 = x1.sum(0) # 3 * 3
    item2 = x1.sum(1) # 2 * 3
    item3 = x1.sum(-1) # 2 * 3
    item4 = item2.transpose(0, 1).mm(item3) # 3 * 3
    item5 = item1.mm(item4) # 3 * 3
    item = (item5 * item5).sum() # 1

    print("item: {}".format(item))
    xgrad = torch.autograd.grad(outputs=item, inputs=[x1], create_graph=True)
    print("xgrad: {}".format(xgrad))
    xgrad_x1_2rd = torch.autograd.grad(outputs=xgrad[0].sum(), inputs=[x1], create_graph=True)
    print("xgrad_x1_2rd: {}".format(xgrad_x1_2rd))

def ft4():
    x1 = torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    ])
    print("transpose 0, 1: {}".format(x1.transpose(0, 1)))
    print("transpose 1, 2: {}".format(x1.transpose(1, 2)))
    print("sum 0: {}".format(x1.sum(0)))
    print("sum 0 and expand 0: {}".format(x1.sum(0).unsqueeze(0).expand(x1.shape)))
    print("sum -1 and expand -1: {}".format(x1.sum(-1).unsqueeze(-1).expand(x1.shape)))

if __name__ == "__main__":
    # ft()
    # ft2()
    ft3()
    # ft4()
