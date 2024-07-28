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

if __name__ == "__main__":
    ft()
    ft2()
