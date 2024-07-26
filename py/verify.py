# 验证结果，与auto.py进行比对即可

import ae

if __name__ == "__main__":
    x1_ = ae.tensor([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [1, 2],
            [2, 1],
        ]
    ], True)
    x1 = x1_.inverse()
    x2_ = ae.tensor([
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
    ], True)
    x2 = x2_.reshape([2, 2, 3])
    ct = ae.tensor([
        [
            [1, 1, 1]
        ],
        [
            [1, 1, 1]
        ]
    ])
    item1 = x1.mm(x2)
    item2 = item1.transpose()
    item3 = item2.mm(x2)
    print("item3: {}".format(item3))
    item4 = item3.mm(ct.transpose())
    item5 = item4.transpose()
    item = item5.mm(ct.transpose())
    print("item: {}".format(item))
    item.backward()
    print("xgrad: {}, {}".format(x1_.grad(), x2_.grad()))
    item.clear_grad()
    item.create_grad_graph()
    x1_grad = x1_.grad_graph()
    x2_grad = x2_.grad_graph()
    item.clear_grad_graph()
    print("xgrad v2: {} {}".format(x1_grad, x2_grad))
    x1_grad.backward()
    print("xgrad_x1_2rd: {} {}".format(x1_.grad(), x2_.grad()))
    x1_grad.clear_grad()
    x2_grad.backward()
    print("xgrad_x2_2rd: {} {}".format(x1_.grad(), x2_.grad()))
    x2_grad.clear_grad()

    a = ae.tensor([[1, 2], [3, 4]])
    b = ae.tensor([3, 4])
    b.update(a)
    print("b: {}, sum: {}, add1: {}, sub1: {}, mul2: {}, div2: {}, pow2: {}".format(b, b.sum(), b + 1, b - 1, b * 2, b / 2, b ** 2))
