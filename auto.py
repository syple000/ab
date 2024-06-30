from math import *

def f(x1, x2):
    return (2*x1 + x2) * (x1 - x2) + x2/x1 + x2**(x1+x2+1) + log(x1+x2) + sin(x2-x1)/cos(x1+x2)

if __name__ == "__main__":
    x1 = 3
    x2 = 4
    delta = 0.00001
    print("f(x1=1, x2=2) = ", f(x1, x2))
    print("dx1|x1=1, x2=2, = ", (f(x1+delta, x2)-f(x1, x2))/delta)
    print("dx2|x1=1, x2=2, = ", (f(x1, x2+delta)-f(x1, x2))/delta)
    df_dx1_pos = (f(x1+delta, x2)-f(x1, x2))/delta
    df_dx1_neg = (f(x1, x2)-f(x1-delta, x2))/delta
    print("dx1*2|x1=1, x2=2, = ", (df_dx1_pos - df_dx1_neg)/delta)
    df_dx2_pos = (f(x1, x2+delta)-f(x1, x2))/delta
    df_dx2_neg = (f(x1, x2)-f(x1, x2-delta))/delta
    print("dx2*2|x1=1, x2=2, = ", (df_dx2_pos-df_dx2_neg)/delta)

    df_dx1_dx2 = ((f(x1+delta, x2+delta)-f(x1, x2+delta))/delta - (f(x1+delta, x2)-f(x1, x2))/delta)/delta
    print("dx1*dx2| x1=1, x2=2, = ", df_dx1_dx2)
    df_dx2_dx1 = ((f(x1-delta, x2-delta)-f(x1-delta, x2))/delta - (f(x1, x2-delta)-f(x1, x2))/delta)/delta
    print("dx2*dx1| x1=1, x2=2, = ", df_dx2_dx1)
