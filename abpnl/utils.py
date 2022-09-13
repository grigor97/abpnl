def compute_wrong_orders(cusal_order, A):
    wrong = 0
    for i in range(A.shape[0] - 1):
        for j in range(i+1, A.shape[0]):
            if A[cusal_order[j], cusal_order[i]] == 1:
                wrong += 1

    return wrong
