def find_non_dominated_points(X):
    P = []  # Lista punktów niezdominowanych
    number_of_compares = 0
    n_o_cord = len(X[0])
    
    while len(X) > 0:
        Y = X[0]
        j = 1
        id_y = 0
        
        while j < len(X):
            greater = 0
            lower = 0

            for x, y in zip(X[j], Y):
                number_of_compares += 1
                if x >= y:
                    greater += 1
                if x <= y:
                    lower += 1
            if greater == n_o_cord:
                X.pop(j)
            elif lower == n_o_cord:
                Y = X[j]
                X.pop(id_y)
                id_y = j - 1
            else:
                j += 1
        
        if Y not in P:
            P.append(Y)

        X = [xk for xk in X if not all(y <= xk_i for y, xk_i in zip(Y, xk))]
        
        if len(X) == 1:
            P.append(X[0])
            break
    
    return P, number_of_compares


X = [(5,5), (3,6), (4,4), (5,3), (3,3), (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3, 5)]
P, number_of_compares = find_non_dominated_points(X)
print(P, number_of_compares)
