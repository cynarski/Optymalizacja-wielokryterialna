def find_non_dominated_points(X):
    P = []  # Lista punktów niezdominowanych
    number_of_compares = 0
    n = len(X)
    k = len(X[0]) if n > 0 else 0

    xmin = [min(x[i] for x in X) for i in range(k)]
    distances = [(sum((x[i] - xmin[i]) ** 2 for i in range(k)), j) for j, x in enumerate(X)]
    distances.sort()
    
    visited_indices = set()

    for _, index in distances:
        if index in visited_indices:
            continue

        current_point = X[index]
        P.append(current_point)

        for i, x in enumerate(X):
            number_of_compares += 1
            if i not in visited_indices:
                if all(c <= xi for c, xi in zip(current_point, x)):
                    number_of_compares += k - 1
                    visited_indices.add(i)

    return P, number_of_compares


X = [(5,5), (3,6), (4,4), (5,3), (3,3), (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3, 5)]
# P, number_of_compares = find_non_dominated_points(X)
# print(P, number_of_compares)
