def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

def weighted_edit_distance(s1, s2, insert_cost, delete_cost, replace_cost):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j * insert_cost
            elif j == 0:
                dp[i][j] = i * delete_cost
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1] + insert_cost,
                               dp[i - 1][j] + delete_cost,
                               dp[i - 1][j - 1] + replace_cost)

    return dp[m][n]

s1 = "kitten"
s2 = "sitting"
print("Edit Distance:", edit_distance(s1, s2))
print("Weighted Edit Distance:", weighted_edit_distance(s1, s2, 1, 1, 2))
