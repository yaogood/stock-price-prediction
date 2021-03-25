

def f (m, n):
    x = int(m)
    y = m-int(m)

    res_x = 0
    res_y = 0
    i = x // n
    reminder = x % n
    while i >= n:
        res_x += i * 10
        i = i // n
    res_x += reminder

    j = y * n
    reminder = j - 1
    count = 1
    while j != 1 and count <= 10:
        res_y += j / 10
        j = j * n
        count += 1
    
    return res_x+res_y


print(f(1.6, 2))