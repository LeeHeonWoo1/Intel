import matplotlib.pyplot as plt

def hail(n):
    result_list = []
    while n != 1 :
        if n%2 == 0:
            result_list.append(n)
            n = int(n / 2)
        else:
            result_list.append(n)
            n = n*3 + 1
    
    result_list.append(1)
    return result_list

plt.plot(hail(331))
plt.show()