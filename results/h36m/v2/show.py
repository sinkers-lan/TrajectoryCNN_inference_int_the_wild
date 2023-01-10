import numpy as np
import matplotlib.pyplot as plt
# img_gen = np.load('img_gen.npy')
# plt.figure(figsize=(16,8))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     x = img_gen[40, i, :, 0]
#     y = img_gen[40, i, :, 1]
#     plt.scatter(x,y)
#     list1 = [0,1,2,3,4,5]
#     plt.plot(x[list1], y[list1])
#     list2 = [6,7,8,9]
#     plt.plot(x[list2], y[list2])
#     list3 = [11,10,9,12,13]
#     plt.plot(x[list3], y[list3])
# plt.show()
test_out = np.load('../test.npy')
plt.figure(figsize=(16, 80))
start = 200
end = start + 100
for i in range(start, end):
    plt.subplot(20, 5, i + 1 - start)
    x = test_out[i, :, 0]
    y = test_out[i, :, 1]
    plt.scatter(x, y)
    list1 = [10, 9, 8, 7, 0]
    plt.plot(x[list1], y[list1])
    list2 = [16, 15, 14, 8, 11, 12, 13]
    plt.plot(x[list2], y[list2])
    list3 = [3, 2, 1, 0, 4, 5, 6]
    plt.plot(x[list3], y[list3])
# plt.savefig('../test1.png')
plt.show()
