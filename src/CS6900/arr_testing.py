import numpy as np

arr = np.array([
    [[5, 8, 3], [9, 4, 2]],
    [[3, 9, 5], [2, 6, 8]]])
arr1 = np.mean(arr, axis=(0, 1))

arr2 = np.random.random((2,3,4))

print(arr)
print(arr[0, 1, 1])
print(arr1)
print((5+9+3+2)/4)
print(arr2)
print(arr2[0,1,1])
print(arr2[0][1][1])

arr3 = np.ones((10,11,5))
arr3[0,0,4] = 10

arr4 = np.mean(arr3, axis=2)

arr5 = np.array([1,2,3,4,5])
arr6 = np.array([6,7,8,9,10])

print(arr4)
print(arr5*arr6)