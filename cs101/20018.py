import sys
from collections import defaultdict

def merge_sort_count(arr, temp_arr, left, right):
    if left >= right:
        return 0
    mid = (left + right) // 2
    inv_count = 0
    inv_count += merge_sort_count(arr, temp_arr, left, mid)
    inv_count += merge_sort_count(arr, temp_arr, mid + 1, right)
    inv_count += merge_and_count(arr, temp_arr, left, mid, right)
    return inv_count

def merge_and_count(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inv_count = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
            k += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
            k += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1
    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1
    for idx in range(left, right + 1):
        arr[idx] = temp_arr[idx]
    return inv_count

def main():
    n = int(sys.stdin.readline())
    arr = []
    for _ in range(n):
        arr.append(int(sys.stdin.readline()))
    temp_arr = [0] * n
    inv_count = merge_sort_count(arr, temp_arr, 0, n - 1)
    
    count = defaultdict(int)
    for num in arr:
        count[num] += 1
    equal = 0
    for c in count.values():
        equal += c * (c - 1) // 2
    
    total_pairs = n * (n - 1) // 2
    result = total_pairs - inv_count - equal
    print(result)

if __name__ == "__main__":
    main()