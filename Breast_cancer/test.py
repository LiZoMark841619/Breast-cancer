from collections import deque
import os

def get_valid_number(min_value: int, max_value: int) -> int:
    while True:
        try:
            value = int(input())
            if value in range(min_value, max_value+1):
                return value
        except ValueError:
            print('Invalid value! Try again with iteger! ')

def deque_updating(some_deque: deque, max_length: int) -> deque:
    while len(some_deque) != max_length:
        some_deque.extend(map(int, input().split()))
    return some_deque

def pop_left_right(some_deque: deque) -> bool:
    while some_deque:
        left_most = some_deque.popleft()
        right_most = some_deque.pop()
        return some_deque[0] <= left_most and some_deque[-1] <= right_most

if __name__ == '__main__':
    with open(os.environ['OUTPUT_PATH'], 'w') as fptr:
        T = get_valid_number(min_value=1, max_value=5)
        for _ in range(T):
            some_deque = deque()
            n = get_valid_number(min_value=1, max_value=10**5)
            updated_deque = deque_updating(some_deque, n)
            result = pop_left_right(updated_deque)
            fptr.write('Yes\n' if result is True else 'No\n')
    fptr.close()
    