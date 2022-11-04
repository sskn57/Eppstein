import numpy as np
N = 8
def make_graph_data(N):
    for i in range(1, N*N+1):
        # print(i, int((i-1)/N))
        if int((i-1)/N) == N-1:
            #最終列
            if i%N == 0:
                pass
            else:
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+1}] {r}")
        else:
            if i%N == 1:
                # 最初の行
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+1}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N+1}] {r}")
            elif i%N == 0:
                # 最終行
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N-1}] {r}")
            else:
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+1}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N+1}] {r}")
                r = np.random.randint(1, 10)
                print(f"[{i}] [{i+N-1}] {r}")

def make_pos_data(N):
    for i in range(1, N*N+1):
        y = (i-1)%N + 1
        x = int((i-1)/N) + 1
        print(f"{[i]} {x} {y}")

make_graph_data(N)
# make_pos_data(N)