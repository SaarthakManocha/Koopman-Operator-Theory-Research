import time
for _ in range(1500):
   with open("tmp.txt", "w") as f: f.write("x" * 10000)
   time.sleep(0.05)
