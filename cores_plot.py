import numpy as np
import matplotlib.pyplot as plt

n_cores = [1,20,40,60,80,100,128]
run_times = [3968.925694465637, 239.18060326576233, 93.63527464866638, 67.5806736946106, 64.05664825439453, 55.212268590927124, 46.79753279685974]

speed_up = 3968.925694465637/np.array(run_times)
speed_up

plt.figure(figsize=(8, 5))
plt.plot(n_cores, speed_up)
# Add labels, title, and grid
plt.xlabel("Number of cores")
plt.ylabel("Speed up]")
plt.grid(True)
# Show the plot
plt.tight_layout()
plt.show()