
import time

# Calculate the start time
start = time.time()

# Code here


import os
n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
print(f"method 1 gives: {n_cores}")

import multiprocessing
m_cores = multiprocessing.cpu_count()
print(f"method 2 gives: {m_cores}")



# Calculate the end time and time taken
end = time.time()
length = end - start

length