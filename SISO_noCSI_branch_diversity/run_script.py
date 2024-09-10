import deepSISOnoCSI_bd
import time
start_time = time.time()

k = 1
n = 2
deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'gamma')

with open(f'report_{k}.txt', mode='w') as file_id:
    file_id.write(f'Execution time = {(time.time() - start_time) / 3600} hrs.')
