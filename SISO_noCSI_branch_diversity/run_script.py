import deepSISOnoCSI_bd
import time
start_time = time.time()

k = 1
n = 2
# deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'rayleigh')
# deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'gamma')
# deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'gumbel')
# deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'custom')
deepSISOnoCSI_bd.branch_diversity_routine(k, n, 10, 'foldedNormal')
