import numpy as np
from scipy.special import comb, factorial
from time import time

fn = 'mnist'
global_d = 28 * 28 
K = 1
v_range = [0, 21]

partition = 0
m_range = [0, 785]

global_comb = dict()
global_powe = dict()

def my_comb(d, m):
	if (d, m) not in global_comb:
		global_comb[(d, m)] = comb(d, m, exact=True)

	return global_comb[(d, m)]

def my_powe(k, p):
	if (k, p) not in global_powe:
		global_powe[(k, p)] = k ** p

	return global_powe[(k, p)]

def get_count(d, m, n, v):
	if v == 0 and m == 0 and n == 0:
		return 1
	# early stopping
	if (v == 0 and m != n) or min(m, n) < 0 or max(m, n) > d or m + n < v:
		return 0

	if v == 0:
		return my_comb(d, m) * my_powe(K, m)
	else:
		c = 0
		# the number which are assigned to the (d-v) dimensions
		for i in range(max(0, n-v), min(m, d-v, int(np.floor((m+n-v) * 0.5))) + 1):
			if (m+n-v) / 2 < i:
				break
			x = m - i
			y = n - i
			j = x + y - v
			# the second one implies n <= m+v
			if j < 0 or x < j:
				continue
			tmp = my_powe(K-1, j) * my_comb(v, x-j) * my_comb(v-x+j, j)
			if tmp != 0:
				tmp *= my_comb(d-v, i) * my_powe(K, i)
				c += tmp

		return c


print('fn =', fn, 'Range of L0 norm =', v_range, 'partition =', partition, 'm_range =', m_range)
real_ttl = (K+1)**global_d
for v in range(v_range[0],v_range[1]):
	ttl = 0
	complete_cnt = []
	for m in range(m_range[0], m_range[1]):
		start = time()
		for n in range(m, min(m+v, global_d)+1):
			c = get_count(global_d, m, n, v)
			if c != 0:
				complete_cnt.append(((m, n), c))
				ttl += c
				# symmetric between d, m, n, v and d, n, m, v
				if n > m:
					ttl += c
		
		if m % 100 == 0:
			print('v = {}, m = {:10d}/{:10d}, ttl ratio = {:.4f}, # of dict = {}'.format(v, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
			print(fn, len(global_powe), len(global_comb), time() - start)
	
	np.save('list_counts/{}/complete_count_{}_{}'.format(fn, v, partition), complete_cnt)
	
	del complete_cnt 
	del global_comb, global_powe
	global_comb = dict()
	global_powe = dict()
