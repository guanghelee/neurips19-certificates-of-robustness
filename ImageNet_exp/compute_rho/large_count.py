import numpy as np
from scipy.special import comb, factorial
from time import time

import argparse

parser = argparse.ArgumentParser(description='Get thresholds')
parser.add_argument('--p', type=int, required=True,
                    help='partition number')
parser.add_argument('--r', type=int, required=True,
                    help='computed radius')
args = parser.parse_args()


count = dict()
# imagenet
fn = 'imagenet'
global_d = 224 * 224 * 3 
K = 255
v_range = [args.r, args.r+1]
# v_range = [1, 2]
# v_range = [2, 3]
# v_range = [3, 4]
# v_range = [4, 5]
# v_range = [5, 6]
# v_range = [6, 7]
# v_range = [7, 8]
# v_range = [8, 9]
# v_range = [9, 10]
# v_range = [10, 11]
# v_range = [11, 12]
# v_range = [12, 13]
# v_range = [13, 14]
# v_range = [14, 15]
# v_range = [15, 16]

divider = 10
partition = args.p

if partition == 0:
	m_range = [0, (150529 // divider)]
elif partition == divider - 1:
	m_range = [(150529 // divider) * partition, 150529]
else:
	m_range = [(150529 // divider) * partition, (150529 // divider) * (partition + 1)]

# partition = 0
# m_range = [0, (150529 // divider) * 1]

# partition = 1
# m_range = [(150529 // 5) * 1, (150529 // 5) * 2]

# partition = 2
# m_range = [(150529 // 5) * 2, (150529 // 5) * 3]

# partition = 3
# m_range = [(150529 // 5) * 3, (150529 // 5) * 4]

# partition = 4
# m_range = [(150529 // 5) * 4, 150529]

# 150529 for imagenet

# fn = 'mnist'
# global_d = 28 * 28 
# K = 1
# v_range = [0, 785]
# # v_range = [0, 5]

# partition = 0
# m_range = [0, 785]

# partition = 1
# m_range = [785 // 2, 785]

# fn = 'cifar'
# global_d = 32 * 32 * 3
# K = 255
# v_range = [0, 200]

# partition = 0
# m_range = [0, 3073]

# 3073 for cifar10

global_comb = dict()
global_powe = dict()
# global_fact = dict()

def my_comb(d, m):
	# return comb(d, m, exact=True)
	if (d, m) not in global_comb:
		global_comb[(d, m)] = comb(d, m, exact=True)

	return global_comb[(d, m)]


def my_powe(k, p):
	# return k ** p
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
		return comb(d, m, exact=True) * (K ** m)
		# return my_comb(d, m) * my_powe(K, m)
	else:
		c = 0
		# the number which are assigned to the (d-v) dimensions
		# for i in range(0, min(m, d-v)+1):
		for i in range(max(0, n-v), min(m, d-v, int(np.floor((m+n-v) * 0.5))) + 1):
		# for i in range(0, m+1):
			if (m+n-v) / 2 < i:
				break
			x = m - i
			y = n - i
			j = x + y - v
			# the second one implies n <= m+v
			if j < 0 or x < j:
				continue
			tmp = my_powe(K-1, j) * my_comb(v, x-j) * my_comb(v-x+j, j)
			# tmp = my_powe(K-1, j) * my_fact(v) / my_fact(x - j) / my_fact(y - j) / my_fact(j)
			if tmp != 0:
				# tmp *= get_count(d-v, i, i, 0)
				tmp *= my_comb(d-v, i) * my_powe(K, i)
				c += tmp

		return c


print('fn =', fn, 'v_range =', v_range, 'partition =', partition, 'm_range =', m_range)
real_ttl = (K+1)**global_d
for v in range(v_range[0],v_range[1]):
# for v in [1]:
	ttl = 0
	# complete_cnt = np.zeros((global_d + 1, global_d + 1), dtype=object)
	# complete_cnt = dict()
	complete_cnt = []
	print('done for opening an array...')
	# for m in range(global_d + 1):
	for m in range(m_range[0], m_range[1]):
		# for n in range(m, global_d + 1):
		start = time()
		for n in range(m, min(m+v, global_d)+1):
		# for n in range(m, m+v+1):
			c = get_count(global_d, m, n, v)
			if c != 0:
				# complete_cnt[(m, n)] = c
				complete_cnt.append(((m, n), c))
				ttl += c
				# symmetric between d, m, n, v and d, n, m, v
				if n > m:
					ttl += c
		
		# if fn == 'imagenet' or m % 100 == 0:
		if m % 100 == 0:
			print('v = {}, m = {:10d}/{:10d}, ttl ratio = {:.4f}, # of dict = {}/{}'.format(v, m, m_range[1], ttl / real_ttl, len(count), len(complete_cnt)))
			print(fn, len(global_powe), len(global_comb), time() - start)
			# print(np.mean(hit_powe), np.mean(hit_comb))
	
	# np.save('counts/{}/cache_count'.format(fn), [count, v, global_d, K])
	np.save('list_counts/{}/complete_count_{}_{}'.format(fn, v, partition), complete_cnt)
	
	del complete_cnt 
	del global_comb, global_powe
	global_comb = dict()
	global_powe = dict()


# hit_comb = []
# hit_powe = []

# disable checking
# check = ttl - real_ttl
# print(v, '/', v_range[1], check, 'should be 0, # of dict =', len(count))
# if check != 0:
# 	exit(0) 

# if d == 0:
# 	if m == 0 and n == 0 and v == 0:
# 		return 1
# 	else:
# 		return 0

# def my_fact(n):
# 	if n not in global_fact:
# 		global_fact[n] = factorial(n, exact=True)
# 	return global_fact[n]
