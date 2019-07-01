import numpy as np
from tqdm import trange
import argparse

parser = argparse.ArgumentParser(description='Get thresholds')
parser.add_argument('--r_start', default=0, type=int, 
                    help='[r_start, r_end)')
parser.add_argument('--r_end', default=21, type=int, 
                    help='[r_start, r_end)')
parser.add_argument('--a', type=int, default=80,
                    help='alpha = a / 100')
args = parser.parse_args()

v_range = [args.r_start, args.r_end]

fn = 'mnist'
K = 1
global_d = 28 * 28 
p_range = [0, 1]

a = args.a
b = 100

frac_alpha = a / b
frac_beta = (1 - frac_alpha) / K
print('alpha = {}, beta = {}'.format(frac_alpha, frac_beta))
print('v =', v_range)

Z = K * 100
hZ = K * 50
tZ = K * 10

alpha = a * K
beta = b - a
print('alpha = {}, beta = {}'.format(alpha, beta))
print(Z, alpha + K * beta)

if fn == 'mnist':
	fp = open('thresholds/{}/{}.txt'.format(fn, frac_alpha), 'w')
	fp.write('r\trho_r^{-1}(0.5)\terr\n')

for v in range(v_range[0], v_range[1]):
	alpha = a * K
	beta = b - a

	search_exponent = 20
	tenth_Z = (tZ ** search_exponent) * (Z ** (global_d - search_exponent))
	half_Z = hZ * (Z ** (global_d - 1))
	# debug!!!!!!!!!!
	# half_Z = tenth_Z

	complete_cnt = []
	for p in range(p_range[0], p_range[1]):
		cnt = np.load('list_counts/{}/complete_count_{}_{}.npy'.format(fn, v, p))
		complete_cnt += list(cnt)
		print('reading', p, 'done')
	raw_cnt = 0
	
	outcome = []
	for ((m, n), c) in complete_cnt:
		outcome.append((
			# likelihood ratio x flips m, x bar flips n
			# and then count, m, n
			(alpha ** (n - m)) * (beta ** (m - n)), c, m, n
		))
		if m != n:	
			outcome.append((
				(alpha ** (m - n)) * (beta ** (n - m)), c, n, m
			))

		raw_cnt += c
		if m != n:
			raw_cnt += c

	print(v, raw_cnt - (K+1) ** global_d)
	# continue
	outcome = sorted(outcome, key = lambda x: -x[0])

	# the given probability
	p_given = 0
	# worst case, check when it is equal to 0.5
	q_worst = 0

	for i in trange(len(outcome)):
		ratio, cnt, m, n = outcome[i]
		p = (alpha ** (global_d - m)) * (beta ** m)
		q = (alpha ** (global_d - n)) * (beta ** n)
		q_delta = q * cnt

		# if fn == 'imagenet' and i % 20 == 0:
		# 	print(i, '/', len(outcome))

		# if q_worst + q_delta < half_Z:
			# q_worst += q_delta
		if q_delta < half_Z:
			half_Z -= q_delta

			p_delta = p * cnt
			p_given += p_delta
		else:
			# return the smallest q_bar >= 0.5
			print('enter')
			upper = cnt
			lower = 0
			while upper - lower != 1:
				mid = (lower + upper) >> 1
				# upper should be always >= half_Z
				# lower should be always < half_Z
				running_Z = q_worst + mid * q
				if running_Z < half_Z:
					lower = mid
				else:
					upper = mid

			q_worst += upper * q
			p_given += upper * p
			print('found')
			print(upper - lower)
			# print(q_worst - half_Z)
			# print(q_worst - q - half_Z)
			# exit(0)
			break

	# print('error is bounded by', p)
	################ numerical error = alpha ** d + 10 ** (-search_exponent) ################

	# should output upper bound
	upper = 10 ** search_exponent
	lower = 0

	# goal: find the smallest p >= p_given 
	# the unnormalized p_given = p_given x Z = p_given x (10 ** 10) x (Z / (10 **10) ) <= p * (10 ** 10) x (Z / (10 **10) )
	# here we are actually searching all possible integer value of p * (10 ** 10).
	# after we found it, we have to divide it by 10 ** 10
	while upper - lower != 1:
		mid = (lower + upper) >> 1
		running_p = mid * tenth_Z
		if running_p < p_given:
			lower = mid
		else:
			upper = mid

	if upper != 10 ** search_exponent:
		# it's ok to do it since we know threshold >= 0.5
		p_thre = '0.' + str(upper)[:20]
	else:
		p_thre = '1.0'

	if fn == 'mnist':
		# don't search it for imagenet to save time
		upper = 10 ** search_exponent
		lower = 0

		# goal: find the smallest p >= p_given 
		# the unnormalized p_given = p_given x Z = p_given x (10 ** 10) x (Z / (10 **10) ) <= p * (10 ** 10) x (Z / (10 **10) )
		# here we are actually searching all possible integer value of p * (10 ** 10).
		# after we found it, we have to divide it by 10 ** 10
		while upper - lower != 1:
			mid = (lower + upper) >> 1
			running_p = mid * tenth_Z
			if running_p < p:
				lower = mid
			else:
				upper = mid

		if upper != 10 ** search_exponent:
			# it's ok to do it since we know threshold >= 0.5
			error = str(upper) + '* 10^-{}'.format(search_exponent)
	else:
		error = 'NA'


	# print('alpha = {}, v = {}, p_given should be > {}'.format(alpha, v, p_given))
	# print('v = {}, p_given > {}'.format(v, upper / (10 ** search_exponent)))
	# fp.write('{}\t{}\n'.format(v, upper / (10 ** search_exponent)))
	print('v = {}, p_given > {}, error bdd by {}'.format(v, p_thre, error))
	fp.write('{}\t{}\t{}\n'.format(v, p_thre, error))

	if p_thre == '1.0':
		break

	if fn != 'mnist':
		fp.close()

# 	alpha = a / b
# 	# beta also = (b - a) / (b * K)
# 	beta = (1 - alpha) / K

# # ========================

# # fp = open('thresholds/{}/{}.txt'.format(fn, alpha), 'w')
# # fp.write('v\tp_given\n')

# # for v in range(v_range[0], v_range[1]):
# 	complete_cnt = []
# 	for p in range(p_range[0], p_range[1]):
# 		cnt = np.load('list_counts/{}/complete_count_{}_{}.npy'.format(fn, v, p))
# 		complete_cnt += list(cnt)
# 	raw_cnt = 0
	
# 	outcome = []
# 	for ((m, n), c) in complete_cnt:
# 		outcome.append((
# 			# likelihood ratio x flips m, x bar flips n
# 			# and then count, m, n
# 			(alpha ** (n - m)) * (beta ** (m - n)), c, m, n
# 		))
# 		if m != n:	
# 			outcome.append((
# 				outcome[-1][0] ** (-1), c, n, m
# 			))

# 		raw_cnt += c
# 		if m != n:
# 			raw_cnt += c

# 	# print(v, raw_cnt - (K+1) ** global_d)
	
# 	outcome = sorted(outcome, key = lambda x: -x[0])

# 	# the given probability
# 	p_given = 0.
# 	# worst case, check when it is equal to 0.5
# 	q_worst = 0.

# 	for i in range(len(outcome)):
# 		ratio, cnt, m, n = outcome[i]
# 		p = (alpha ** (global_d - m)) * (beta ** m)
# 		q = (alpha ** (global_d - n)) * (beta ** n)
# 		q_delta = q * cnt


# 		# q_delta = np.exp(np.log(q) + np.log(cnt))
# 		if q_worst + q_delta < 0.5:
# 			q_worst += q_delta
# 			p_delta = p * cnt

# 			p_given += p_delta
# 		else:
# 			frac = (0.5 - q_worst) / q
# 			p_given += p * frac
# 			# print('found')
# 			break

# 	print('alpha = {}, v = {}, p_given should be > {}'.format(alpha, v, p_given))
# # # 	fp.write('{}\t{}\n'.format(v, p_given))
