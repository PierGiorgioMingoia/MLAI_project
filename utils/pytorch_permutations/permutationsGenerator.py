import random
import numpy as np
import argparse


# Checks if every vector of lists has at least <target_distance> hamming distance from v
def is_min_hamming_distance(v, lists, target_distance):

    for vector in lists:
        distance = 0

        for i, element in enumerate(v):
            if v[i] != vector[i]:
                distance = distance + 1

        if distance < target_distance:
            return False

    return True


def large_hamming_distance_permutations(n, limit=30):

    result = []
    sorted_permutation = [i for i in range(n)]
    current_permutation = list(sorted_permutation)

    # try to find permutations with max distance from the already selected ones
    # if the found ones are not enough, repeat lowering the expected distance
    for current_expected_distance in range(n, 0, -1):
        attempts = 0
        max_attempts = 1000 * limit

        while attempts < max_attempts:
            attempts = attempts + 1
            random.shuffle(current_permutation)

            # compute distances between the current permutation and the ones already contained in the result:
            if current_permutation != sorted_permutation and current_permutation not in result and \
                    is_min_hamming_distance(current_permutation, result, current_expected_distance):

                result.append(list(current_permutation))

                if len(result) == limit:
                    print("Min hamming distance:", current_expected_distance)
                    return result


def factorial(n):

    if n == 0:
        return 1

    return n * factorial(n - 1)


parser = argparse.ArgumentParser()
parser.add_argument("-n", default=9, type=int, help="Length of the list to be permuted")
parser.add_argument("--n_permutations", default=30, type=int, help="Number of desired output permutations")
parser.add_argument("--output", default="permutations", help="Name or path where the output file will be saved")
args = parser.parse_args()

if args.n_permutations >= factorial(args.n):
    raise Exception("Error: max number of permutations of ", args.n, "elements is", factorial(args.n))

res = np.array(large_hamming_distance_permutations(args.n, args.n_permutations))

for perm in res:
    print(perm)

print(len(res))

np.save(args.output, res)
