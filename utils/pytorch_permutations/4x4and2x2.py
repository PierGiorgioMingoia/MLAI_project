import numpy as np
from scipy.spatial.distance import cdist
import argparse



parser = argparse.ArgumentParser(description='Number of possible permutations')
parser.add_argument('--classes', default=10, type=int, help='Number of permutations to select, not considering the correct one')

args = parser.parse_args()

# We can generate a full random set and then computing the hamming distance from that set


if __name__ == "__main__":

    outname2x2 = './test/2x2_permutation_%d' % (args.classes)
    outname4x4 = './test/4x4_permutation_%d' % (args.classes)

    permu2x2 = range(4)
    permu4x4 = range(16)

    P2x2 = np.array([permu2x2] , dtype=int)
    P4x4 = np.array([permu4x4], dtype=int)

    for i in range(args.classes+1):
        tmp2x2 = np.random.permutation(permu2x2)
        tmp4x4 = np.random.permutation(permu4x4)
        P2x2 = np.append(P2x2,[tmp2x2], axis=0)
        P4x4 = np.append(P4x4,[tmp4x4], axis=0)

    print(P2x2)
    print(P4x4)
    np.save(outname2x2, P2x2)
    np.save(outname4x4, P4x4)

    
