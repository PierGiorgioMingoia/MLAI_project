import argparse
import numpy as np
import itertools
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Number of possible permutations')
parser.add_argument('--classes', default=30, type=int, help='Number of permutation to select, not considering the correct one')

args = parser.parse_args()

if __name__ == "__main__":
    outname = '/content/MLAI_project/utils/pytorch_permutations/permutations/permutations_hamming_%d' % (args.classes)

    # Create all possible permutations
    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))
    print(P_hat.size)
    n = P_hat.shape[0]

    for i in range(args.classes+1):
        if i == 0:
            j = 0
            P = np.array(P_hat[j]).reshape([1,-1])
        else:
            # Concatenate P with the max distance permutation
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        # Remove from all permutation the selected permutation
        P_hat = np.delete(P_hat, j, axis=0)
        # Compute distance between all of the permutation remaining
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()
        
        #j = D.argmax()

        m = int(D.shape[0] / 2)
        S = D.argsort()
        j = S[np.random.randint(m - 10, m + 10)]

    P = np.delete(P, 0, axis = 0)
    np.save(outname, P)
    print('file created -->' + outname)
