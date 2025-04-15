import numpy as np
import time
import reconPhaseTieTvWeighted


def recon_phase_tie_tv_weighted_batch(data, LAMBDA, VERBOSE):
    """
    Wrapper function for batch simulations
    """
    # allocate output
    PHASE = [None] * len(LAMBDA)

    # get structure fields
    b1 = data['bNear']
    b2 = data['bFar']
    OTF1 = data['tieOtfNear']
    OTF2 = data['tieOtfFar']
    W1 = data['W_high']
    W2 = data['W_low']
    reg = data['sval']
    T = data['T']
    tol = data['tol']
    x0=None

    start_time = time.time()
    for ii, lambda_ in enumerate(LAMBDA):
        # Reconstruction
        phase = reconPhaseTieTvWeighted.reconPhaseTieTvWeighted(b1, b2, OTF1, OTF2, W1, W2, lambda_, x0,T, lambda_*10, VERBOSE, reg, tol)
        PHASE[ii] = phase

    print(time.time() - start_time)

    return PHASE
