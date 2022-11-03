import random
import numpy as np
import torch
import Levenshtein

def set_random_seed(seed_num=1):
	random.seed(seed_num)
	np.random.seed(seed_num)
	torch.manual_seed(seed_num)
	torch.cuda.manual_seed(seed_num)
	torch.cuda.manual_seed_all(seed_num)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

### Levenshtein
def calculate_levenshtein(h, y, lh, ly, decoder, labels, debug = False):

    if debug:
        print(f"\n----- IN LEVENSHTEIN -----\n")
        # Add any other debug statements as you may need
        # you may want to use debug in several places in this function
    
    # TODO: look at docs for CTC.decoder and find out what is returned here
    '''
        The decoder.decode function return the following:
        1. beam_results -> results from a given beam search
        2. beam_scores -> CTC score of each beam [model's confidence that the beam is correct: p=1/np.exp(beam_score)]
        3. timesteps -> timestep at which the nth output character has peak probability
        4. out_lens -> has the results for each beam search for each item in the batch

        Refer the shape from the documentation

    '''
    beam_results, beam_scores, timesteps, out_lens= decoder.decode(h, seq_lens=lh)

    # It is not config['BATCH_SIZE'] (Try to think of an edge case)
    batch_size = 0 # TODO
    distance = 0 # Initialize the distance to be 0 initially

    for i in range(batch_size): 
        # TODO: Loop through each element in the batch
        # Find the beam results for the first beam, for each element in the batch 
        # Reverse encode the results using the dictionary we created above

        # Do the same for the input

        # What is the point of lh and ly, why do we need these lengths

        # Calculate levenshtein distance using Levenshtein.distance()
        pass

    distance /= batch_size # TODO: Uncomment this, but think about why we are doing this

    return distance

