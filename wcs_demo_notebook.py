import matplotlib.pyplot as plt
from wcs_helper_functions import *
import numpy as np
from scipy import stats
from random import random
import statistics as stat

munsellInfo = readChipData('./WCS_data_core/chip.txt')
namingData = readNamingData('./WCS_data_core/term.txt')
fociData = readFociData('./WCS_data_core/foci-exp.txt')
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')

diff_perc_dict = {}

for i in range(1, 111):

    curr_lang_naming = namingData[i]
    # print(curr_lang_naming)
    curr_lang_speaker = speakerInfo[i]
    if i == 97:
        print(curr_lang_speaker)
        print(len(curr_lang_naming))
    curr_male_index = []

    curr_female_index = []
    if (len(curr_lang_speaker) >= 10):

        # gender groups
        for j in range(1, len(curr_lang_speaker) + 1):
            if (speakerInfo[i][j][0][1] == 'M'):
                if (j < len(curr_lang_naming)):
                    curr_male_index.append(j)
            else:
                if (j < len(curr_lang_naming)):
                    curr_female_index.append(j)
        if i == 20:
            print(curr_male_index)
            print(curr_female_index)

        # mode for each chip
        curr_male_chip = []
        curr_female_chip = []
        k = 0
        for c in range(1, 331):
            # print(c)
            male_mode = stat.mode([curr_lang_naming[k][c] for k in curr_male_index])
            curr_male_chip.append(male_mode)
            female_mode = stat.mode([curr_lang_naming[k][c] for k in curr_female_index])
            curr_female_chip.append(female_mode)

        plt.subplot(5, 10, 4)
        # generate chip
        encoded_terms_male = map_array_to(curr_male_chip, generate_random_values(curr_male_chip))
        plotValues(encoded_terms_male, i, "male")
        encoded_terms_female = map_array_to(curr_female_chip, generate_random_values(curr_female_chip))
        plotValues(encoded_terms_female, i, "female")
        # count difference
        diff_count = 0
        for l in range(1, len(curr_male_chip)):

            if curr_male_chip[l] != curr_female_chip[l]:
                diff_count += 1
        # print(diff_count)
        diff_perc = diff_count / 330
        diff_perc_dict[i] = diff_perc

print(len(diff_perc_dict.values()))
print(len(diff_perc_dict.keys()))
