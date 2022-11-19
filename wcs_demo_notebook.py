import matplotlib.pyplot as plt
from wcs_helper_functions import *
import numpy as np
from scipy import stats
from random import random, choices
import statistics as stat

# Color chips, Letter and number, 330 color chips
munsellInfo = readChipData('./WCS_data_core/chip.txt')
namingData = readNamingData('./WCS_data_core/term.txt')
fociData = readFociData('./WCS_data_core/foci-exp.txt')
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')

diff_perc_dict = {}


# new hypo: there is gender difference in color naming
# 1) stratify the languages in WCS by roughly #color terms present in that language, such that you have groups of
# languages that
#     - have 3 basic color terms
#     - have 4 basic color terms
#     - have 5 basic color terms
#      - have …….
# Then, within each group where you know #color terms is held roughly constant,
# you compare things between M and F groups.
# You then report the results of gender difference in each of these groups,
# where #color terms is controlled for. If the hypothesis is held generally true,
# we should expect gender difference to be present in many groups
# (as opposed to only in those groups with higher #color terms).

def count_males_and_females():
    number_of_males = 0
    number_of_females = 0
    for speaker_ind in age_gender_of_speaker_for_lang:
        if age_gender_of_speaker_for_lang[speaker_ind][0][1] == "M":
            number_of_males += 1
        else:
            number_of_females += 1
    return number_of_males, number_of_females


def get_male_and_female_indices():
    male_ind = []
    female_ind = []
    for speaker_ind in age_gender_of_speaker_for_lang:
        if age_gender_of_speaker_for_lang[speaker_ind][0][1] == "M":
            male_ind.append(speaker_ind)
        else:
            female_ind.append(speaker_ind)
    return male_ind, female_ind


def get_uniq_color_terms(male_ind, female_ind):
    male_color_term_names = []
    female_color_term_names = []
    for speaker_index in male_ind:
        for speaker_responses in responses_for_lang[speaker_index].values():
            male_color_term_names.append(speaker_responses)
    for speaker_index in female_ind:
        for speaker_responses in responses_for_lang[speaker_index].values():
            female_color_term_names.append(speaker_responses)
    return list(set(male_color_term_names)), list(set(female_color_term_names))


def sample_male_and_female_indices():
    male_ind_sample = choices(male_indices, k=number_of_samples)
    female_ind_sample = choices(female_indices, k=number_of_samples)
    return male_ind_sample, female_ind_sample


def clean_age_gender_of_speaker_for_lang(unclean_age_gender_of_speaker_for_lang):
    cleaned_age_gender_of_speaker_for_lang = {}
    for speaker_index in unclean_age_gender_of_speaker_for_lang:
        if unclean_age_gender_of_speaker_for_lang[speaker_index][0][0] == "0":
            continue
        else:
            cleaned_age_gender_of_speaker_for_lang[speaker_index] = unclean_age_gender_of_speaker_for_lang[speaker_index]
    return cleaned_age_gender_of_speaker_for_lang


lang_index_is_female_more = {}
num_of_trues = 0
num_of_falses = 0

for language_index in range(1, 111):
    responses_for_lang = namingData[language_index]
    age_gender_of_speaker_for_lang = clean_age_gender_of_speaker_for_lang(speakerInfo[language_index])
    if len(age_gender_of_speaker_for_lang) < 10:
        continue

    num_of_males, num_of_females = count_males_and_females()
    male_indices, female_indices = get_male_and_female_indices()
    number_of_samples = min(num_of_males, num_of_females)

    male_more_color_terms_than_female = 0
    female_more_color_terms_than_male = 0
    for trial in range(1000):
        male_indices_sample, female_indices_sample = sample_male_and_female_indices()
        uniq_male_color_term_names, uniq_female_color_term_names = get_uniq_color_terms(male_indices_sample,
                                                                                        female_indices_sample)
        if len(uniq_male_color_term_names) > len(uniq_female_color_term_names):
            male_more_color_terms_than_female += 1
        else:
            female_more_color_terms_than_male += 1
    lang_index_is_female_more[language_index] = (female_more_color_terms_than_male > male_more_color_terms_than_female)

    if lang_index_is_female_more[language_index]:
        num_of_trues += 1
    else:
        num_of_falses += 1
    print(language_index, lang_index_is_female_more[language_index])
    print("Trues: ", num_of_trues)
    print("Falses: ", num_of_falses)
    # Skip languages with fewer than 10 speakers

    # if language_index == 20:
    #     print(male_color_term_names)
    #     print(female_color_term_names)

    # plt.subplot(5, 10, 4)
    # # generate chip
    # encoded_terms_male = map_array_to(curr_male_chip, generate_random_values(curr_male_chip))
    # plotValues(encoded_terms_male, language_index, "male")
    # encoded_terms_female = map_array_to(curr_female_chip, generate_random_values(curr_female_chip))
    # plotValues(encoded_terms_female, language_index, "female")
    # # count difference
    # diff_count = 0
    # for l in range(1, len(curr_male_chip)):
    #
    #     if curr_male_chip[l] != curr_female_chip[l]:
    #         diff_count += 1
    # # print(diff_count)
    # diff_perc = diff_count / 330
    # diff_perc_dict[language_index] = diff_perc

# print(len(diff_perc_dict.values()))
# print(len(diff_perc_dict.keys()))
print(lang_index_is_female_more)
