from wcs_helper_functions import *
from random import choices


def get_male_and_female_indices():
    """
    Get the speaker indices for a specific language based on gender.
    Returns
    -------
    Tuple(List[int], List[int]).
        - The first list contains indices for all male speakers.
        - The second list contains indices for all female speakers
    """
    male_ind = []
    female_ind = []
    for speaker_ind in age_gender_of_speaker_for_lang:
        if age_gender_of_speaker_for_lang[speaker_ind][0][1] == "M":
            male_ind.append(speaker_ind)
        elif age_gender_of_speaker_for_lang[speaker_ind][0][1] == "F":
            female_ind.append(speaker_ind)
    return male_ind, female_ind


def get_uniq_color_terms(male_ind, female_ind):
    """
    Parameters
    ----------
    male_ind: List of indices for all male speakers of a specific language
    female_ind: List of indices for all female speakers of a specific language

    Returns
    -------
    Tuple(List[str], List[str])
        - The first list contains unique color terms used by male speakers to describe all color chips
        - The second list contains unique color terms used by female speakers to describe all color chips
    """
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
    """
    Randomly selects (without replacement) indices from male and female speakers, number_of_samples times
    Returns
    -------
    Tuple(List[int], List[int])
        - The first list contains a sample of male speaker indices.
        - The second list contains a sample of female speaker indices.
        - The length for both lists equals number_of_samples
    """
    male_ind_sample = choices(male_indices, k=number_of_samples)
    female_ind_sample = choices(female_indices, k=number_of_samples)
    return male_ind_sample, female_ind_sample


def clean_age_gender_of_speaker_for_lang(unclean_age_gender_of_speaker_for_lang):
    """
    Data cleaning for the variable containing information about the speakers' age and gender.
    Parameters
    ----------
    unclean_age_gender_of_speaker_for_lang: Unclean data containing information about the speakers.

    Returns
    -------
    Data containing information about the speakers, excluding speakers with an age of 0.
    """
    cleaned_age_gender_of_speaker_for_lang = {}
    for speaker_index in unclean_age_gender_of_speaker_for_lang:
        if unclean_age_gender_of_speaker_for_lang[speaker_index][0][0] == "0":
            continue
        else:
            cleaned_age_gender_of_speaker_for_lang[speaker_index] = unclean_age_gender_of_speaker_for_lang[
                speaker_index]
    return cleaned_age_gender_of_speaker_for_lang


def run_trials(num_of_trials):
    """
    Runs num_of_trials trials to calculate the proportion of trials where:
        1. males used more unique color terms than females.
        2. females used more unique color terms than males.
        3. both genders used the same amount of unique color terms
    Parameters
    ----------
    num_of_trials: Number of trials to run

    Returns
    -------
    Tuple(int, int, int):
        1. Number of trials where males used more unique color terms than females.
        2. Number of trials where females used more unique color terms than males.
        3. Number of trials where both genders used the same amount of unique color terms
    """
    male_more_colorterms_than_female = 0
    female_more_colorterms_than_male = 0
    equal_colorterms = 0
    for trial in range(num_of_trials):
        male_indices_sample, female_indices_sample = sample_male_and_female_indices()
        uniq_male_color_term_names, uniq_female_color_term_names = get_uniq_color_terms(male_indices_sample,
                                                                                        female_indices_sample)
        if len(uniq_male_color_term_names) == len(uniq_female_color_term_names):
            equal_colorterms += 1
        elif len(uniq_male_color_term_names) > len(uniq_female_color_term_names):
            male_more_colorterms_than_female += 1
        else:
            female_more_colorterms_than_male += 1
    return male_more_colorterms_than_female, female_more_colorterms_than_male, equal_colorterms


def value_for_lang_index():
    """
    Returns
    -------
    "M" or "F" depending on which gender (male or female, respectively) used more unique color terms to describe all
    color chips of a specific language. Returns "E" if both genders used the same amount of unique color terms.
    """
    if female_more_color_terms_than_male > male_more_color_terms_than_female \
            and female_more_color_terms_than_male > equal_color_terms:
        return 'F'
    elif male_more_color_terms_than_female > female_more_color_terms_than_male \
            and male_more_color_terms_than_female > equal_color_terms:
        return "M"
    else:
        return "E"


# Language Index, Speaker Index, Color Chip Index, Color Chip Speaker Response
namingData = readNamingData('./WCS_data_core/term.txt')
# Language Index, Speaker Index, List[Tuple(Speaker Age, Speaker Gender)]
speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')

# Dictionary where key is the language index and the value is a string:
#   1. "M": The total unique list of color terms used by male speakers is more than female speakers
#   2. "F": The total unique list of color terms used by female speakers is more than male speakers
#   3. "E": The total unique list of color terms used is the same for both genders
lang_index_is_female_more = {}

# TODO: Debug code. Remove before submission.
lang_ind_group_by_num_of_color_terms = {}  # num_of_term: ind
lang_ind_and_winner_group_by_num_of_color_terms = {}  # num_of_term:(ind,winner)
list_of_most_occurrence_by_group = []
female_more_than_male = 0
male_more_than_female = 0
equal = 0

for language_index in range(1, 111):
    responses_for_lang = namingData[language_index]
    age_gender_of_speaker_for_lang = clean_age_gender_of_speaker_for_lang(speakerInfo[language_index])
    # Skip languages with fewer than 10 speakers
    if len(age_gender_of_speaker_for_lang) < 10:
        continue

    male_indices, female_indices = get_male_and_female_indices()
    number_of_samples = min(len(male_indices), len(female_indices))

    male_more_color_terms_than_female, female_more_color_terms_than_male, equal_color_terms = run_trials(100)
    lang_index_is_female_more[language_index] = value_for_lang_index()

    # TODO: Debug code. Remove before submission.
    if lang_index_is_female_more[language_index] == 'F':
        female_more_than_male += 1
    elif lang_index_is_female_more[language_index] == 'M':
        male_more_than_female += 1
    else:
        equal += 1
    print(language_index, lang_index_is_female_more[language_index])
    print("Female More: ", female_more_than_male)
    print("Male More: ", male_more_than_female)
    print("Equal: ", equal)

    # stratify into groups based on numbers of color terms START
    all_uniq_male_color_term_names, all_uniq_female_color_term_names = \
        get_uniq_color_terms(male_indices, female_indices)

    all_uniq_color_term_names = list(set(all_uniq_male_color_term_names+all_uniq_female_color_term_names))
    number_of_uniq_color_term_names = len(all_uniq_color_term_names)

    if number_of_uniq_color_term_names in lang_ind_group_by_num_of_color_terms.keys():
        lang_ind_group_by_num_of_color_terms[number_of_uniq_color_term_names].append(language_index)
    else:
        lang_ind_group_by_num_of_color_terms[number_of_uniq_color_term_names] = [language_index]
    # stratify into groups based on numbers of color terms END

    # organize the finegrain genders based on groups START
    ind_and_winner = (language_index, lang_index_is_female_more[language_index])
    if number_of_uniq_color_term_names in lang_ind_and_winner_group_by_num_of_color_terms.keys():
        lang_ind_and_winner_group_by_num_of_color_terms[number_of_uniq_color_term_names].append(ind_and_winner)
    else:
        lang_ind_and_winner_group_by_num_of_color_terms[number_of_uniq_color_term_names] = [ind_and_winner]
    # organize the finegrain genders based on groups END

print(lang_index_is_female_more)
