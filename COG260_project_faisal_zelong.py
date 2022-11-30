import numpy as np
import random
from scipy import stats
import statistics
from wcs_helper_functions import readFociData, readSpeakerData, readNamingData


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


def get_number_of_color_term_used(male_ind, female_ind):
    """
    Parameters
    ----------
    male_ind: List of indices for all male speakers of a specific language
    female_ind: List of indices for all female speakers of a specific language

    Returns
    -------
    Tuple(List[str], List[str])
        - The first list contains number of color terms used by each male speakers to describe all color chips
        - The second list contains number of color terms used by each female speakers to describe all color chips
    """
    number_of_color_term_each_male_used = []
    number_of_color_term_each_female_used = []
    for speaker_index in male_ind:
        terms_used_by_speaker_at_index = len(list(set(responses_for_lang[speaker_index].values())))

        if '*' in list(set(responses_for_lang[speaker_index].values())):
            terms_used_by_speaker_at_index -= 1

        number_of_color_term_each_male_used.append(terms_used_by_speaker_at_index)
    for speaker_index in female_ind:
        terms_used_by_speaker_at_index = len(list(set(responses_for_lang[speaker_index].values())))

        if '*' in list(set(responses_for_lang[speaker_index].values())):
            terms_used_by_speaker_at_index -= 1

        number_of_color_term_each_female_used.append(terms_used_by_speaker_at_index)
    return number_of_color_term_each_male_used, number_of_color_term_each_female_used


def sample_male_and_female_indices(number_of_samples):
    """
    Randomly selects (without replacement) indices from male and female speakers, number_of_samples times
    Returns
    -------
    Tuple(List[int], List[int])
        - The first list contains a sample of male speaker indices.
        - The second list contains a sample of female speaker indices.
        - The length for both lists equals number_of_samples
    """
    male_ind_sample = random.sample(male_indices, k=number_of_samples)
    female_ind_sample = random.sample(female_indices, k=number_of_samples)
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
    number_of_samples = min(len(male_indices), len(female_indices))
    male_more_colorterms_than_female = 0
    female_more_colorterms_than_male = 0
    equal_colorterms = 0
    for trial in range(num_of_trials):
        male_indices_sample, female_indices_sample = sample_male_and_female_indices(number_of_samples)
        uniq_male_color_term_each_used, uniq_female_color_term_each_used = \
            get_number_of_color_term_used(male_indices_sample, female_indices_sample)

        mean_uniq_color_terms_male = statistics.mean(uniq_male_color_term_each_used)
        mean_uniq_color_term_female = statistics.mean(uniq_female_color_term_each_used)

        # TODO: Remove permutation test for each trial
        # p_val_male, p_val_female = permutation(uniq_female_color_term_each_used, uniq_female_color_term_each_used)
        # if p_val_male > 0.05:
        #     permut_trial[0] = 'T'
        # if p_val_female > 0.05:
        #     permut_trial[1] = 'T'

        if mean_uniq_color_terms_male == mean_uniq_color_term_female:
            equal_colorterms += 1
        elif mean_uniq_color_terms_male > mean_uniq_color_term_female:
            male_more_colorterms_than_female += 1
        else:
            female_more_colorterms_than_male += 1
    return male_more_colorterms_than_female, female_more_colorterms_than_male, equal_colorterms


def t_test(alpha=0.05):
    """
    Runs num_of_trials trials to calculate the proportion of trials where:
        1. Mean of male used color terms significantly greater than mean of female used color terms
        2. Mean of male used color terms significantly lesser than mean of female used color terms
        3. Mean of male used color terms no significant different from mean of female used color terms
    at significance level 95% = alpha = 0.05
    Parameters
    ----------
    alpha: significance level

    Returns
    -------
    Tuple(int, int, int):
        1. Mean of male used color terms significantly greater than mean of female used color terms
        2. Mean of male used color terms significantly lesser than mean of female used color terms
        3. Mean of male used color terms no significant different from mean of female used color terms
    """

    # it looks  like there is no need use equal number between 2 genders because we are using the mean here.
    male_indices_sample, female_indices_sample = get_male_and_female_indices()
    male_term_len_list, female_term_len_list = get_number_of_color_term_used(male_indices_sample,
                                                                             female_indices_sample)
    # H_null: male >= female
    # H_alt: male < female
    t_test_two_sided = stats.ttest_ind(male_term_len_list, female_term_len_list,
                                       alternative='two-sided')

    p_val = t_test_two_sided[1]

    if p_val < alpha:  # significant diff
        # is male mean sig greater than female mean
        t_test_greater = stats.ttest_ind(male_term_len_list, female_term_len_list,
                                         alternative='greater')
        # is male mean sig less than female mean
        t_test_less = stats.ttest_ind(male_term_len_list, female_term_len_list,
                                      alternative='less')
        if t_test_greater[1] < alpha:
            return "M"
        if t_test_less[1] < alpha:
            return "F"
        print("error at t test")
        return "error"
    else:  # no significant diff
        return "E"


def choose_m_f_e_for_lang_index():
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


def sort_by_values_len(dct):
    """
    Convert dictionary values from lists to lengths of lists and sort dictionary based on its values in increasing order
    Parameters
    ----------
    dct: dict[any, List[any]]

    Returns
    -------
    A dictionary of key, int sorted in increasing order by its values
    """
    dict_len = {key: len(value) for key, value in dct.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = [{item[0]: dct[item[0]]} for item in sorted_key_list]
    return sorted_dict


def get_most_occurrence_element_keep_tie(lst):
    """
    Given a list, find the most occurring element(s)
    Parameters
    ----------
    lst: A list of elements

    Returns
    -------
    A tuple of the most occurring element(s) and the number of occurrences for such elements
    """
    uniq_lst = set(lst)
    max_elements = []
    max_occurrence = 0
    for element in uniq_lst:
        num_of_occurrence = lst.count(element)

        if max_occurrence < num_of_occurrence:
            max_occurrence = num_of_occurrence
            max_elements = [element]
        elif max_occurrence == num_of_occurrence:
            max_elements.append(element)

    return max_elements, max_occurrence


def get_dict_to_list_by_fine_grained_gender_most_occurrence(list_of_sorted_lang_ind_and_winner_dicts, count_index):
    list_of_most_occurrence_by_group = []  # [how many term this group of languages use,
    #                                         how many member has the most occurring str,
    #                                         the most occurring str of this group,
    #                                         how many times this str has occurred,
    #                                         percentage (this occur/total member)]

    for a_group_of_languages_with_same_number_of_color_terms in list_of_sorted_lang_ind_and_winner_dicts:
        # the current key, which is also the number of color terms that this group has
        key = list(a_group_of_languages_with_same_number_of_color_terms)[0]
        fine_grained_gender_of_each_language_member = []
        # go through members of this group, get the gender that has used more terms
        for language_member in a_group_of_languages_with_same_number_of_color_terms[key]:
            fine_grained_gender_of_each_language_member.append(language_member[count_index])  # 1: unique 2: t test mean

        # most_occurrence_of_this_group = max(fine_grained_gender_of_each_language_member,
        #                                     key=fine_grained_gender_of_each_language_member.count)

        most_occurrence_elements, most_occurrence_count = \
            get_most_occurrence_element_keep_tie(fine_grained_gender_of_each_language_member)

        occurrence_percentage = most_occurrence_count / len(a_group_of_languages_with_same_number_of_color_terms[key])

        list_of_most_occurrence_by_group.append([key, len(a_group_of_languages_with_same_number_of_color_terms[key]),
                                                 most_occurrence_elements,
                                                 most_occurrence_count,
                                                 occurrence_percentage])
    return list_of_most_occurrence_by_group


def count_threshold(permu_lst, mean_threshold, operation_str):
    """
    Helper function for permutation test
    Parameters
    ----------
    permu_lst
    mean_threshold
    operation_str

    Returns
    -------

    """
    count = 0
    if operation_str == 'less':
        for permu_value in permu_lst:
            if permu_value < mean_threshold:
                count += 1
    if operation_str == 'equal':
        for permu_value in permu_lst:
            if permu_value == mean_threshold:
                count += 1
    return count


def permutation(num_of_terms_ech_male_used, num_of_terms_ech_female_used):
    """
    Runs a permutation test on the number of unique terms used by males and females
    Parameters
    ----------
    num_of_terms_ech_male_used
    num_of_terms_ech_female_used

    Returns
    -------

    """
    # permutation test START
    permu_expect_array_male = []
    permu_expect_array_female = []
    all_terms_used = num_of_terms_ech_male_used + num_of_terms_ech_female_used

    mean_male = statistics.mean(num_of_terms_ech_male_used)
    mean_female = statistics.mean(num_of_terms_ech_female_used)

    for i in range(0, 1000):
        all_terms_used_permut = np.random.permutation(all_terms_used)

        num_of_terms_each_male_used_permut = all_terms_used_permut[:len(num_of_terms_ech_male_used)]
        num_of_terms_each_female_used_permut = all_terms_used_permut[-len(num_of_terms_ech_female_used):]

        mean_male_permut = statistics.mean(num_of_terms_each_male_used_permut)
        mean_female_permut = statistics.mean(num_of_terms_each_female_used_permut)

        permu_expect_array_male.append(mean_male_permut)
        permu_expect_array_female.append(mean_female_permut)

    count_no_different_male = count_threshold(permu_expect_array_male, mean_male, 'equal')
    permutation_p_value_male = count_no_different_male / 1000

    count_no_different_female = count_threshold(permu_expect_array_female, mean_female, 'equal')
    permutation_p_value_female = count_no_different_female / 1000

    return permutation_p_value_male, permutation_p_value_female


def get_num_of_basic_color_term(language_idx: int):
    """
    Returns the approximate number of basic color terms for a specific language
    Parameters
    ----------
    language_idx: The language index for the language we want to find the number of basic color terms

    Returns
    -------
    The average of all numbers of basic color terms across all speakers of the language
    """
    num_of_basic_term_for_each_speaker = []
    for speaker_index in fociData[language_idx]:
        num_of_basic_term_for_each_speaker.append(len(fociData[language_idx][speaker_index]))
    return statistics.mean(num_of_basic_term_for_each_speaker)


def data_cleaning():
    """
    Performs data cleaning and decides whether to skip the language based on the number of speakers it has
    Returns
    -------
    Either -1 or the cleaned dictionary containing the age and gender of the all speakers for the language
    """
    age_gender_of_speaker_for_lang_temp = clean_age_gender_of_speaker_for_lang(speakerInfo[language_index])
    # Skip languages with fewer than 10 speakers
    if len(age_gender_of_speaker_for_lang_temp) < 10:
        return -1
    else:
        return age_gender_of_speaker_for_lang_temp


if __name__ == "__main__":
    # Language Index, Speaker Index, Color Chip Index, Color Chip Speaker Response
    namingData = readNamingData('./WCS_data_core/term.txt')
    # Language Index, Speaker Index, List[Tuple(Speaker Age, Speaker Gender)]
    speakerInfo = readSpeakerData('./WCS_data_core/spkr-lsas.txt')
    # fociData[1][1]:{'A:0','B:1'} language-speaker-colorterm-foci-coord
    fociData = readFociData('./WCS_data_core/foci-exp.txt')

    # Dictionary where key is the language index and the value is a string:
    #   1. "M": The total unique list of color terms used by male speakers is more than female speakers
    #   2. "F": The total unique list of color terms used by female speakers is more than male speakers
    #   3. "E": The total unique list of color terms used is the same for both genders
    lang_index_is_female_more = {}
    lang_index_is_female_more_t_test = {}

    lang_ind_group_by_num_of_color_terms = {}  # num_of_term: ind
    lang_ind_and_winner_group_by_num_of_color_terms = {}  # num_of_term:(ind,winner)

    lang_grouping = {}
    lang_grouping_sorted = {}
    lang_grouping_unique_most_occur = {}
    lang_grouping_t_test_mean_most_occur = {}

    # TODO: Debug code. Remove before submission.
    female_more_than_male = 0
    male_more_than_female = 0
    equal = 0

    for language_index in range(1, 111):

        # 1. Get speaker data for language
        responses_for_lang = namingData[language_index]

        # 2. Data cleaning
        result = data_cleaning()
        if result == -1:
            continue
        else:
            age_gender_of_speaker_for_lang = result

        # 3. Split into gender groups
        male_indices, female_indices = get_male_and_female_indices()

        # 4. Run trials and count how many "F", "M", and "E" we get from all trials
        male_more_color_terms_than_female, female_more_color_terms_than_male, equal_color_terms = run_trials(1000)

        # 5. Assign the language "F", "M", or "E" based on which letter has the highest count
        lang_index_is_female_more[language_index] = choose_m_f_e_for_lang_index()

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

        # 6. Run permutation test on number of unique terms used by all male and female speakers
        num_of_terms_each_male_used, num_of_terms_each_female_used = \
            get_number_of_color_term_used(male_indices, female_indices)
        permut_trial = ['F', 'F']
        permutation_p_val_male, permutation_p_val_female = \
            permutation(num_of_terms_each_male_used, num_of_terms_each_female_used)
        if permutation_p_val_male > 0.05:
            permut_trial[0] = 'T'
        if permutation_p_val_female > 0.05:
            permut_trial[1] = 'T'

        # 7. Run T-test
        t_str = t_test()  # "E", "M", "F", "error"
        lang_index_is_female_more_t_test[language_index] = t_str

        # 8. Calculate the number of basic color terms for this language
        mean_of_basic_color_term_for_lang = round(get_num_of_basic_color_term(language_index))

        # 8.1 Organize all results for this language into a dictionary
        language_index_and_result = (language_index,
                                     lang_index_is_female_more[language_index],
                                     lang_index_is_female_more_t_test[language_index],
                                     permutation_p_val_male, permutation_p_val_female,
                                     permut_trial)

        # 9. Group the language based on the number of color terms
        if mean_of_basic_color_term_for_lang in lang_grouping:
            lang_grouping[mean_of_basic_color_term_for_lang].append(language_index_and_result)
        else:
            lang_grouping[mean_of_basic_color_term_for_lang] = [language_index_and_result]

    # 10 Sort the groups in increasing order based on how many languages they contain
    lang_grouping_sorted = sort_by_values_len(lang_grouping)

    # 11. Get the most re-occurring key ('M','F','E') of each group and organize other results for display
    lang_grouping_unique_most_occur = \
        get_dict_to_list_by_fine_grained_gender_most_occurrence(lang_grouping_sorted, 1)
    lang_grouping_t_test_mean_most_occur = \
        get_dict_to_list_by_fine_grained_gender_most_occurrence(lang_grouping_sorted, 2)
