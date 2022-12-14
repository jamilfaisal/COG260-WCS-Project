import numpy as np
import random

from matplotlib import pyplot as plt
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

        # Removed permutation test for each trial because it is too time-consuming to run
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
    # H_null: male == female
    # H_alt: male != female
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
    """
    count = 0
    if operation_str == '<':
        for permu_value in permu_lst:
            if permu_value < mean_threshold:
                count += 1
    if operation_str == '==':
        for permu_value in permu_lst:
            if permu_value == mean_threshold:
                count += 1
    if operation_str == '>=':
        # null mu_female >= mu_male
        for permu_value in permu_lst:
            if permu_value >= mean_threshold:
                count += 1
    return count


def calculate_permutation_p_values(num_of_terms_ech_male_used, num_of_terms_ech_female_used):
    """
    Runs a permutation test on the number of unique terms used by males and females
    """
    # permutation test START
    permu_expect_array_male = []
    permu_expect_array_female = []
    permu_expect_array_diff = []
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
        permu_expect_array_diff.append(mean_male_permut - mean_female_permut)

    count_no_different_male = count_threshold(permu_expect_array_male, mean_male, '==')
    permutation_p_value_male = count_no_different_male / 1000

    count_no_different_female = count_threshold(permu_expect_array_female, mean_female, '==')
    permutation_p_value_female = count_no_different_female / 1000
    count_no_different_diff = count_threshold(permu_expect_array_diff, mean_male - mean_female, '>=')
    permutation_p_value_diff = count_no_different_diff / 1000

    return permutation_p_value_male, permutation_p_value_female, permutation_p_value_diff


def lst_to_dict_by_count_element_occurence(lst):
    dict_counted = {}
    for element in lst:
        if element in dict_counted:
            dict_counted[element] += 1
        else:
            dict_counted[element] = 1

    return dict_counted


def run_permutation_test():
    # null: mean_m == mean_fem
    # alt: mean_fem != fem

    final_permutation_result = ['alt', 'alt', 'alt']
    num_of_terms_each_male_used, num_of_terms_each_female_used = \
        get_number_of_color_term_used(male_indices, female_indices)
    permut_pval_male, permutat_pval_female, permut_p_val_diff = \
        calculate_permutation_p_values(num_of_terms_each_male_used, num_of_terms_each_female_used)
    if permut_pval_male > 0.05:
        final_permutation_result[0] = 'null'
    if permutat_pval_female > 0.05:
        final_permutation_result[1] = 'null'
    if permut_p_val_diff > 0.05:
        final_permutation_result[2] = 'null'
    return permut_pval_male, permutat_pval_female, permut_p_val_diff, final_permutation_result


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
    speaker_count = 0
    all_speaker_response = {}
    bct_response = []

    for speaker_index in fociData[language_idx]:
        num_of_basic_term_for_each_speaker.append(len(fociData[language_idx][speaker_index]))

    for speaker_index in fociData[language_idx]:
        speaker_count += 1
        unique_speaker_response = fociData[language_idx][speaker_index].keys()
        for response in unique_speaker_response:
            if response in all_speaker_response.keys():
                all_speaker_response[response] += 1
            else:
                all_speaker_response[response] = 1
    for resp in all_speaker_response:
        if all_speaker_response[resp] != '*':
            if all_speaker_response[resp] == speaker_count:
                bct_response.append(resp)

    return max(set(num_of_basic_term_for_each_speaker), key=num_of_basic_term_for_each_speaker.count)


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


def plot_pie_charts_for_lang_groups(language_groups):
    """
    Plots and saves one pie chart for each language group. The pie chart shows the percentage of languages based on
     whether:
        1. Males used more unique color terms than females
        2. Females used more unique color terms than males
        3. Males and Females used the same number of unique color terms
    Parameters
    ----------
    language_groups: Dict
        key: Number of basic color terms
        Value: List of all languages that have key as the number of basic color terms. Each element of the list
            represents information about which gender group used more unique color terms for one language.

    Returns
    -------

    """
    for num_of_b_color_term, val in language_groups.items():
        all_responses_in_letters = [x[1] for x in val]
        mylabels = []
        all_responses_counted = []
        if all_responses_in_letters.count("F") >= 1:
            mylabels.append("Females More")
            all_responses_counted.append(all_responses_in_letters.count("F"))
        if all_responses_in_letters.count("M") >= 1:
            mylabels.append("Males More")
            all_responses_counted.append(all_responses_in_letters.count("M"))
        if all_responses_in_letters.count("E") >= 1:
            mylabels.append("Equal")
            all_responses_counted.append(all_responses_in_letters.count("E"))
        pie_chart_values = np.array(all_responses_counted)

        fig, ax = plt.subplots()
        pie = ax.pie(pie_chart_values, labels=mylabels, autopct='%1.2f%%')
        ax.axis('equal')
        plt.title("Result for Languages with " + str(num_of_b_color_term) + " basic color terms", y=1.08)
        if len(mylabels) > 1:
            plt.legend(pie[0], mylabels, bbox_to_anchor=(1, 0), loc="lower right",
                       bbox_transform=plt.gcf().transFigure)
        plt.savefig("Pie_Charts/" + str(num_of_b_color_term))


def plot_pie_chart_for_all_langs(language_groups):
    """
    Plots one pie chart for all langauge groups. The pie chart shows the percentage of languages based on
     whether:
        1. Males used more unique color terms than females
        2. Females used more unique color terms than males
        3. Males and Females used the same number of unique color terms
    Parameters
    ----------
    language_groups: Dict
        key: Number of basic color terms
        Value: List of all languages that have key as the number of basic color terms. Each element of the list
            represents information about which gender group used more unique color terms for one language.

    Returns
    -------

    """
    all_responses_in_letters = []
    for language_group, languages in language_groups.items():
        for langauge_data in languages:
            all_responses_in_letters.append(langauge_data[1])
    mylabels = ["Females More", "Males More", "Equal"]
    all_responses_counted = [all_responses_in_letters.count("F"), all_responses_in_letters.count("M"),
                             all_responses_in_letters.count("E")]
    pie_chart_values = np.array(all_responses_counted)
    fig, ax = plt.subplots()
    pie = ax.pie(pie_chart_values, labels=mylabels, autopct='%1.2f%%')
    ax.axis('equal')
    plt.title("Result for All Languages Combined", y=1.08)
    if len(mylabels) > 1:
        plt.legend(pie[0], mylabels, bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=plt.gcf().transFigure)
    plt.savefig("Pie_Charts/All_Languages")


def plot_stacked_bar_chart(language_groups):
    """
    Plots and saves a stacked-bar chart representation of plot_pie_chart_for_all_langs.
    Parameters
    ----------
    language_groups: Dict
        key: Number of basic color terms
        Value: List of all languages that have key as the number of basic color terms. Each element of the list
            represents information about which gender group used more unique color terms for one language.
    """
    fig, ax = plt.subplots()
    x = sorted([language_group for language_group in language_groups])
    male_percentages = []
    female_percentages = []
    equal_percentages = []
    for language_group in x:
        all_responses_in_letters = [lang_data[1] for lang_data in language_groups[language_group]]
        female_more = all_responses_in_letters.count("F")
        male_more = all_responses_in_letters.count("M")
        equal_more = all_responses_in_letters.count("E")
        female_percentages.append(female_more / len(language_groups[language_group]) * 100)
        male_percentages.append(male_more / len(language_groups[language_group]) * 100)
        equal_percentages.append(equal_more / len(language_groups[language_group]) * 100)
    ax.bar(range(len(x)), female_percentages, width=0.8, label="Female More")
    ax.bar(range(len(x)), male_percentages, width=0.8, label="Male More", bottom=female_percentages)
    ax.bar(range(len(x)), equal_percentages, width=0.8, label="Equal",
           bottom=[x + y for x, y in zip(male_percentages, female_percentages)])

    for rect in ax.patches:
        height = rect.get_height()
        width = rect.get_width()

        if height == int(height):
            label_text = f'{int(height)}%'
        else:
            label_text = f'{height:.2f}%'

        label_x = rect.get_x() + width / 2
        label_y = rect.get_y() + height / 2
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)
    ax.set_ylabel("Percent of Gender Using More Unique Color Terms")
    ax.set_xlabel("Group Number")
    plt.xticks(range(len(x)), x)
    fig.set_size_inches(15, 7)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig("Stacked_Bar_Chart/stacked_bar_chart", bbox_inches="tight")


def generate_latex_tables():
    lst_group_permu_count = []
    lst_group_t_count = []
    total_t_count = [0, 0, 0]
    total_permu_count = [0, 0]

    lst_group_permu_percent = []
    lst_group_t_percent = []
    lst_group_comparison = []
    for group_index in lang_grouping.keys():
        group_t_count = [0, 0, 0]
        group_permu_count = [0, 0]
        group_comparison_count = [0, 0, 0]
        for member in lang_grouping[group_index]:
            t_test_str = member[2]
            comparison_str = member[1]
            if t_test_str == 'F':
                group_t_count[1] += 1
                total_t_count[1] += 1
            elif t_test_str == 'M':
                group_t_count[0] += 1
                total_t_count[0] += 1
            elif t_test_str == 'E':
                group_t_count[2] += 1
                total_t_count[2] += 1

            if comparison_str == 'F':
                group_comparison_count[1] += 1
            elif comparison_str == 'M':
                group_comparison_count[0] += 1
            elif comparison_str == 'E':
                group_comparison_count[2] += 1

            permu_str = member[6][2]
            if permu_str == 'alt':
                # null: mu_female
                group_permu_count[0] += 1
                total_permu_count[0] += 1
            elif permu_str == 'null':
                group_permu_count[1] += 1
                total_permu_count[1] += 1
        lst_group_t_count.append(group_t_count)
        lst_group_permu_count.append(group_permu_count)
        lst_group_comparison.append(group_comparison_count)

    for group in lst_group_t_count:
        n = sum(group)
        lst_group_t_percent.append([int(str_count / n * 100) for str_count in group])
    for group in lst_group_permu_count:
        n = sum(group)
        lst_group_permu_percent.append([int(str_count / n * 100) for str_count in group])
    # \begin{table}[H]
    print('\\begin{table}[H]')
    # \begin{center}
    print('\\begin{center}')
    #
    print('\\caption{Basic Color Term Grouping Results}')
    #
    print('\\label{bct-group-table}')
    print('\\vskip 0.12in')
    #
    #
    print('\\begin{tabular}{clll}')
    #
    print('\\hline')
    #
    print('Group $\#$     & Female More & Male More & Equal    \\\\')
    # \hline
    print('\\hline')
    #

    for group_ind in range(len(lst_group_comparison)):
        bct = list(lang_grouping.keys())[group_ind]
        percent = []
        for status_ind in range(len(lst_group_comparison[group_ind])):
            percent.append(lst_group_comparison[group_ind][status_ind] / sum(lst_group_comparison[group_ind]))
        print('%s       & %s(%s\\%%) & %s(%s\\%%) & %s(%s\\%%)\\\\' % (str(bct),
                                                                       str(lst_group_comparison[group_ind][1]),
                                                                       str(round(percent[1], 1) * 100),
                                                                       str(lst_group_comparison[group_ind][0]),
                                                                       str(round(percent[0], 1) * 100),
                                                                       str(lst_group_comparison[group_ind][2]),
                                                                       str(round(percent[2], 1) * 100),
                                                                       ))


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'serif'  # Globally change plot font to match latex font

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
        male_more_color_terms_than_female, female_more_color_terms_than_male, equal_color_terms = run_trials(50)

        # 5. Assign the language "F", "M", or "E" based on which letter has the highest count
        lang_index_is_female_more[language_index] = choose_m_f_e_for_lang_index()

        # 6. Run permutation test on number of unique terms used by all male and female speakers
        permutation_p_val_male, permutation_p_val_female, permutation_p_val_diff, permut_trial = run_permutation_test()

        # 7. Run T-test
        t_str = t_test()  # "E", "M", "F", "error"
        lang_index_is_female_more_t_test[language_index] = t_str

        # 8. Calculate the number of basic color terms for this language
        mean_of_basic_color_term_for_lang = round(get_num_of_basic_color_term(language_index))

        # 8.1 Organize all results for this language into a dictionary
        language_index_and_result = (language_index,
                                     lang_index_is_female_more[language_index],
                                     lang_index_is_female_more_t_test[language_index],
                                     permutation_p_val_male, permutation_p_val_female, permutation_p_val_diff,
                                     permut_trial)

        # 9. Group the language based on the number of color terms
        if mean_of_basic_color_term_for_lang in lang_grouping:
            lang_grouping[mean_of_basic_color_term_for_lang].append(language_index_and_result)
        else:
            lang_grouping[mean_of_basic_color_term_for_lang] = [language_index_and_result]

    # 10. Plot pie charts for each group
    plot_pie_charts_for_lang_groups(lang_grouping)
    # 10.1 Plot pie chart for all the languages as one group
    plot_pie_chart_for_all_langs(lang_grouping)

    # 11. Plot stacked bar chart
    plot_stacked_bar_chart(lang_grouping)
    # 12. Sort the groups in increasing order based on how many languages they contain
    lang_grouping_sorted = sort_by_values_len(lang_grouping)

    # 13. Get the most re-occurring key ('M','F','E') of each group and organize other results for display
    lang_grouping_unique_most_occur = \
        get_dict_to_list_by_fine_grained_gender_most_occurrence(lang_grouping_sorted, 1)
    lang_grouping_t_test_mean_most_occur = \
        get_dict_to_list_by_fine_grained_gender_most_occurrence(lang_grouping_sorted, 2)

    # 14. Print data in latex form to generate a table
    generate_latex_tables()
