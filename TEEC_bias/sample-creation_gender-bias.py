import pandas as pd
import random
import re

MASTER_SEED = 42

def generate_datasets_for_seed(seed):
    random.seed(seed)

    # --- NEUTRAL SENTENCE PAIRS ---
    template_ids_neutral = list(range(7, 11))
    selected_templates_neutral = templates_df.iloc[template_ids_neutral].reset_index(drop=True)

    # Generate name pairs for each group
    name_pairs_neutral = []
    name_pairs_neutral += generate_name_pairs(0, 1, 33, "Hindu")
    name_pairs_neutral += generate_name_pairs(4, 5, 33, "Muslim")
    name_pairs_neutral += generate_name_pairs(2, 3, 34, "Christian")

    # Add random pronoun pairs for gender-neutral evaluation
    all_pronouns_neutral = list(zip(pronoun_df['Male (Telugu)'].dropna(), pronoun_df['Female (Telugu)'].dropna()))
    pronoun_pairs_neutral = random.sample(all_pronouns_neutral, 10)
    pronoun_pairs_neutral = [(m, f, "Pronoun") for m, f in pronoun_pairs_neutral]

    all_pairs_neutral = name_pairs_neutral + pronoun_pairs_neutral

    neutral_rows = []
    for t_index, row in selected_templates_neutral.iterrows():
        telugu_template = row[1]
        for m_name, f_name, group in all_pairs_neutral:
            # Male sentence
            sent1 = telugu_template.replace("[Name/Relation/Pronoun]", m_name)
            sent1 = sent1.replace("[పేరు/బంధం/ఆయన/ఆమె]", m_name)
            sent1 = sent1.replace("[వెళ్తున్నాడు /వెళ్తుంది /వెళ్తున్నారు]", "వెళ్తున్నాడు")
            # Female sentence
            sent2 = telugu_template.replace("[Name/Relation/Pronoun]", f_name)
            sent2 = sent2.replace("[పేరు/బంధం/ఆయన/ఆమె]", f_name)
            sent2 = sent2.replace("[వెళ్తున్నాడు /వెళ్తుతుంది /వెళ్తున్నారు]", "వెళ్తుతుంది")
            neutral_rows.append({
                "Sentence 1 (male)": sent1,
                "Sentence 2 (female)": sent2,
                "Name1": m_name,
                "Name2": f_name,
                "Group": group,
                "Source": "name" if group != "Pronoun" else "pronoun",
                "Template_ID": t_index + 1
            })

    # --- POSITIVE EMOTION SENTENCE PAIRS ---
    templates_emotion = templates_df.iloc[:7, 1].tolist()
    output_rows_positive = []
    for emotion_idx in range(min(10, len(emotion_df))):
        emotion_row = emotion_df.iloc[emotion_idx]
        emotion_word = emotion_row['Telugu_Word']
        pairs = get_stratified_pairs(seed_offset=1000 + emotion_idx * 10)
        for pair_idx, (male_name, female_name, group) in enumerate(pairs):
            random.seed(seed + 2000 + emotion_idx * 100 + pair_idx)
            selected_template_idx = random.randint(0, len(templates_emotion) - 1)
            selected_template = templates_emotion[selected_template_idx]
            template_col = f'Template {selected_template_idx + 1}'
            # Use the custom emotion form if available for the template, else default to Telugu word
            emotion_form = emotion_row[template_col] if template_col in emotion_df.columns and pd.notna(emotion_row[template_col]) else emotion_word
            sent1 = fill_template(selected_template, male_name, 'male', emotion_form)
            sent2 = fill_template(selected_template, female_name, 'female', emotion_form)
            output_rows_positive.append({
                'Emotion_Type': 'Positive',
                'Emotion_Index': emotion_idx + 1,
                'Template_Index': selected_template_idx + 1,
                'Template': selected_template,
                'Sentence 1': sent1,
                'Sentence 2': sent2,
                'Group': group,
                'Male_Name': male_name,
                'Female_Name': female_name,
                'Emotion': emotion_word,
                'Emotion_Form_Used': emotion_form
            })

    # --- NEGATIVE EMOTION SENTENCE PAIRS ---
    output_rows_negative = []
    if len(emotion_df) >= 40:
        negative_emotions_indices = list(range(10, 40))
        random.seed(seed + 5000)
        selected_negative_indices = random.sample(negative_emotions_indices, 10)
        for neg_idx, emotion_idx in enumerate(selected_negative_indices):
            emotion_row = emotion_df.iloc[emotion_idx]
            emotion_word = emotion_row['Telugu_Word']
            pairs = get_stratified_pairs(seed_offset=6000 + emotion_idx * 10)
            for pair_idx, (male_name, female_name, group) in enumerate(pairs):
                random.seed(seed + 7000 + emotion_idx * 100 + pair_idx)
                selected_template_idx = random.randint(0, len(templates_emotion) - 1)
                selected_template = templates_emotion[selected_template_idx]
                template_col = f'Template {selected_template_idx + 1}'
                emotion_form = emotion_row[template_col] if template_col in emotion_df.columns and pd.notna(emotion_row[template_col]) else emotion_word
                sent1 = fill_template(selected_template, male_name, 'male', emotion_form)
                sent2 = fill_template(selected_template, female_name, 'female', emotion_form)
                output_rows_negative.append({
                    'Emotion_Type': 'Negative',
                    'Emotion_Index': neg_idx + 1,
                    'Original_Index': emotion_idx + 1,
                    'Template_Index': selected_template_idx + 1,
                    'Template': selected_template,
                    'Sentence 1': sent1,
                    'Sentence 2': sent2,
                    'Group': group,
                    'Male_Name': male_name,
                    'Female_Name': female_name,
                    'Emotion': emotion_word,
                    'Emotion_Form_Used': emotion_form
                })

    # Combine all results into DataFrames
    neutral_df = pd.DataFrame(neutral_rows)
    positive_df = pd.DataFrame(output_rows_positive)
    negative_df = pd.DataFrame(output_rows_negative)
    return neutral_df, positive_df, negative_df

# --- Load input files as CSV ---
name_df = pd.read_csv("Religion_Gender_Names_Telugu.csv")
pronoun_df = pd.read_csv("Noun_Phrases.csv")
templates_df = pd.read_csv("bias_templates.csv")
emotion_df = pd.read_csv("telugu_emotion_words.csv")

# --- Helper functions ---
def generate_name_pairs(male_col, female_col, count, group):
    # Generate (male, female) name pairs for a group, sampled with replacement
    males = name_df.iloc[:, male_col].dropna().tolist()
    females = name_df.iloc[:, female_col].dropna().tolist()
    return [(random.choice(males), random.choice(females), group) for _ in range(count)]

def replace_verb(match, gender):
    # Pick the correct verb form based on gender
    options = match.group(1).split('/')
    if gender == 'male': return options[0]
    elif gender == 'female': return options[1] if len(options) > 1 else options[0]
    else: return options[-1] if len(options) > 2 else options[0]

def fill_template(template, name_or_pronoun, gender, emotion_form):
    # Fill the template with the given name/pronoun, emotion, and gender-specific verb
    template = template.replace('[పేరు/ఆయన/ఆమె/బంధం]', name_or_pronoun)
    template = template.replace('[భావ విశేషణ]', emotion_form)
    template = template.replace('[భావం]', emotion_form)
    return re.sub(r'\[([^\[\]]+/[^\[\]]+(?:/[^\[\]]+)?)\]', lambda m: replace_verb(m, gender), template)

def get_stratified_pairs(seed_offset=0):
    # For each group, get 13 male/female name pairs and 5 pronoun pairs
    random.seed(MASTER_SEED + seed_offset)
    pairs = []
    for group in ['Christian', 'Muslim', 'Hindu']:
        male_col = f'{group} Male'
        female_col = f'{group} Female'
        male_names = name_df[male_col].dropna().sample(13, random_state=MASTER_SEED + seed_offset)
        female_names = name_df[female_col].dropna().sample(13, random_state=MASTER_SEED + seed_offset + 1)
        for m, f in zip(male_names, female_names):
            pairs.append((m, f, group))
    pmale = pronoun_df['Male (Telugu)'].dropna().sample(5, random_state=MASTER_SEED + seed_offset + 2).tolist()
    pfemale = pronoun_df['Female (Telugu)'].dropna().sample(5, random_state=MASTER_SEED + seed_offset + 3).tolist()
    for pm, pf in zip(pmale, pfemale):
        pairs.append((pm, pf, 'Pronoun'))
    return pairs

neutral_df, positive_df, negative_df = generate_datasets_for_seed(MASTER_SEED)

# --- Save the generated DataFrames to CSV files ---
# Uncomment the lines below to save the outputs

# neutral_df.to_csv("gender_bias_neutral.csv", index=False)
# positive_df.to_csv("gender_bias_positive.csv", index=False)
# negative_df.to_csv("gender_bias_negative.csv", index=False)
