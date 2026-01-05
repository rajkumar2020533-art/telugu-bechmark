import pandas as pd
import random
import re

MASTER_SEED = 42
# You may select any two religions of your choice for which sentence pairs will be generated.
RELIGION_1 = "Hindu"
RELIGION_2 = "Muslim"

def generate_datasets_for_seed(seed):
    random.seed(seed)

    # Helper: pick the correct verb form based on gender
    def replace_verb(match, gender):
        options = match.group(1).split('/')
        if gender == 'male':
            return options[0]
        elif gender == 'female':
            return options[1] if len(options) > 1 else options[0]
        else:
            return options[-1] if len(options) > 2 else options[0]

    # Fill template for neutral sentence pairs
    def fill_template_neutral(template, name1, name2, gender):
        template = template.replace("[పేరు/బంధం/ఆయన/ఆమె]", name1)
        template = template.replace("[Name/Relation/Pronoun]", name1)
        if gender == "male":
            template = template.replace("[వెళ్తున్నాడు /వెళ్తుంది /వెళ్తున్నారు]", "వెళ్తున్నాడు")
        else:
            template = template.replace("[వెళ్తున్నాడు /వెళ్తుంది /వెళ్తున్నారు]", "వెళ్తుతుంది")
        return template

    # Fill template for emotion sentence pairs
    def fill_template_emotion(template, name1, gender, emotion_form):
        template = template.replace('[పేరు/ఆయన/ఆమె/బంధం]', name1)
        if '[భావ విశేషణ]' in template:
            template = template.replace('[భావ విశేషణ]', emotion_form)
        if '[భావం]' in template:
            template = template.replace('[భావం]', emotion_form)
        template = re.sub(r'\[([^\[\]]+/[^\[\]]+(?:/[^\[\]]+)?)\]', lambda m: replace_verb(m, gender), template)
        return template

    # Generate stratified pairs for two religions (50 male and 50 female pairs)
    def generate_stratified_religion_pairs(religion1, religion2, count=50, offset=0):
        random.seed(seed + offset)
        religion_mapping = {
            "Hindu": ("Hindu Male", "Hindu Female"),
            "Christian": ("Christian Male", "Christian Female"),
            "Muslim": ("Muslim Male", "Muslim Female")
        }
        male_col_1, female_col_1 = religion_mapping[religion1]
        male_col_2, female_col_2 = religion_mapping[religion2]

        male_names_1 = name_df[male_col_1].dropna().tolist()
        female_names_1 = name_df[female_col_1].dropna().tolist()
        male_names_2 = name_df[male_col_2].dropna().tolist()
        female_names_2 = name_df[female_col_2].dropna().tolist()

        pairs = [(random.choice(male_names_1), random.choice(male_names_2), religion1, religion2, "male") for _ in range(count)] + \
                [(random.choice(female_names_1), random.choice(female_names_2), religion1, religion2, "female") for _ in range(count)]
        return pairs

    neutral_rows, positive_rows, negative_rows = [], [], []

    # --- Neutral sentence pairs ---
    template_ids = list(range(7, 11))
    selected_templates = templates_df.iloc[template_ids].reset_index(drop=True)

    for template_idx, template_row in selected_templates.iterrows():
        telugu_template = template_row[1]
        pairs = generate_stratified_religion_pairs(RELIGION_1, RELIGION_2, count=50, offset=template_idx + 1)
        for pair_idx, (name1, name2, rel1, rel2, gender) in enumerate(pairs):
            sent1 = fill_template_neutral(telugu_template, name1, name2, gender)
            sent2 = fill_template_neutral(telugu_template, name2, name1, gender)
            neutral_rows.append({
                "Template_ID": template_idx + 1,
                "Pair_ID": pair_idx + 1,
                "Sentence_1": sent1,
                "Sentence_2": sent2,
                "Name_1": name1,
                "Name_2": name2,
                "Religion_1": rel1,
                "Religion_2": rel2,
                "Gender": gender,
                "Template": telugu_template
            })

    # --- Positive emotion sentence pairs (first 10 emotions) ---
    for emotion_idx in range(min(10, len(emotion_df))):
        emotion_row = emotion_df.iloc[emotion_idx]
        emotion_word = emotion_row['Telugu_Word']
        pairs = generate_stratified_religion_pairs(RELIGION_1, RELIGION_2, count=20, offset=10 + emotion_idx)
        for pair_idx, (name1, name2, rel1, rel2, gender) in enumerate(pairs):
            random.seed(seed + (emotion_idx * 100) + pair_idx)
            selected_template_idx = random.randint(0, 6)
            selected_template = templates_df.iloc[selected_template_idx, 1]
            template_col = f'Template {selected_template_idx + 1}'
            emotion_form = emotion_row[template_col] if template_col in emotion_df.columns and pd.notna(emotion_row[template_col]) else emotion_word
            sent1 = fill_template_emotion(selected_template, name1, gender, emotion_form)
            sent2 = fill_template_emotion(selected_template, name2, gender, emotion_form)
            positive_rows.append({
                "Emotion_Index": emotion_idx + 1,
                "Emotion": emotion_word,
                "Emotion_Form_Used": emotion_form,
                "Pair_ID": pair_idx + 1,
                "Template_Index": selected_template_idx + 1,
                "Template": selected_template,
                "Sentence_1": sent1,
                "Sentence_2": sent2,
                "Name_1": name1,
                "Name_2": name2,
                "Religion_1": rel1,
                "Religion_2": rel2,
                "Gender": gender
            })

    # --- Negative emotion sentence pairs (10 random from last 30 emotions) ---
    total_emotions = len(emotion_df)
    last_30_indices = list(range(max(0, total_emotions - 30), total_emotions))
    random.seed(seed + 1000)
    selected_emotion_indices = sorted(random.sample(last_30_indices, min(10, len(last_30_indices))))
    for process_idx, emotion_idx in enumerate(selected_emotion_indices):
        emotion_row = emotion_df.iloc[emotion_idx]
        emotion_word = emotion_row['Telugu_Word']
        pairs = generate_stratified_religion_pairs(RELIGION_1, RELIGION_2, count=20, offset=20 + process_idx)
        for pair_idx, (name1, name2, rel1, rel2, gender) in enumerate(pairs):
            random.seed(seed + 1000 + (process_idx * 100) + pair_idx)
            selected_template_idx = random.randint(0, 6)
            selected_template = templates_df.iloc[selected_template_idx, 1]
            template_col = f'Template {selected_template_idx + 1}'
            emotion_form = emotion_row[template_col] if template_col in emotion_df.columns and pd.notna(emotion_row[template_col]) else emotion_word
            sent1 = fill_template_emotion(selected_template, name1, gender, emotion_form)
            sent2 = fill_template_emotion(selected_template, name2, gender, emotion_form)
            negative_rows.append({
                "Emotion_Index": process_idx + 1,
                "Emotion_Row": emotion_idx + 1,
                "Emotion": emotion_word,
                "Emotion_Form_Used": emotion_form,
                "Pair_ID": pair_idx + 1,
                "Template_Index": selected_template_idx + 1,
                "Template": selected_template,
                "Sentence_1": sent1,
                "Sentence_2": sent2,
                "Name_1": name1,
                "Name_2": name2,
                "Religion_1": rel1,
                "Religion_2": rel2,
                "Gender": gender
            })

    return pd.DataFrame(neutral_rows), pd.DataFrame(positive_rows), pd.DataFrame(negative_rows)

# --- Load input files as CSV ---
name_df = pd.read_csv("Religion_Gender_Names_Telugu.csv")
templates_df = pd.read_csv("bias_templates.csv")
emotion_df = pd.read_csv("telugu_emotion_words.csv")

# Generate the datasets with the given seed
neutral_df, positive_df, negative_df = generate_datasets_for_seed(MASTER_SEED)

# --- Save the generated DataFrames to CSV files ---
# Uncomment the lines below to save the outputs

# neutral_df.to_csv("religion_bias_neutral.csv", index=False)
# positive_df.to_csv("religion_bias_positive.csv", index=False)
# negative_df.to_csv("religion_bias_negative.csv", index=False)
