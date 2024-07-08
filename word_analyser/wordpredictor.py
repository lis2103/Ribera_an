import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import spacy
from nltk.corpus import stopwords

#load the spaCy model
nlp = spacy.load("en_core_web_sm")

#load the intermediate data
intermediate_file_path = 'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/top_50_word_growth.csv'
intermediate_df = pd.read_csv(intermediate_file_path)

#ensure the 'year' column is correctly set as the index
intermediate_df['year'] = intermediate_df['year'].apply(lambda x: x.split(',')[0].strip("()'"))
intermediate_df.set_index('year', inplace=True)

#convert the index to integer
intermediate_df.index = intermediate_df.index.astype(int)

#define the years variable correctly
years = np.array(intermediate_df.index).reshape(-1, 1)

#list all columns in the dataset to understand the available words
all_words = intermediate_df.columns 
print("Available Words:", all_words)

#define stopwords and add custom stop words
stop_words = set(stopwords.words('english'))
custom_stop_words = {'dobrev'}  #strange term that kept popping up
stop_words.update(custom_stop_words)

#function to check if a word is a proper noun using spaCy
def is_proper_noun(word):
    doc = nlp(word)
    for token in doc:
        if token.ent_type_ in ['PERSON', 'GPE', 'ORG']:  #proper nouns, places, organizations
            return True
    return False

#filter the words to exclude stop words and proper nouns
filtered_words = [word for word in all_words if word.lower() not in stop_words and not is_proper_noun(word)]
print("Filtered Words:", filtered_words)

#create a DF to store the projected growth rates
projections = pd.DataFrame(columns=['word', 'projected_growth'])

#linear regression to project the growth rates
for word in filtered_words:
    print(f"Processing word: {word}")  # Debug: Print the word being processed
    growth_rates = intermediate_df[word].values
    if len(set(growth_rates)) <= 1:
        print(f"Skipping word '{word}' due to insufficient variation in growth rates.")
        continue 
    model = LinearRegression()
    model.fit(years, growth_rates)
    projected_growth = model.predict(np.array([[2028]]))[0]  
    new_row = pd.DataFrame({'word': [word], 'projected_growth': [projected_growth]})
    projections = pd.concat([projections, new_row], ignore_index=True)

#print the projections for all words
print("Projections DataFrame:")
print(projections)

#ensure 'projected_growth' column is numeric
projections['projected_growth'] = pd.to_numeric(projections['projected_growth'], errors='coerce')

#filter out negative growth rates and get the top 30 words with positive growth rates
top_30_words = projections[projections['projected_growth'] > 0].nlargest(30, 'projected_growth')

#if fewer than 30 positive words are found, take the top available
if len(top_30_words) < 30:
    top_30_words = projections.nlargest(30, 'projected_growth')

#print the top 30 words DataFrame
print("Top 30 Words DataFrame:")
print(top_30_words)

#plot the results
plt.figure(figsize=(14, 8))
plt.bar(top_30_words['word'], top_30_words['projected_growth'], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Words')
plt.ylabel('Projected Growth Rate')
plt.title('Top 30 Words Projected to Grow the Most in Use Over the Next 5 Years')
plt.show()
