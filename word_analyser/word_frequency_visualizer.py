import pandas as pd

#foad the CSV file
file_path = 'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/word_frequencies_per_year.csv'
df = pd.read_csv(file_path)

#ensure the first column is correctly named 'year' and set it as the index
df.rename(columns={df.columns[0]: 'year'}, inplace=True)
df['year'] = df['year'].astype(str) 
df.set_index('year', inplace=True)

#calculate the year-to-year difference
df_diff = df.diff().fillna(0)

#calculate average growth
avg_growth = df_diff.mean().sort_values(ascending=False)

#get the top 50 words by average growth
top_50_words = avg_growth.head(50).index

#create a DataFrame for the top 50 words with year-by-year growth
top_50_growth_df = df_diff[top_50_words]

#insert average growth column
top_50_growth_df['Average Growth'] = top_50_growth_df.mean(axis=1)

#reorder columns to have 'Average Growth' first
cols = ['Average Growth'] + [col for col in top_50_growth_df.columns if col != 'Average Growth']
top_50_growth_df = top_50_growth_df[cols]

#sort DataFrame by index to ensure years are in ascending order
top_50_growth_df = top_50_growth_df.sort_index()

#save the result to a CSV file
output_file_path = 'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/top_50_word_growth.csv'
top_50_growth_df.to_csv(output_file_path, index=True)

#print the result
print(top_50_growth_df)
