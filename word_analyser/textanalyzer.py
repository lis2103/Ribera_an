import os
import re
import requests
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords
import nltk
import feedparser

#download NLTK stopwords
nltk.download('stopwords')

#define stopwords and add custom stop words
stop_words = set(stopwords.words('english'))
custom_stop_words = {'tz', 'th'} 
stop_words.update(custom_stop_words)

#function to clean and tokenize text
def clean_and_tokenize(text):
    text = text.lower()  #convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  #remove short words
    text = re.sub(r'\d+', '', text)  #remove numbers
    text = re.sub(r'\W+', ' ', text)  #remove non-alphanumeric characters
    tokens = text.split()  #tokenize by splitting on spaces
    tokens = [word for word in tokens if word not in stop_words]  #remove stopwords
    return tokens



def fetch_gdelt_articles(query, start_year, end_year):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&maxrecords=250&format=json&startdatetime={start_datetime}&enddatetime={end_datetime}&lang=English"
    valid_years = []
    
    for year in range(start_year, end_year + 1):
        start_datetime = f"{year}0101000000"
        end_datetime = f"{year}1231235959"
        folder = f'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/gdelt_articles/{year}'
        
        #skip fetching if directory already exists and contains files
        if os.path.exists(folder) and os.listdir(folder):
            print(f"Skipping GDELT articles for {year} as they already exist.")
            valid_years.append(year)
            continue
        
        url = base_url.format(query=query, start_datetime=start_datetime, end_datetime=end_datetime)
        response = requests.get(url)
        
        #check if the request was successful
        if response.status_code != 200:
            print(f"Error fetching articles for {year}: {response.status_code} - {response.text}")
            continue
        
        try:
            data = response.json()
            valid_years.append(year)  
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON for {year}: {response.text}")
            continue
        
        os.makedirs(folder, exist_ok=True)
        
        for i, result in enumerate(data.get('articles', [])):
            if result.get('language', '').lower() == 'english': 
                title = result.get('title', '')
                content = result.get('seendate', '') 
                with open(f"{folder}/article_{i+1}.txt", 'w', encoding='utf-8') as file:
                    file.write(title + "\n")
                    file.write(content + "\n")
    
    return valid_years

def fetch_rss_articles(feed_urls, keywords, year):
    folder = f'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/rss_articles/{year}'
    
    #skip fetching if directory already exists and contains files
    if os.path.exists(folder) and os.listdir(folder):
        print(f"Skipping RSS articles for {year} as they already exist.")
        return
    
    valid_articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if 'summary' in entry:
                content = entry.summary
            elif 'content' in entry:
                content = entry.content[0].value
            else:
                content = ''
            if any(keyword in content.lower() for keyword in keywords):
                valid_articles.append({
                    'title': entry.title,
                    'content': content,
                    'published': entry.published if 'published' in entry else ''
                })
    
    save_rss_articles(valid_articles, year)

def save_rss_articles(articles, year):
    folder = f'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/rss_articles/{year}'
    os.makedirs(folder, exist_ok=True)
    for i, article in enumerate(articles):
        with open(f"{folder}/article_{i+1}.txt", 'w', encoding='utf-8') as file:
            file.write(article['title'] + "\n")
            file.write(article['content'] + "\n")

def collect_data(queries, rss_feeds, start_year, end_year):
    valid_years_set = set()
    
    #extract keywords from queries
    keywords = set()
    for query in queries:
        #remove logical operators and special characters to extract keywords
        query_keywords = re.findall(r'\b\w+\b', query.lower())
        keywords.update(query_keywords)
    
    #fetch and save GDELT articles
    for query in queries:
        valid_years = fetch_gdelt_articles(query, start_year, end_year)
        valid_years_set.update(valid_years)
    
    #fetch and save RSS articles
    for year in range(start_year, end_year + 1):
        fetch_rss_articles(rss_feeds, keywords, year)
        valid_years_set.add(year)
    
    return valid_years_set


def process_texts_in_directory(directory):
    word_counts_per_year = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                year = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                #read text from file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                #clean and tokenize text
                tokens = clean_and_tokenize(text)
                word_count = Counter(tokens)
                
                if year not in word_counts_per_year:
                    word_counts_per_year[year] = Counter()
                
                word_counts_per_year[year] += word_count
    
    #normalize word counts by total words (per 1000 words) for each year
    normalized_word_counts = {}
    for year, word_count in word_counts_per_year.items():
        total_words = sum(word_count.values())
        normalized_word_counts[year] = {word: (count / total_words) * 1000 for word, count in word_count.items()}
    
    return normalized_word_counts

def process_data(valid_years_set):
    #process the text files and get word counts
    directory = 'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/gdelt_articles' 
    rss_directory = 'C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/rss_articles'
    word_counts_per_year = process_texts_in_directory(directory)
    rss_word_counts_per_year = process_texts_in_directory(rss_directory)
    
    #merge the two dictionaries
    for year, counts in rss_word_counts_per_year.items():
        if year in word_counts_per_year:
            word_counts_per_year[year].update(counts) 
        else:
            word_counts_per_year[year] = Counter(counts)
    
    #convert the word counts per year to a DataFrame
    df = pd.DataFrame.from_dict(word_counts_per_year, orient='index').fillna(0)
    df = df[df.index.isin(map(str, valid_years_set))]
    
    #save
    df.to_csv('C:/Users/ANTOI/Programming_Projects/Ribera Project/Code/Word Analyser/word_frequencies_per_year.csv')
    print("Word frequencies per year saved to 'word_frequencies_per_year.csv'")

queries = [
    '("silver economy" OR health OR fitness OR "healthcare technology" OR "elderly care" OR wellness OR "public health" OR "mental health")',
    '("telemedicine" OR "remote monitoring" OR "digital health" OR "health apps" OR "wearable technology" OR "elderly healthcare")',
    '("chronic disease management" OR "preventive care" OR "health promotion" OR "healthy aging" OR "nutrition")',
    '("personalized medicine" OR "physical therapy" OR "mental wellness" OR "fitness trends")'
]
rss_feeds = [
    'http://feeds.bbci.co.uk/news/rss.xml',
    'http://feeds.reuters.com/reuters/topNews'
]
start_year = 2017
end_year = 2024

valid_years_set = collect_data(queries, rss_feeds, start_year, end_year)
process_data(valid_years_set)
