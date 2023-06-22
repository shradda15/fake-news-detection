import streamlit as st
import pandas as pd
import requests
import csv
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')



def get_keywords(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Count the frequency of each token
    keyword_counts = Counter(filtered_tokens)
    
    # Get the top 3 most frequent tokens as keywords
    keywords = [token for token, _ in keyword_counts.most_common(10)]
    
    return keywords

def check_keyword_match(sentences_to_compare):
    first_sentence_keywords = get_keywords(Title)
    match_count = 0
    print(first_sentence_keywords)
    
    for compare_sentence in sentences_to_compare:
        compare_sentence_tokens = word_tokenize(compare_sentence.lower())
        keyword_match_count = 0        
        for keyword in first_sentence_keywords:
            if keyword in compare_sentence_tokens:
                keyword_match_count += 1
        
        if keyword_match_count >=3:
            match_count += 1

    return match_count >=2





# #headline extraction
def scraper(Title):
    url = 'https://news.google.com/search'
    params = {'q': Title, 'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'}
    response = requests.get(url, params=params)
    print(requests.get(url, params=params))
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the first 3 headlines and URLs of the top news articles
    articles = soup.find_all('a', class_='DY5T1d')
    count = 0
    list1=[]
    for article in articles:
            title = article.text
            url = 'https://news.google.com' + article['href'][1:]
            list1.append(title)
            count += 1
            if count == 10:
                break
    # print(list1) 
    list2=[word.lower() for word in list1]    
    print(list2)   
    
    global result
    result = check_keyword_match(list2)
    return result




#checking input in csv file
def search_csv(Title):
    with open('file1.csv', 'r') as file:
        reader = csv.DictReader(file)
        keywords = get_keywords(Title)

        matching_results = []
        if len(keywords)>=3:
            for row in reader:
                title = row['TITLE']
                if all(keyword.lower() in title.lower() for keyword in keywords):
                    matching_results.append(row['RESULT'])
        if matching_results:
            return matching_results[0]
        else:
            return scraper(Title)

        



#message or news
def news_message(text):
# Load the dataset
    df = pd.read_csv('news_message.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['title'], df['category'], test_size=0.2, random_state=42)

    # Convert the text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier on the training set
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = classifier.predict(X_test)

    # Calculate the accuracy of the classifier on the testing set
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Use the classifier to predict the label of a new text
    X_new = vectorizer.transform([text])
    prediction = classifier.predict(X_new)[0] 
    if prediction == "news" :
        search_csv()
    else:
        global result
        # result = 'Irrelevant'
        search_csv()
        



# Define a function to write data to CSV
def write_to_csv(data):
    with open('file1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)  
    print("Data appended to CSV file.")     


# Create a form for user input
st.title("FAKE NEWS DETECTION")
with st.form("my_form"):
    # Add form inputs
    global url
    global Title
    #global result
    Title = st.text_input("Enter title")
    url = st.text_input("Enter url")

    # Add a submit button
    submit_button = st.form_submit_button(label='SUBMIT')
      
   

# When the form is submitted, write the data to CSV
global result
if submit_button:
    
    if "" in url:
        st.write("")
        result = search_csv(Title)
    else:
        result = search_csv(Title)
        
    #st.write(result)
    if  result=='False' or result==False or result=='FALSE':
        st.write("Result: Fake News")  
    elif result=='True' or result==True or result=='TRUE':
        st.write("Result: Real News")
    write_to_csv([Title , url , result])
