import streamlit as st
import pandas as pd
import requests
import csv
import easyocr
import re
import numpy as np
from PIL import Image
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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
global check_url


def is_mal(url):
    # Provide your VirusTotal API key
    api_key = '13b314d85cd7bf5122ccfd05637aedbf2d6513192161a1cb90fdee713f1e06c5'

    # Make a request to the VirusTotal API
    params = {'apikey': api_key, 'resource': url}
    response = requests.get('https://www.virustotal.com/vtapi/v2/url/report', params=params)

    if response.status_code == 200:
        result = response.json()

        # Check the response for malicious indicators
        if result['response_code'] == 1:
            if result['positives'] > 0:
                return True
            else:
                return False
        else:
            print("The URL is not in the VirusTotal database.")
    else:
        print("Failed to query the VirusTotal API.")


# extract url from title
def extract_urls(sentence):
    # Regular expression pattern for matching URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    # Find all matches of URLs in the sentence
    urls = re.findall(url_pattern, sentence)
    return urls

# extracts url from title
def extract_domains(string):
    domain_extensions = ['.com', '.org', '.net', '.ru', '.gov', '.edu', '.mil', '.int', '.biz', '.info', '.mobi', '.name', '.pro', '.coop', '.aero', '.travel', '.jobs', '.tel', '.xxx', '.govt', '.edu.au', '.app', '.blog', '.shop', '.tech', '.media', '.music', '.photography', '.design', '.agency', '.guru', '.club', '.store', '.online', '.app', '.studio', '.io', '.eo', '.to', '.xyz', '.us', '.me', '.site', '.ly']
    list1=string.split()
    list2=[]
    for i in range(0,len(domain_extensions)):
        for j in range(0,len(list1)):
            if domain_extensions[i] in list1[j]:
                list2.append(list1[j])
    return list2      

# return title without url
def extract_text_without_urls(string):
    url = extract_urls(string)
    if url ==[]:
        url = extract_domains(string)
    # Remove URLs from the string 
    for i in range(0,len(url)):
        text_without_urls = re.sub(url[i], "", string)
    return text_without_urls.strip()


import requests
from requests.exceptions import MissingSchema

def search_keywords(url, keywords):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            page_content = response.text.lower()
            num_keywords_present = 0
            for keyword in keywords:
                if keyword.lower() in page_content:
                    num_keywords_present += 1
            return num_keywords_present >= 2
        else:
            print("Failed to access the URL.")
            return False
    except MissingSchema:
        corrected_url = 'https://' + url
        return search_keywords(corrected_url, keywords)



# extracting the url of first headline
def get_first_headline_url(query):
    search_url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US%3Aen"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the first headline link
    # headline_link = soup.find('a', {'class': 'VDXfz'}).get('href')
    headline_link = soup.find('a', {'class': 'VDXfz'})
    if headline_link is not None:
        link = headline_link.get('href')
        # Construct the complete URL for the headline
        headline_url = f"https://news.google.com{link}"
        return headline_url
    else:
        st.write(" ")


# url appending in spreadsheets for phantom buster
def url_append(url,comment):
    # Path to the JSON file containing your service account credentials
    credentials_path = 'C:\\Users\\Vamshidhar\\OneDrive - White Cap\\Desktop\\ps2\Ps2\\streamlit\\fakenews-389613-078c3b88069d.json'
    # Google Spreadsheet URL
    spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1X1LjEug49gz8x2e8ebLLzfF-8qCuFtSNjupKIAmqwhw/edit?usp=sharing'
    credentials_path = 'C:\\Users\\Vamshidhar\\OneDrive - White Cap\\Desktop\\ps2\Ps2\\streamlit\\fakenews-389613-078c3b88069d.json'
    # Extract the spreadsheet ID from the URL
    spreadsheet_id = spreadsheet_url.split('/')[5]
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(credentials)
    # Open the spreadsheet by its ID
    spreadsheet = client.open_by_key(spreadsheet_id)
    # Select the first sheet in the spreadsheet
    worksheet = spreadsheet.get_worksheet(0)
    # Append the URL to the spreadsheet using CSV format
    csv_data = [[url]]
    global csv_string
    csv_string = '\n'.join([','.join(row) for row in csv_data])
    # Append the data to the worksheet
    worksheet.append_row(values=[url,comment])
    print("URL and comment appended successfully!")



# extracting imp keywords from title and insta caption
def get_keywords(sentence):
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    keyword_counts = Counter(filtered_tokens)
    keywords = [token for token, _ in keyword_counts.most_common(10)]
    return keywords


# Scrape the Instagram caption from the given post URL.

# def scrape_instagram_caption(post_url: str) -> str:
#     response = requests.get(post_url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     caption_element = soup.select_one('meta[property="og:description"]')
#     caption = caption_element['content'] if caption_element else ''
#     return caption

def scrape_instagram_caption(post_url: str) -> str:
    try:
        response = requests.get(post_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        caption_element = soup.select_one('meta[property="og:description"]')
        caption = caption_element['content'] if caption_element else ''
        return caption
    except MissingSchema:
        corrected_url = 'https://' + post_url
        return scrape_instagram_caption(corrected_url)


# removing unimp words from caption
def extract_words_after_given_word(sentence: str, given_word: str) -> str:
    words = sentence.split()
    try:
        index = words.index(given_word)
        words_after = ' '.join(words[index+1:])
        return words_after
    except ValueError:
        return ""
    
# checks match b/w title and caption
def check_match(sentence1, sentence2):
    keywords1 = get_keywords(sentence1)
    keywords2 = get_keywords(sentence2)
    matching_keywords = set(keywords1) & set(keywords2)
    print("Match:",matching_keywords)
    return len(matching_keywords) >= 1  



# keyword match b/w senternces in google news and title
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
        
        if keyword_match_count >=3:   #keywords in sentences
            match_count += 1   #sentences
        
    if match_count==0:
        
        st.write('Irrelevant')
        
    else:
        return match_count >=2    

def img(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        # Initialize the OCR reader
        reader = easyocr.Reader(['en'])
        # Perform OCR on the image array
        results = reader.readtext(img_array)
        # Extract the text from the OCR results
        text = ''
        for result in results:
            text += result[1] + ' '
    print(text)
    return text


# headline extraction
def scraper(news):
    url = 'https://news.google.com/search'
    params = {'q': news, 'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'}
    response = requests.get(url, params=params)
    # print(requests.get(url, params=params))
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the first 10 headlines and URLs of the top news articles
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
    list2=[word.lower() for word in list1]    
    print(list2)   
    global result
    result = check_keyword_match(list2)
    return result



#checking input in csv file
def search_csv(news):
    with open('file1.csv', 'r') as file:
        reader = csv.DictReader(file)
        keywords = get_keywords(news)

        matching_results = []
        if len(keywords)>=3:
            for row in reader:
                title = row['TITLE']
                if all(keyword.lower() in title.lower() for keyword in keywords):
                    matching_results.append(row['RESULT'])
        if matching_results:
            return matching_results[0]
        else:
            return scraper(news)
        

# print real news
def first(Title):
    url = 'https://news.google.com/search'
    params = {'q': Title, 'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'}
    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the first headline from the search results
    first_article = soup.find('a', class_='DY5T1d')
    headline = first_article.text
    return headline



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
    global url
    global Title
    global uploaded_file
    Title = st.text_area("Enter Title")
    url = st.text_input("Enter URL")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit_button = st.form_submit_button(label='SUBMIT')
      

check_url = extract_urls(Title) or extract_domains(Title)   

# When the form is submitted, write the data to CSV
global result
if submit_button:
    if uploaded_file:
        global txt2
        a = img(uploaded_file)
        result = search_csv(a)
        txt2 =False
        global headline
        headline = get_first_headline_url(a)
    elif check_url!=[]:
        txt2 =False #txt2 is used for extracting insta caption
        Title1 = extract_text_without_urls(Title)
        # url = extract_urls(Title)
        # if url ==[]:
        #     url = extract_domains(Title)
        url = check_url[0]
        key = get_keywords(Title1)    
        page = search_keywords(url,key)
        if page == True:
            result = scraper(Title1)
            mal_url=is_mal(check_url)
            # if mal_url == True:
            #     st.write("URL is not safe.")
            # else:
            #     st.write("URL is safe.") 
        else:
            st.write("Content in url not matched")
            mal_url=is_mal(check_url)
            if mal_url == True:
                st.write("URL is not safe.")
            else:
                st.write("URL is safe.")     
            result = ''
    elif Title and "instagram.com" in url:
        txt = scrape_instagram_caption(url)
        given_word = "Instagram:"
        headline = get_first_headline_url(Title)
        txt1 = extract_words_after_given_word(txt,given_word)
        txt2 = check_match(Title,txt1)
        if(txt2 == True):
            result = search_csv(Title)
            mal_url=is_mal(url)
            if mal_url == True:
                st.write("URL is not safe.")
            else:
                st.write("URL is safe.")    
        else:
            result=' '
            st.write("Content in url not matched.")    
            mal_url=is_mal(url)
            if mal_url == True:
                st.write("URL is not safe.")
            else:
                st.write("URL is safe.") 
    elif Title and "" in url:
        st.write("")
        headline = get_first_headline_url(Title)
        txt2 = False
        result = search_csv(Title)
    else:
        txt2 = False
        headline = get_first_headline_url(Title)
        result = search_csv(Title) 
    
    if txt2:
        result = search_csv(Title)
        if result == 'False' or result == False or result == 'FALSE':
            st.write("Result: Fake News")
            st.write("Check here for real news: ", headline)
            url_append(url, first(Title))
                # Convert the CSV string to bytes
            csv_bytes = csv_string.encode('utf-8')
            mal_url=is_mal(url)
            if mal_url == True:
                st.write("URL is not safe.")
            else:
                st.write("URL is safe.") 
            write_to_csv([Title , url , result])
    elif  result=='False' or result==False or result=='FALSE':
        st.write("Result: Fake News")
        st.write("Check here for real news: ", headline)
        mal_url=is_mal(url)
        if mal_url == True:
            st.write("URL is not safe.")
        else:
            st.write("URL is safe.") 
        write_to_csv([Title , url , result])
    elif result=='True' or result==True or result=='TRUE':
        st.write("Result: Real News")
        if not url=='':
            mal_url=is_mal(url)
            if mal_url == True:
                st.write("URL is not safe.")
            else:
                st.write("URL is safe.") 
        write_to_csv([Title , url , result])
    else:
        st.write(' ')  
        write_to_csv([Title , url , 'Irrelevant'])
