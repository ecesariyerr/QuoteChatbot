import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def scrape_quotes():
    base_url = 'http://quotes.toscrape.com/'
    url = base_url
    all_quotes = []

    while url:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        for q in soup.find_all('div', class_='quote'):
            text   = q.find('span', class_='text').get_text()
            author = q.find('small', class_='author').get_text()
            tags   = [t.get_text() for t in q.find_all('a', class_='tag')]
            all_quotes.append({'text': text, 'author': author, 'tags': tags})

        next_li = soup.find('li', class_='next')
        if next_li:
            url = urljoin(base_url, next_li.find('a')['href'])
        else:
            url = None

    return all_quotes

def save_to_json(data, filename='quotes.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{len(data)} alıntı {filename} dosyasına kaydedildi.")

def filter_by_author(quotes_list, author_name):
    return [q for q in quotes_list if q['author'] == author_name]

def search_quotes(vectorizer, tfidf_matrix, quotes, query, top_k=5):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idxs = scores.argsort()[-top_k:][::-1]
    return [
        {'text': quotes[i]['text'], 'author': quotes[i]['author'], 'score': float(scores[i])}
        for i in top_idxs
    ]

if __name__ == '__main__':
    # 1) Scrape ve JSON'a kaydet
    quotes = scrape_quotes()
    save_to_json(quotes)

    # 2) Örnek filtreleme
    einstein_quotes = filter_by_author(quotes, 'Albert Einstein')
    print(f"Albert Einstein için {len(einstein_quotes)} alıntı bulundu.")
    for q in einstein_quotes:
        print('-', q['text'])

    # 3) TF–IDF matrisi oluştur
    texts = [q['text'] for q in quotes]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 4) Dinamik sorgu döngüsü
    while True:
        query = input("\nAramak istediğiniz kelimeyi girin (çıkmak için 'exit'): ")
        if query.lower() in ('exit', 'quit'):
            print("Program sonlandırılıyor…")
            break

        results = search_quotes(vectorizer, tfidf_matrix, quotes, query, top_k=5)
        print(f"\n“{query}” için en benzer 5 alıntı:")
        for r in results:
            print(f"{r['score']:.3f} — {r['text']} ({r['author']})")
