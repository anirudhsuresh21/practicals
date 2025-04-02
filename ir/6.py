import requests
from bs4 import BeautifulSoup

def crawl_website(url, word):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).text, "html.parser")
        links = [link.get("href") for link in soup.find_all("a")]
        matches = list(set(link for link in links if link and word.lower() in link.lower()))
        print(f"=== Crawler Results ===\nTotal links: {len(links)}\nMatches for '{word}': {len(matches)}")
        for i, link in enumerate(matches, 1):
            print(f"{i}. {link}")
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")

if __name__ == "__main__":
    crawl_website("https://facebook.com", "login")
