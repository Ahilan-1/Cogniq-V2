import aiohttp
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from markupsafe import Markup
from functools import lru_cache
from retrying import retry
import re
from fastapi.templating import Jinja2Templates
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

# Summarization model and search settings
MAX_RESULTS = 10
MIN_SUMMARIES = 5

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow external mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

templates = Jinja2Templates(directory="templates")

# Summarization pipeline setup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ERROR_PHRASES = [
    "Access Denied", "No useful summary available", "Your access to the NCBI website",
    "possible misuse/abuse situation", "has been temporarily blocked", "is not an indication of a security issue",
    "a run away script", "to restore access", "please have your system administrator contact",
    "Log In", "Continue with phone number", "Email or username", "Password", "Forgot password"
]

# Function to search Google and return raw HTML
@lru_cache(maxsize=128)
async def google_search(query, session):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to parse search results from Google
def parse_search_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='tF2Cxc')[:MAX_RESULTS]:
        title = g.find('h3').text if g.find('h3') else 'No title found'
        link = g.find('a')['href']
        snippet = g.find('div', class_='VwiC3b').text if g.find('div', class_='VwiC3b') else 'No snippet found'
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

# Function to clean up a summary (remove unwanted phrases)
def clean_summary(summary):
    for phrase in ERROR_PHRASES:
        summary = summary.replace(phrase, "")
    return summary.strip()

# Function to fetch and summarize the article or YouTube transcript
async def fetch_and_summarize(url, session):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        summary = clean_summary(article.summary)
        if len(summary) > 0:
            return summary
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return None

# Function to combine summaries into one readable text
def combine_summaries(summaries):
    combined_text = " ".join(summaries)
    if combined_text:
        return combined_text
    return "No useful summary available."

# Route for the main search page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Search function to process queries and fetch summaries
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    async with aiohttp.ClientSession() as session:
        google_html = await google_search(query, session)
        google_results = parse_search_results(google_html)
        
        summaries = await asyncio.gather(*[fetch_and_summarize(result['link'], session) for result in google_results])
        valid_summaries = [summary for summary in summaries if summary]
        
        if len(valid_summaries) >= MIN_SUMMARIES:
            combined_summary = combine_summaries(valid_summaries)
        else:
            combined_summary = "No useful summary available."

        return templates.TemplateResponse("result.html", {
            "request": request,
            "query": query,
            "google_results": google_results,
            "google_combined_summary": Markup(combined_summary),
        })

# Autocomplete suggestions API
@app.get("/suggestions", response_class=JSONResponse)
async def get_suggestions(query: str):
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={query}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                try:
                    data = json.loads(text)
                    return data[1]
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse suggestions for query: {query}")
                    return []
    return []

# Running the FastAPI application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
