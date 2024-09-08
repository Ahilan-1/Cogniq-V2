import aiohttp
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
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
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

MAX_RESULTS = 10
MIN_SUMMARIES = 5

# Initialize the FastAPI app
app = FastAPI()

# Add CORS Middleware to allow cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain(s)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", clean_up_tokenization_spaces=True)


ERROR_PHRASES = [
    "Access Denied", "No useful summary available", "Your access to the NCBI website",
    "possible misuse/abuse situation", "has been temporarily blocked", "is not an indication of a security issue",
    "a run away script", "to restore access", "please have your system administrator contact",
    "Log In", "Continue with phone number", "Email or username", "you agree to our Terms of Service", 
    "Cookie Policy", "Privacy Policy", "Content Policies.", "Password", "Forgot password", "cookies", "Accept", "Cookie Settings"
]

# Function to perform Google search and fetch results
@lru_cache(maxsize=128)
async def google_search(query, session):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to parse Google search results
def parse_search_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='tF2Cxc')[:MAX_RESULTS]:
        title = g.find('h3').text if g.find('h3') else 'No title found'
        link = g.find('a')['href']
        snippet = g.find('div', class_='VwiC3b').text if g.find('div', class_='VwiC3b') else 'No snippet found'
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

# Function to extract YouTube video ID
def extract_youtube_video_id(url):
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    youtube_match = re.match(youtube_regex, url)
    if youtube_match and 'channel' not in url:
        return youtube_match.group(6)
    return None

# Function to fetch an article
@retry(stop_max_attempt_number=3, wait_fixed=1000)
async def fetch_article(url, session):
    async with session.get(url, timeout=10) as response:
        if response.status == 200:
            return await response.text()
        else:
            response.raise_for_status()

# Check if the summary is valid
def is_valid_summary(summary):
    return not any(phrase in summary for phrase in ERROR_PHRASES)

# Fetch and summarize the article
async def fetch_and_summarize(url, session):
    try:
        video_id = extract_youtube_video_id(url)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            if is_valid_summary(summary):
                return summary
        else:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            if is_valid_summary(article.summary):
                return article.summary
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return None

# Handle search request
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    logger.debug(f"Search query: {query}")
    async with aiohttp.ClientSession() as session:
        google_html = await google_search(query, session)
        google_results = parse_search_results(google_html)

        summaries = await asyncio.gather(*[fetch_and_summarize(result['link'], session) for result in google_results])
        summaries = [summary for summary in summaries if summary]

        combined_summary = "No useful summary available."
        if summaries:
            combined_summary = " ".join(summaries)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "query": query,
            "google_results": google_results,
            "combined_summary": Markup(combined_summary),
        })

# Handle suggestions request
@app.get("/suggestions", response_class=JSONResponse)
async def get_suggestions(query: str):
    logger.debug(f"Suggestions query: {query}")
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

# Exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"HTTP error occurred: {exc.detail}"},
    )

# Exception handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )

# Run the FastAPI app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
