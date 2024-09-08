import aiohttp
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from bs4 import BeautifulSoup
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from markupsafe import Markup
from functools import lru_cache
from retrying import retry
import re
from fastapi.templating import Jinja2Templates
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

MAX_RESULTS = 10
MIN_SUMMARIES = 5

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ERROR_PHRASES = [
    "Access Denied", "No useful summary available", "Your access to the NCBI website",
    "possible misuse/abuse situation", "has been temporarily blocked",
    "is not an indication of a security issue", "a run away script",
    "to restore access", "please have your system administrator contact",
    "Log In", "Continue with phone number", "Email or username",
    "Password", "Forgot password"
]

@lru_cache(maxsize=128)
async def google_search(query, session):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Error in Google search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error performing search")

def parse_search_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='tF2Cxc')[:MAX_RESULTS]:
        title = g.find('h3')
        title = title.text if title else 'No title found'
        link = g.find('a')['href'] if g.find('a') else ''
        snippet = g.find('div', class_='VwiC3b')
        snippet = snippet.text if snippet else 'No snippet found'
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

def extract_youtube_video_id(url):
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(youtube_regex, url)
    return match.group(6) if match and 'channel' not in url else None

@retry(stop_max_attempt_number=3, wait_fixed=1000)
async def fetch_article(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
            else:
                response.raise_for_status()
    except Exception as e:
        logger.error(f"Error fetching article {url}: {str(e)}")
        return None

def is_valid_summary(summary):
    return not any(phrase in summary for phrase in ERROR_PHRASES)

async def fetch_and_summarize(url, session):
    try:
        video_id = extract_youtube_video_id(url)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        else:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            summary = article.summary

        if is_valid_summary(summary):
            return summary
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        try:
            html = await fetch_article(url, session)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all('p')
                text = ' '.join([para.get_text() for para in paragraphs[:5]])
                summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                if is_valid_summary(summary):
                    return summary
        except Exception as e:
            logger.error(f"Error fetching summary for {url}: {str(e)}")
    return None

def highlight_important_sentences(text, query, num_sentences=3):
    sentences = sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    freq_dist = FreqDist(words)
    query_words = set(query.lower().split())
    
    def sentence_importance(sentence):
        sentence_words = set(word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum())
        return sum(freq_dist[word] for word in sentence_words) + sum(5 for word in sentence_words if word in query_words)
    
    ranked_sentences = sorted([(sentence, sentence_importance(sentence)) for sentence in sentences], 
                              key=lambda x: x[1], reverse=True)
    
    highlighted_sentences = set(sentence for sentence, _ in ranked_sentences[:num_sentences])
    
    highlighted_text = []
    for sentence in sentences:
        if sentence in highlighted_sentences:
            highlighted_text.append(f'<mark class="highlight">{sentence}</mark>')
        else:
            highlighted_text.append(sentence)
    
    return ' '.join(highlighted_text)

def combine_summaries(summaries, query):
    combined_text = " ".join(summaries)
    if combined_text:
        sentences = sent_tokenize(combined_text)
        summary = ' '.join(sentences[:6])
        return highlight_important_sentences(summary, query)
    return "No useful summary available."

def filter_results(results):
    return [result for result in results if not any(phrase in result['snippet'] for phrase in ERROR_PHRASES)]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    logger.info(f"Search request received for query: {query}")
    try:
        async with aiohttp.ClientSession() as session:
            google_html = await google_search(query, session)
            google_results = parse_search_results(google_html)
            google_results = filter_results(google_results)

            google_summaries = []
            google_tasks = [fetch_and_summarize(result['link'], session) for result in google_results]
            google_summaries_results = await asyncio.gather(*google_tasks)

            for summary, result in zip(google_summaries_results, google_results):
                if summary and is_valid_summary(summary):
                    highlighted_summary = highlight_important_sentences(summary, query)
                    result['summary'] = Markup(highlighted_summary)
                    google_summaries.append(summary)

            if len(google_summaries) >= MIN_SUMMARIES:
                google_combined_summary = combine_summaries(google_summaries, query)
            else:
                google_combined_summary = "No useful summary available."

            return templates.TemplateResponse("results.html", {
                "request": request,
                "query": query,
                "google_results": google_results,
                "google_combined_summary": Markup(google_combined_summary),
            })
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during the search process")

@app.get("/suggestions", response_class=JSONResponse)
async def get_suggestions(query: str):
    try:
        url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data[1]
    except Exception as e:
        logger.error(f"Error fetching suggestions: {str(e)}")
        return []

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"HTTP error occurred: {exc.detail}"},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
