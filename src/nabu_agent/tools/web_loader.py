import asyncio
import logging
import os
from typing import List

import httpx
import trafilatura
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import SearxSearchWrapper
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

load_dotenv()


# def search_and_fetch(query: str, num_results: int = 3, chunk_size: int = 500) -> str:
#     # search via SearxNG
#     searx = SearxSearchWrapper(searx_host=os.environ["SEARX_HOST"])
#     results = searx.results(query, num_results=num_results)

#     output_texts = []

#     # fetch page content
#     for i, r in enumerate(results):
#         url = r["link"]
#         print(url)
#         try:
#             loader = WebBaseLoader(web_path=url)
#             docs = loader.lazy_load()
#             content = ""
#             for doc in docs:
#                 # take only first document
#                 content += doc.page_content.replace("\n", "")[:chunk_size]

#                 # format nicely with source

#             output_texts.append(f"###\n{content}")
#         except Exception:
#             logger.info(f"Url: {url} failed to load.")

#     # Combine into single text block
#     print(output_texts)
#     return "\n".join(output_texts)


async def fetch_with_playwright(url: str, timeout: int = 15000) -> str:
    """Use Playwright to render JS-heavy pages."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=timeout)
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        logger.warning(f"Playwright fetch failed for {url}: {e}")
        return ""


async def fetch_content(url: str, use_playwright_fallback: bool = True) -> str:
    """Try fetching with httpx + Trafilatura, fallback to Playwright if needed."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(url)
            html = resp.text
            text = trafilatura.extract(
                html, include_comments=False, include_tables=False
            )
            if text:
                return text.strip()
            elif use_playwright_fallback:
                html = await fetch_with_playwright(url)
                text = trafilatura.extract(html)
                return text.strip() if text else ""
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return ""


@tool
async def search_internet(query: str) -> str:
    """
    Search the web for current and factual information.

    Use this tool when:
    - The user asks about **recent events**, **news**, **statistics**, **companies**, **products**, or any topic requiring up-to-date data.
    - You need to verify or fact-check a detail that may have changed over time.

    The tool performs a search using SearxNG, then fetches and extracts readable page content
    from the top results using Trafilatura (for clean text extraction) and Playwright (for dynamic pages).

    Args:
        query (str): The query describing what to search for.

    Returns:
        str: A summarized snippet of relevant text from multiple web sources,
        including brief source attributions.
    """
    num_results = 2
    chunk_size = 1500
    searx = SearxSearchWrapper(searx_host=os.environ["SEARX_HOST"])
    results = searx.results(query, num_results=num_results)
    urls = [r["link"] for r in results]

    tasks = [fetch_content(url) for url in urls]
    contents = await asyncio.gather(*tasks)

    output_texts = []
    for url, content in zip(urls, contents):
        if content:
            snippet = content[:chunk_size].replace("\n", " ")
            output_texts.append(f"### Source: {url}\n{snippet}")
        else:
            logger.info(f"Url: {url} returned no content.")

    return "\n\n".join(output_texts)
