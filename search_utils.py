import aiohttp
from bs4 import BeautifulSoup
import asyncio
import urllib.parse
import random
import re


def _build_search_queries(sentence: str):
    text = re.sub(r"\s+", " ", sentence).strip()
    words = [w for w in re.findall(r"[a-zA-Z0-9']+", text) if len(w) > 2]

    queries = []
    if text:
        short_quote = " ".join(words[:10]).strip()
        medium_quote = " ".join(words[:16]).strip()
        if medium_quote:
            queries.append(f'"{medium_quote}"')
        if short_quote and short_quote != medium_quote:
            queries.append(f'"{short_quote}"')
        if short_quote:
            queries.append(short_quote)

    return queries[:3]


def _extract_snippets_from_html(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    snippets = []
    seen = set()

    # Strategy 1: Classic DDG HTML result blocks
    for result in soup.select('div.result')[:6]:
        link = result.select_one('a.result__a, a.result__url, h2 a')
        snippet_el = result.select_one('a.result__snippet, div.result__snippet, .result__body')
        if not link:
            continue

        raw_url = link.get('href', '').strip()
        if not raw_url:
            continue

        snippet_text = snippet_el.get_text(' ', strip=True) if snippet_el else ''
        if len(snippet_text) < 20:
            continue

        clean_url = raw_url
        if "uddg=" in raw_url:
            try:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                clean_url = parsed.get('uddg', [raw_url])[0]
            except Exception:
                clean_url = raw_url

        key = (clean_url, snippet_text[:120])
        if key in seen:
            continue
        seen.add(key)
        snippets.append({'snippet': snippet_text, 'url': clean_url})

    # Strategy 2: fallback for alternative DDG markup
    if not snippets:
        for a in soup.find_all('a', href=True):
            href = a.get('href', '').strip()
            text = a.get_text(' ', strip=True)
            if not href or not text:
                continue
            if 'duckduckgo.com' in href and 'uddg=' not in href:
                continue

            clean_url = href
            if 'uddg=' in href:
                try:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    clean_url = parsed.get('uddg', [href])[0]
                except Exception:
                    clean_url = href

            container_text = ''
            parent = a.find_parent(['div', 'article', 'li'])
            if parent:
                container_text = parent.get_text(' ', strip=True)
            snippet_text = container_text if len(container_text) >= 30 else text

            key = (clean_url, snippet_text[:120])
            if key in seen:
                continue
            seen.add(key)
            snippets.append({'snippet': snippet_text[:400], 'url': clean_url})

            if len(snippets) >= 5:
                break

    return snippets[:5]

async def search_internet_for_sentence_async(sentence: str, session: aiohttp.ClientSession):
    """
    Searches the internet for a sentence to simulate plagiarism detection using DuckDuckGo Html.
    """
    snippets = []
    queries = _build_search_queries(sentence)
    if not queries:
        return snippets

    endpoints = [
        "https://html.duckduckgo.com/html/",
        "https://duckduckgo.com/html/",
    ]

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # Add a tiny random delay to avoid hitting DDG too hard at exactly the same time
        await asyncio.sleep(random.uniform(0.1, 0.5))

        for query in queries:
            for endpoint in endpoints:
                data = {'q': query}
                try:
                    async with session.post(endpoint, data=data, headers=headers, timeout=12) as response:
                        if response.status != 200:
                            continue

                        html = await response.text()
                        extracted = _extract_snippets_from_html(html)
                        if extracted:
                            snippets.extend(extracted)
                            break
                except Exception:
                    continue

            if snippets:
                break

        # De-duplicate snippets while preserving order
        deduped = []
        seen = set()
        for item in snippets:
            key = (item.get('url', ''), item.get('snippet', '')[:120])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        snippets = deduped[:5]
    except Exception as e:
        print(f"Error fetching real sources: {repr(e)}")
        
    return snippets
