import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import time
from urllib.parse import urlparse, urljoin
import concurrent.futures


async def scrape_page(url, *, session):
    pr = urlparse(url)
    if pr.scheme == 'file':
        async with aiofiles.open(pr.path) as fin:
            html_doc = await fin.read()
    else:
        async with session.get(pr.geturl()) as resp:
            html_doc = await resp.text()
    return html_doc


def parse_page(html_doc, url):
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, urljoin
    soup = BeautifulSoup(html_doc, 'html.parser')
    raw_links = soup.find_all('a')
    local_fragments = []
    real_links = []
    for l in raw_links:
        pl = urlparse(l.get('href'))
        if pl.fragment and not pl.path:
            local_fragments.append(pl.geturl())
        else:
            real_links.append(pl.geturl())

    # TODO check fragments

    return [urljoin(url, l) for l in real_links]


async def check_exists(url, *, session, last_hit):
    pr = urlparse(url)
    if pr.scheme == 'file':
        return Path(pr.path).exists()
    else:
        last_hit_time = last_hit.get(pr.netloc, 0)
        cur_time = time.monotonic()
        # keep to under 10Hz
        sleep_time = max(0, (1 + last_hit_time) - cur_time)
        last_hit[pr.netloc] = cur_time + 1 + sleep_time
        if sleep_time:
            await asyncio.sleep(sleep_time)
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(resp.status)
                    if resp.status == 429:
                        last_hit[pr.netloc] = last_hit[pr.netloc] + 120
                        return await check_exists(url, session=session, last_hit=last_hit)
                return resp.status == 200
        except Exception:
            return False


async def worker(seen, queue, session, base_url,
                 check_external, executor, last_hit):
    while True:
        url = await queue.get()
        if url.startswith(base_url):
            if url.endswith('html'):
                try:
                    html_doc = await scrape_page(url, session=session)
                except Exception:
                    print(f'loading {url} failed')
                try:
                    links = await asyncio.get_event_loop().run_in_executor(
                        executor, parse_page, html_doc, url)
                except Exception:
                    print(f'parsing {url} failed')
                for l in links:
                    if l not in seen:
                        seen.add(l)
                        await queue.put(l)
            else:
                if not (await check_exists(url, session=session,
                                           last_hit=last_hit)):
                    print(f'{url} does not exist')
        elif check_external:
            if not (await check_exists(url, session=session,
                                       last_hit=last_hit)):
                print(f'{url} does not exist')

        queue.task_done()


async def leader(base_url, start_page='', check_external=False):
    seen = set()
    last_hit = {}
    q = asyncio.Queue()
    await q.put(urljoin(base_url, start_page))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.Task(
                worker(
                    seen, q, session, base_url,
                    check_external, executor, last_hit)
            ) for j in range(32)]

            await q.join()
            for t in tasks:
                t.cancel()

# loop = asyncio.new_event_loop()
# loop.run_until_complete(leader('file:///home/tcaswell/source/p/matplotlib/doc/build/html/', 'index.html', check_external=False))
# loop.run_until_complete(leader('file:///home/tcaswell/source/p/matplotlib/doc/build/html/', 'index.html', check_external=True))
