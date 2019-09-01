from calendar import monthrange, month_name
from bs4 import BeautifulSoup
import re
from bs4.element import Tag
from typing import *
from tqdm import tqdm
import cv2
import numpy as np

import aiohttp
import aiofiles
import asyncio
import requests

event_loop = asyncio.get_event_loop()


def all_days_of_year():
    for month in range(1, 13):
        for day in range(1, monthrange(2020, month)[1] + 1):
            yield (month, day)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_all_data(to_location):
    people = []
    urls = [f'https://famousbirthdays.com/{month_name[month].lower()}{day}.html' for month, day in all_days_of_year()]
    htmls = []
    # Fetch HTML pages
    for chunk in tqdm(chunks(urls, 63), 'Fetching HTML Pages'):
        htmls += get_responses_from_urls(chunk)

    # Parse names and image urls from htmls
    image_urls, names = [], []
    for html in tqdm(htmls, 'Parsing HTML'):
        new_urls, new_names = people_from_html(html)
        image_urls += new_urls
        names += new_names
    print([name for name in image_urls if not '.jpg' in name])
    all_images = []
    print('Fetching images')
    for chunk in tqdm(chunks(image_urls, 63), 'Fetching images', total=len(image_urls)//63):
        all_images += get_responses_from_urls(chunk)
    for i in range(len(all_images) - 1, -1, -1):
        if isinstance(all_images[i], Exception):
            del all_images[i]
            print(f'Removing {names[i]} from list')
            del names[i]
    np.save(to_location + 'images', all_images)
    np.save(to_location + 'names', names)


def get_responses_from_urls(urls: List[str]):
    responses = []
    responses = event_loop.run_until_complete(
        asyncio.gather(
            *(fetch(url) for url in urls),
            return_exceptions=True
        )
    )
    for idx, resp in enumerate(responses):
        if isinstance(resp, Exception):
            print(f'Failed to fetch {urls[idx]}.')
    return responses


async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if url[-4:] == '.jpg':
                nparr = np.frombuffer(await response.read(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            return await response.text()


def fetch_html(url):
    response = requests.get(url)
    assert response.status_code == 200, f'Failed to load {url}. Got status code {response.status_code}.'
    return requests.get(url).content


def people_from_html(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    people_elements: List[Tag] = soup.find_all('a', {'class': 'face person-item'})
    image_urls, names = [], []
    for el in people_elements:
        name_element = el.find('div', {'class': 'name'})
        name = re.split(r'[,(]', name_element.text.strip())[0].strip()  # Remove white characters and age
        styles = el.attrs['style']
        image_url = re.findall(r'url\((.*)\)', styles, re.MULTILINE)
        if image_url[0] != r'https://www.famousbirthdays.com/faces/large-default.jpg':
            image_urls.append(image_url[0])
            names.append(name)
    return image_urls, names


def load_image(image_url: str):
    response = requests.get(image_url)
    assert response.status_code == 200, f'Failed to load image at {image_url}. Got status code {response.status_code}.'
    image = np.asarray(bytearray(response.content), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


if __name__ == '__main__':
    get_all_data('../../data/famousbirthdays/')
