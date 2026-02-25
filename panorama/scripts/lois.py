import bs4
import casanova
import tqdm

from ural import format_url
from pathlib import Path
from os.path import join

from html_to_markdown import convert

FETCH_CSV = "./panorama/scrape/lois.csv"
OUTPUT_CSV = "./panorama/lois.csv"

def treat_page(row):
    label, url, path = row[0], row[2], row[8]
    soup = bs4.BeautifulSoup(Path(join("./lois", path)).read_text(), "html.parser")
    
    title = soup.select_one("h1.fr-h1")
    if title:
        title = title.text.strip()

    headline = soup.select_one(".fr-text--lead.vp-page-chapo")
    if headline:
        headline = headline.text.strip()

    categories = soup.select_one(".vp-page-thematic ul li a")
    if categories:
        categories = [a.text.strip() for a in categories]
        categories = '|'.join(categories)

    tags = soup.select(".tagsBox ul.vp-tags-list li a")
    if tags:
        tags = [a.text.strip() for a in tags]
        tags = '|'.join(tags)

    content = soup.select_one(".vp-page-content .field--name-field-bloc-paragraphe")
    for node in content.select(".fr-card"):
        node.clear()
    content = convert(str(content))
    
    assert content

    yield label, url, title, headline, categories, tags, content
            
with open(OUTPUT_CSV, 'w') as export:
    writer = casanova.writer(export, [
        "label",
        "url",
        "title",
        "headline",
        "categories",
        "tags",
        "content",
    ])

    rows = [row for row in casanova.reader(FETCH_CSV)]
    for row in tqdm.tqdm(rows, total=len(rows)):
        for label, url, title, headline, categories, tags, content in treat_page(row):
            writer.writerow([
                label, url, title, headline, categories, tags, content
            ])
