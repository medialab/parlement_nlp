import bs4
import casanova

from ural import format_url
from pathlib import Path
from os.path import join

FETCH_CSV = "./panorama/scrape/sommaires.csv"
OUTPUT_CSV = "./panorama/scrape/lois.csv"

def category_normalize(category):
    elements = category.split('-')
    elements = [e.strip() for e in elements]
    return '|'.join(elements)

def treat_page(row):
    label, url, path = row[0], row[1], row[7]
    soup = bs4.BeautifulSoup(Path(join("sommaires", path)).read_text(), "html.parser")
    for section in soup.select(".vp-container .paragraph--type--section"):
        h2 = section.select_one("h2")
        if not h2: continue
        
        category = h2.text.strip()
        category = category_normalize(category)
        for link in section.select(".fr-card .fr-card__content .fr-card__title a"):
            href = link.attrs["href"]
            href = format_url("https://vie-publique.fr", href)
            yield label, category, href

            
with open(OUTPUT_CSV, 'w') as export:
    writer = casanova.writer(export, ["label", "category", "url"])
    for row in casanova.reader(FETCH_CSV):
        for label, category, href in treat_page(row):
            writer.writerow([label, category, href])
