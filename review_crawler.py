import json
import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess
from twisted.internet import reactor
from scrapy.utils.project import get_project_settings
from crochet import setup, wait_for

setup()  # Setup Crochet (to run Scrapy inside Streamlit)

class TestSpider(scrapy.Spider):
    name = 'test'

    def __init__(self, url, short_url):
        self.url = url
        self.short_url = short_url
        self.start_urls = [self.url]
        self.all_reviews = []

    def parse(self, response):
        section = response.xpath("/html/body/div[1]/div/div/div/main/div/div[4]/section")

        for div in section.css('div[class="styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ"]'):
            body = div.xpath('article/section/div[2]/p/text()').extract()[0]

            review = {
                "body": body
            }
            self.all_reviews.append(review)

        next_page = section.xpath('div[contains(@class, "styles_pagination__6VmQv")]/nav/a[5]/@href').extract()
        
        if next_page:
            full_next_page_url = f"{self.short_url}{next_page[0]}"
            yield scrapy.Request(
                url=full_next_page_url,
                callback=self.parse
            )
        else:
            # Save data at the end
            with open('reviews.json', 'w') as f:
                json.dump(self.all_reviews, f)

@wait_for(timeout=60.0)
def get_reviews(url):
    short_url = url.split('/review')[0]
    process = CrawlerProcess(get_project_settings())
    crawler = process.create_crawler(TestSpider)
    
    process.crawl(crawler, url=url, short_url=short_url)
    
    # Prevent Scrapy from installing signal handlers
    from twisted.internet import reactor
    if not reactor.running:
        reactor.run(installSignalHandlers=False)
    
    # Read scraped data from json and return list
    with open('reviews.json') as f:
        reviews = json.load(f)
    
    return [r['body'] for r in reviews]

