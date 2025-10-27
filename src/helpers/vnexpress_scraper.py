import requests
from bs4 import BeautifulSoup

def scrape_vnexpress_article(url):
    """
    Scrapes the content, images, and captions from a VnExpress article.

    Args:
        url: The URL of the VnExpress article.

    Returns:
        A dictionary containing the title, content, and a list of images with their captions.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching the URL: {e}"}

    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('h1', class_='title-detail').get_text(strip=True) if soup.find('h1', class_='title-detail') else ""

    article_content = soup.find('article', class_='fck_detail')
    
    content = []
    if article_content:
        for p in article_content.find_all('p', class_='Normal'):
            content.append(p.get_text(strip=True))

    images = []
    if article_content:
        for figure in article_content.find_all('figure', class_='tplCaption'):
            img_tag = figure.find('img')
            caption_tag = figure.find('figcaption')
            
            if img_tag and 'data-src' in img_tag.attrs:
                image_src = img_tag['data-src']
                caption = caption_tag.get_text(strip=True) if caption_tag else ""
                images.append({'src': image_src, 'caption': caption})

    return {
        'title': title,
        'content': "\n".join(content),
        'images': images
    }

if __name__ == '__main__':
    url = "https://vnexpress.net/napas-gioi-thieu-cong-nghe-thanh-toan-qua-nhan-dien-guong-mat-4759393.html"
    scraped_data = scrape_vnexpress_article(url)

    if "error" in scraped_data:
        print(scraped_data["error"])
    else:
        print(f"Title: {scraped_data['title']}\n")
        print(f"Content:\n{scraped_data['content']}\n")
        for image in scraped_data['images']:
            print(f"Image Source: {image['src']}")
            print(f"Image Caption: {image['caption']}\n")
